# coding: utf-8
"""TransUNet training script.

This script mirrors the main UNet training pipeline, with the following changes:
  - Model: TransUNet (R50-ViT-B/16-style hybrid)
  - Outputs: saved under ``checkpoints_transunet/<date>[_vN]/``

It is intentionally offline-friendly (no wandb dependency).

Examples:
  python train_transunet.py
  python train_transunet.py --batch-size 12 --epochs 50
  python train_transunet.py --batch-size 12 --hidden-size 512 --num-layers 8 --num-heads 8
  python train_transunet.py --use-preprocessed --preprocessed-dir ./preprocessed_data
"""

import argparse
import csv
import json
import logging
import os
import random
import time
import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluate import evaluate
from transunet import TransUNet
from utils.dice_score import WeightedCEDiceLoss
from preprocess import CatheterPreprocessingDataset, PreprocessedDataset


# Create a run directory using a date prefix, automatically adding a version
# suffix when the folder already exists.
def make_run_dir(base: Path) -> Path:
    date_str  = datetime.datetime.now().strftime('%Y%m%d')
    candidate = base / date_str
    if not candidate.exists():
        candidate.mkdir(parents=True)
        return candidate
    version = 2
    while True:
        candidate = base / f'{date_str}_v{version}'
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
        version += 1


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def train_model(
        model,
        device,
        train_dataset,
        val_dataset,
        epochs:            int   = 50,
        batch_size:        int   = 8,
        learning_rate:     float = 5e-5,
        save_checkpoint:   bool  = True,
        amp:               bool  = True,
        weight_decay:      float = 1e-4,
        gradient_clipping: float = 1.0,
        experiment_config: dict  = None,
):
    n_train = len(train_dataset)
    n_val   = len(val_dataset)

    # TransUNet results go to a separate folder so UNet checkpoints are untouched
    run_dir   = make_run_dir(Path('./checkpoints_transunet'))
    csv_path  = run_dir / 'history.csv'
    json_path = run_dir / 'summary.json'
    logging.info('Run directory: %s', run_dir)

    csv_fieldnames = [
        'epoch', 'train_loss',
        'val_dice', 'val_miou', 'val_jaccard', 'val_accuracy',
        'val_foreground_accuracy',
        'val_dice_catheter', 'val_dice_guidewire',
        'val_iou_catheter',  'val_iou_guidewire', 'fg_pred_ratio',
        # Composite metric used for selecting the best checkpoint (foreground-oriented).
        'val_selection_score',
        'learning_rate',
    ]
    csv_file   = open(csv_path, 'w', newline='', encoding='utf-8')
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames)
    csv_writer.writeheader()

    # Loss configuration is centralized here to keep logs/summary consistent.
    ce_weight = 0.4
    dice_weight = 0.6
    class_weights = [0.12, 1.5, 3.5]

    loss_desc = f'WeightedCEDiceLoss(CE*{ce_weight}+Dice*{dice_weight}, w={class_weights})'

    summary = {
        'run_dir':  str(run_dir),
        'model':    'TransUNet (R50-ViT-B/16)',
        'training': {
            'epochs':            epochs,
            'batch_size':        batch_size,
            'learning_rate':     learning_rate,
            'weight_decay':      weight_decay,
            'gradient_clipping': gradient_clipping,
            'amp':               amp,
            'optimizer':         'AdamW',
            'scheduler':         'CosineAnnealingLR(T_max=epochs, eta_min=1e-5)',
            'loss':              loss_desc,
            'n_train':           n_train,
            'n_val':             n_val,
            'device':            str(device),
        },
        'extra':             experiment_config or {},
        'best_epoch':        None,
        'best_val_dice':     None,
        'best_val_miou':     None,
        'best_val_jaccard':  None,
        'best_val_accuracy': None,
        # Best-checkpoint selection criterion (see `selection_score` below).
        'best_selection_score': None,
        'best_fg_pred_ratio':   None,
        'total_train_time':  None,
        'history':           [],
    }
    if torch.cuda.is_available():
        summary['training']['gpu'] = torch.cuda.get_device_name(0)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Device:          {device.type}
        Mixed Precision: {amp}
        Output:          {run_dir}
    ''')

    loader_args = dict(
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    train_loader = DataLoader(train_dataset, shuffle=True,  drop_last=True,  **loader_args)
    val_loader   = DataLoader(val_dataset,   shuffle=False, drop_last=True,  **loader_args)

    optimizer   = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler   = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    grad_scaler = torch.amp.GradScaler('cuda', enabled=amp)

    # TransUNet can be biased towards background. We therefore use a weighted
    # CE+Dice loss to counter class imbalance.
    criterion = nn.BCEWithLogitsLoss() if model.n_classes == 1 \
        else WeightedCEDiceLoss(class_weights=class_weights, ce_weight=ce_weight, dice_weight=dice_weight)

    best_dice = 0.0
    best_selection_score = float('-inf')
    epoch_times = []
    train_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss       = 0
        epoch_start_time = time.time()

        # Reset CUDA peak stats at the start of each epoch (so the value is per-epoch)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()

        pbar = tqdm(
            train_loader,
            desc=f'Epoch {epoch:>3}/{epochs}',
            unit='batch',
            dynamic_ncols=True,
            leave=False,
        )
        for batch in pbar:
            images, true_masks = batch['image'], batch['mask']
            images     = images.to(device=device, dtype=torch.float32,
                                   memory_format=torch.channels_last, non_blocking=True)
            true_masks = true_masks.to(device=device, dtype=torch.long, non_blocking=True)

            with torch.autocast(device.type, enabled=amp):
                masks_pred = model(images)
                loss = criterion(masks_pred.squeeze(1), true_masks.float()) \
                    if model.n_classes == 1 else criterion(masks_pred, true_masks)

            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            grad_scaler.step(optimizer)
            grad_scaler.update()

            epoch_loss += loss.item()
            elapsed    = time.time() - epoch_start_time
            speed      = (pbar.n * batch_size) / elapsed if elapsed > 0 else 0
            pbar.set_postfix(loss=f'{loss.item():.4f}', speed=f'{speed:.0f}img/s')

        pbar.close()

        epoch_time     = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        avg_loss       = epoch_loss / max(1, len(train_loader))
        current_lr     = optimizer.param_groups[0]['lr']
        throughput     = n_train / epoch_time

        metrics    = evaluate(model, val_loader, device, amp)
        dice_score = metrics['dice_score']

        # Foreground-oriented model selection (accuracy can be misleading when
        # background dominates).
        iou_cat = float(metrics.get('iou_catheter') or 0.0)
        iou_gw  = float(metrics.get('iou_guidewire') or 0.0)
        fg_pred_ratio = float(metrics.get('fg_pred_ratio') or 0.0)

        # Penalty term: the further `fg_pred_ratio` deviates from 1.0, the more
        # we penalize the score (>1 tends to false positives, <1 tends to misses).
        selection_score = (
            0.5 * float(dice_score) +
            0.25 * iou_cat +
            0.25 * iou_gw -
            0.10 * abs(fg_pred_ratio - 1.0)
        )

        scheduler.step()

        row = {
            'epoch':                   epoch,
            'train_loss':              round(avg_loss,              6),
            'val_dice':                round(metrics['dice_score'], 6),
            'val_miou':                round(metrics['miou'],       6),
            'val_jaccard':             round(metrics['jaccard'],    6),
            'val_accuracy':            round(metrics['accuracy'],   6),
            'val_foreground_accuracy': round(metrics.get('foreground_accuracy') or 0, 6),
            'val_dice_catheter':       round(metrics.get('dice_catheter')  or 0, 6),
            'val_dice_guidewire':      round(metrics.get('dice_guidewire') or 0, 6),
            'val_iou_catheter':        round(metrics.get('iou_catheter')   or 0, 6),
            'val_iou_guidewire':       round(metrics.get('iou_guidewire')  or 0, 6),
            'fg_pred_ratio':           round(fg_pred_ratio, 6),
            'val_selection_score':     round(float(selection_score), 6),
            'learning_rate':           float(current_lr),
        }

        # Ensure we never write unexpected keys (keeps runs robust when adding metrics)
        if csv_writer is not None and hasattr(csv_writer, 'fieldnames'):
            row = {k: row.get(k, '') for k in csv_writer.fieldnames}

        csv_writer.writerow(row)
        csv_file.flush()
        summary['history'].append(row)

        best_marker = ''
        # Keep `best_dice` for reference, but do not use it for saving best_model.
        if dice_score > best_dice:
            best_dice = float(dice_score)

        # Select best_model using `selection_score`.
        if float(selection_score) > float(best_selection_score):
            best_selection_score = float(selection_score)
            torch.save(model.state_dict(), str(run_dir / 'best_model.pth'))
            summary['best_epoch']            = epoch
            summary['best_val_dice']         = round(metrics['dice_score'], 6)
            summary['best_val_miou']         = round(metrics['miou'],       6)
            summary['best_val_jaccard']      = round(metrics['jaccard'],    6)
            summary['best_val_accuracy']     = round(metrics['accuracy'],   6)
            summary['best_selection_score']  = round(float(selection_score), 6)
            summary['best_fg_pred_ratio']    = round(float(fg_pred_ratio), 6)
            best_marker = ' ★'

        remaining_h = (epochs - epoch) * avg_epoch_time / 3600
        logging.info(
            f'[ {epoch:>2}/{epochs}] '
            f'loss={avg_loss:.4f} | '
            f'dice={metrics["dice_score"]:.4f} miou={metrics["miou"]:.4f} | '
            f'cat={metrics.get("dice_catheter", 0):.4f} gw={metrics.get("dice_guidewire", 0):.4f} | '
            f'fg_ratio={fg_pred_ratio:.2f} sel={selection_score:.4f} | '
            f'lr={current_lr:.1e} | {throughput:.0f}img/s | eta={remaining_h:.1f}h'
            f'{best_marker}'
        )

        # Log peak GPU memory usage for the epoch (GB).
        if device.type == 'cuda':
            max_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            logging.info('[Epoch %d] Max GPU memory allocated: %.2f GB', epoch, max_mem_gb)

            # Reset peak stats so the next epoch reports its own peak.
            torch.cuda.reset_peak_memory_stats()

        if save_checkpoint and (epoch % 10 == 0 or epoch == epochs):
            torch.save(model.state_dict(), str(run_dir / f'checkpoint_epoch{epoch}.pth'))
            logging.info('Saved checkpoint: checkpoint_epoch%d.pth', epoch)

    total_s = time.time() - train_start
    summary['total_train_time'] = str(datetime.timedelta(seconds=int(total_s)))
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    csv_file.close()

    logging.info('\n%s', '=' * 60)
    logging.info('Run artifacts saved to: %s', run_dir)
    logging.info('  history.csv    - training/validation metrics per epoch')
    logging.info('  summary.json   - hyperparameters and best metrics')
    logging.info('  best_model.pth - selected epoch %s', summary["best_epoch"])
    logging.info('Best Dice=%s  mIoU=%s', summary["best_val_dice"], summary["best_val_miou"])
    logging.info('%s', '=' * 60)


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
def main():
    script_start = time.time()

    # Default training configuration (kept consistent with the main pipeline).
    epochs        = 50
    batch_size    = 8        # TransUNet uses more VRAM; adjust if needed.
    learning_rate = 5e-5     # pre-3.11_v3 default (more stable)
    classes       = 3
    val_percent   = 0.1
    amp           = True
    use_preprocessed = False
    preprocessed_dir = './preprocessed_data'
    data_dir         = None
    # =======================================================

    parser = argparse.ArgumentParser(
        description='Train TransUNet',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--use-preprocessed', action='store_true')
    parser.add_argument('--preprocessed-dir', type=str,   default=preprocessed_dir)
    parser.add_argument('--batch-size', '-b', type=int,   default=batch_size)
    parser.add_argument('--epochs',     '-e', type=int,   default=epochs)
    parser.add_argument('--lr',               type=float, default=learning_rate)
    parser.add_argument('--data-dir',         type=str,   default=None)
    parser.add_argument('--pretrained',       action='store_true', help='Use pretrained weights')

    # ── Structural parameters (analogous to UNet --base-channels) ──────────
    arch = parser.add_argument_group('model architecture')
    arch.add_argument('--preset', type=str, default='vit-s',
                      choices=['vit-b', 'vit-s'],
                      help='Named preset: vit-b (~118M) or vit-s (~35M, default, ~3x faster)')
    arch.add_argument('--hidden-size',      type=int,   default=768,
                      help='Transformer hidden dim  (768=ViT-B, 384=ViT-S)')
    arch.add_argument('--num-layers',       type=int,   default=12,
                      help='Transformer depth       (12=ViT-B,  6=ViT-S)')
    arch.add_argument('--num-heads',        type=int,   default=12,
                      help='Attention heads         (12=ViT-B,  6=ViT-S)')
    arch.add_argument('--decoder-channels', type=int,   default=None, nargs=4,
                      metavar=('D0','D1','D2','D3'),
                      help='4 decoder stage widths  (default: 384 192 96 48 for ViT-B)')
    args = parser.parse_args()

    use_preprocessed = args.use_preprocessed
    preprocessed_dir = args.preprocessed_dir
    batch_size       = args.batch_size
    epochs           = args.epochs
    learning_rate    = args.lr
    if args.data_dir:
        data_dir = args.data_dir

    # --preset overrides individual arch flags when given
    if args.preset is not None:
        preset_cfg       = TransUNet.PRESETS[args.preset]
        hidden_size      = preset_cfg['hidden_size']
        num_heads        = preset_cfg['num_heads']
        num_layers       = preset_cfg['num_layers']
        decoder_channels = preset_cfg['decoder_channels']
    else:
        hidden_size      = args.hidden_size
        num_heads        = args.num_heads
        num_layers       = args.num_layers
        decoder_channels = tuple(args.decoder_channels) if args.decoder_channels \
                           else TransUNet.PRESETS['vit-b']['decoder_channels']

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    if torch.cuda.is_available():
        logging.info(f'GPU: {torch.cuda.get_device_name(0)}')

    try:
        model = TransUNet(
            n_channels=3, n_classes=classes,
            hidden_size=hidden_size, num_heads=num_heads, num_layers=num_layers,
            decoder_channels=decoder_channels,
            pretrained=args.pretrained
        )
    except TypeError:
        logging.warning(
            'The current TransUNet implementation does not accept the `pretrained` argument; '
            'falling back to standard initialization.'
        )
        model = TransUNet(
            n_channels=3, n_classes=classes,
            hidden_size=hidden_size, num_heads=num_heads, num_layers=num_layers,
            decoder_channels=decoder_channels,
        )

    param_m = sum(p.numel() for p in model.parameters()) / 1e6
    logging.info(
        f'TransUNet  hidden={hidden_size}  heads={num_heads}  layers={num_layers}'
        f'  decoder={decoder_channels}  params={param_m:.1f}M'
    )
    model.to(device=device)

    # ── Dataset ────────────────────────────────────────────────────────────────
    if use_preprocessed:
        logging.info('%s', '=' * 60)
        logging.info('Using preprocessed dataset')
        logging.info('%s', '=' * 60)
        try:
            dataset = PreprocessedDataset(preprocessed_dir=preprocessed_dir)
            from torch.utils.data import random_split as rs
            n_val_n   = int(len(dataset) * val_percent)
            n_train_n = len(dataset) - n_val_n
            train_dataset, val_dataset = rs(
                dataset, [n_train_n, n_val_n],
                generator=torch.Generator().manual_seed(0),
            )
        except RuntimeError as e:
            logging.error(str(e))
            return
    else:
        logging.info('%s', '=' * 60)
        logging.info('Using raw dataset')
        logging.info('%s', '=' * 60)

        if data_dir is None:
            data_dir = os.getenv('DATA_DIR', './segmentation')

        BASE = Path(data_dir)
        if not BASE.exists():
            for cand in [Path('./segmentation'), Path('../unet/segmentation')]:
                if (cand / 'animal_train' / 'images').exists():
                    BASE = cand
                    logging.info(f'Found data at: {BASE}')
                    break

        image_paths, mask_paths = [], []
        for sub in ['animal_train', 'phantom_train']:
            dir_img  = BASE / sub / 'images'
            dir_mask = BASE / sub / 'masks'
            if not dir_img.exists():
                continue
            for img_p in sorted(
                list(dir_img.glob('*.png')) +
                list(dir_img.glob('*.jpg')) +
                list(dir_img.glob('*.jpeg'))
            ):
                mask_p = dir_mask / (img_p.stem + '.npy')
                if mask_p.exists():
                    image_paths.append(str(img_p))
                    mask_paths.append(str(mask_p))

        logging.info(f'Total image/mask pairs: {len(image_paths)}')

        indices = list(range(len(image_paths)))
        random.Random(0).shuffle(indices)
        n_val_n       = int(len(indices) * val_percent)
        val_indices   = indices[:n_val_n]
        train_indices = indices[n_val_n:]

        kw = dict(target_size=512, use_clahe=False,
                  mean=[0.4172, 0.4167, 0.4173], std=[0.2436, 0.2435, 0.2436])
        train_dataset = CatheterPreprocessingDataset(
            image_paths=[image_paths[i] for i in train_indices],
            mask_paths =[mask_paths[i]  for i in train_indices],
            mode='train', **kw)
        val_dataset = CatheterPreprocessingDataset(
            image_paths=[image_paths[i] for i in val_indices],
            mask_paths =[mask_paths[i]  for i in val_indices],
            mode='val', **kw)

    train_model(
        model=model, device=device,
        train_dataset=train_dataset, val_dataset=val_dataset,
        epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, amp=amp,
        experiment_config={
            'model': 'TransUNet',
            'preset': args.preset or 'custom',
            'hidden_size': hidden_size,
            'num_heads': num_heads,
            'num_layers': num_layers,
            'decoder_channels': list(decoder_channels),
            'val_percent': val_percent,
            'preprocessing': 'mixed(pad<=512, fg_centered_crop>512)',
            # Loss configuration is managed inside train_model() to avoid divergence
            # between experiment config and the actual criterion.
        }
    )

    logging.info(f'Total time: {datetime.timedelta(seconds=int(time.time() - script_start))}')


if __name__ == '__main__':
    main()

