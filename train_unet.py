"""U-Net training script.

This script trains a U-Net for multi-class segmentation and writes all run
artifacts under ``checkpoints/<date>[_vN]/`` to keep experiments reproducible.

Artifacts:
  - ``history.csv``: per-epoch training loss and validation metrics
  - ``summary.json``: hyperparameters, configuration, and best epoch metrics
  - ``best_model.pth``: checkpoint selected by highest validation Dice
  - ``checkpoint_epoch*.pth``: periodic checkpoints for recovery/analysis

The script is offline-friendly (no external tracking services required).

Examples:
  python train_unet.py
  python train_unet.py --use-preprocessed --preprocessed-dir ./preprocessed_data
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
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluate import evaluate
from unet import UNet
from utils.dice_score import WeightedCEDiceLoss
from preprocess import CatheterPreprocessingDataset, PreprocessedDataset


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

# Run directory with date and automatic versioning (YYYYMMDD, then _v2, _v3, ...)
def make_run_dir(base: Path) -> Path:
    """
    Example: checkpoints/20260308 -> checkpoints/20260308_v2 -> checkpoints/20260308_v3 -> ...
    """
    date_str = datetime.datetime.now().strftime('%Y%m%d')
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


# ── Training Function ──────────────────────────────────────────────────────────
def train_model(
        model,
        device,
        train_dataset,
        val_dataset,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        save_checkpoint: bool = True,
        amp: bool = False,
        weight_decay: float = 1e-8,
        gradient_clipping: float = 1.0,
        experiment_config: dict = None,
):
    n_train = len(train_dataset)
    n_val   = len(val_dataset)

    # Create specific directory for this run
    run_dir   = make_run_dir(Path('./checkpoints'))
    csv_path  = run_dir / 'history.csv'
    json_path = run_dir / 'summary.json'
    logging.info(f'Run outputs directory: {run_dir}')

    # CSV holds one row per epoch (easy plotting and paper tables).
    csv_fieldnames = [
        'epoch',
        'train_loss',
        # Main metrics
        'val_dice', 'val_miou', 'val_jaccard', 'val_accuracy',
        # Foreground accuracy without background inflation
        'val_foreground_accuracy',
        # Class-specific
        'val_dice_catheter', 'val_dice_guidewire',
        'val_iou_catheter',  'val_iou_guidewire',
        'learning_rate',
    ]
    csv_file   = open(csv_path, 'w', newline='', encoding='utf-8')
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames)
    csv_writer.writeheader()

    # Initialize summary.json
    summary = {
        'run_dir': str(run_dir),
        'training': {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'gradient_clipping': gradient_clipping,
            'amp': amp,
            'optimizer': 'AdamW',
            'scheduler': 'CosineAnnealingLR(T_max=epochs, eta_min=1e-5)',
            'loss': 'WeightedCEDiceLoss(CE*0.3+Dice*0.7, w=[0.05,1.5,4.0])',
            'n_train': n_train,
            'n_val': n_val,
            'device': str(device),
        },
        'extra': experiment_config or {},
        'best_epoch': None,
        'best_val_dice': None,
        'best_val_miou': None,
        'best_val_jaccard': None,
        'best_val_accuracy': None,
        'total_train_time': None,
        'history': [],
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
        prefetch_factor=8,
    )
    train_loader = DataLoader(train_dataset, shuffle=True,  drop_last=True, **loader_args)
    val_loader   = DataLoader(val_dataset,   shuffle=False, drop_last=True, **loader_args)

    optimizer   = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler   = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    grad_scaler = torch.amp.GradScaler('cuda', enabled=amp)

    # WeightedCEDiceLoss: 0.3*CE(w=[0.05,1.5,4.0]) + 0.7*DiceLoss(weighted)
    # bg=0.05 penalises false-positive noise; catheter=1.5 compensates v5 drop;
    # guidewire=4.0 slightly relaxed; Dice weight raised for contour purity
    if model.n_classes == 1:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = WeightedCEDiceLoss(class_weights=[0.05, 1.5, 4.0])

    best_dice   = 0.0
    epoch_times = []
    train_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss       = 0
        epoch_start_time = time.time()

        # tqdm progress bar
        pbar = tqdm(
            train_loader,
            desc=f'Epoch {epoch:>3}/{epochs}',
            unit='batch',
            dynamic_ncols=True,
            leave=False,          # Clear progress bar after epoch
        )
        for batch in pbar:
            images, true_masks = batch['image'], batch['mask']
            bs = images.shape[0]

            images = images.to(device=device, dtype=torch.float32,
                               memory_format=torch.channels_last, non_blocking=True)
            true_masks = true_masks.to(device=device, dtype=torch.long, non_blocking=True)

            with torch.autocast(device.type, enabled=amp):
                masks_pred = model(images)
                if model.n_classes == 1:
                    loss = criterion(masks_pred.squeeze(1), true_masks.float())
                else:
                    # WeightedCEDiceLoss calculates CE + Dice internally
                    loss = criterion(masks_pred, true_masks)

            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            grad_scaler.step(optimizer)
            grad_scaler.update()

            epoch_loss += loss.item()
            # Show batch loss and speed
            elapsed = time.time() - epoch_start_time
            imgs_done = pbar.n * batch_size  # Approximated processed count
            speed = imgs_done / elapsed if elapsed > 0 else 0
            pbar.set_postfix(loss=f'{loss.item():.4f}', speed=f'{speed:.0f}img/s')

        pbar.close()

        epoch_time     = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        avg_loss       = epoch_loss / max(1, len(train_loader))
        current_lr     = optimizer.param_groups[0]['lr']
        throughput     = n_train / epoch_time

        # Full curve evaluation suitable for paper
        metrics    = evaluate(model, val_loader, device, amp)
        dice_score = metrics['dice_score']
        scheduler.step()   # CosineAnnealingLR step per epoch

        # Write to CSV
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
            'learning_rate':           current_lr,
        }
        csv_writer.writerow(row)
        csv_file.flush()
        summary['history'].append(row)

        # Save best model
        best_marker = ''
        if dice_score > best_dice:
            best_dice = dice_score
            torch.save(model.state_dict(), str(run_dir / 'best_model.pth'))
            summary['best_epoch']        = epoch
            summary['best_val_dice']     = round(metrics['dice_score'], 6)
            summary['best_val_miou']     = round(metrics['miou'],       6)
            summary['best_val_jaccard']  = round(metrics['jaccard'],    6)
            summary['best_val_accuracy'] = round(metrics['accuracy'],   6)
            best_marker = ' ★'

        remaining_h = (epochs - epoch) * avg_epoch_time / 3600

        #  Validate foreground prediction ratio
        fg_ratio = metrics.get('fg_pred_ratio', 0)
        fg_warning = ' Warning: FG_LOW!' if fg_ratio < 0.3 else ''

        # Compact logger per epoch
        logging.info(
            f'[{epoch:>3}/{epochs}] '
            f'loss={avg_loss:.4f} | '
            f'dice={metrics["dice_score"]:.4f} miou={metrics["miou"]:.4f} | '
            f'cat={metrics.get("dice_catheter", 0):.4f} gw={metrics.get("dice_guidewire", 0):.4f} | '
            f'fg={fg_ratio:.2f} | '  #  Show foreground ratio
            f'lr={current_lr:.1e} | '
            f'{throughput:.0f}img/s | '
            f'eta={remaining_h:.1f}h'
            f'{best_marker}{fg_warning}'
        )

        # Save checkpoint periodically
        if save_checkpoint and (epoch % 10 == 0 or epoch == epochs):
            torch.save(model.state_dict(),
                       str(run_dir / f'checkpoint_epoch{epoch}.pth'))
            logging.info(f'    checkpoint_epoch{epoch}.pth saved')

    # Training completed: write summary.json
    total_s = time.time() - train_start
    summary['total_train_time'] = str(datetime.timedelta(seconds=int(total_s)))

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    csv_file.close()

    logging.info(f'\n{"="*60}')
    logging.info(f'Metrics data saved to: {run_dir}')
    logging.info(f'   history.csv    - loss / val metrics / lr curve (one row per epoch)')
    logging.info(f'   summary.json   - Hyperparameters + Best results')
    logging.info(f'   best_model.pth - epoch {summary["best_epoch"]}')
    logging.info(f'   Best Dice={summary["best_val_dice"]}  mIoU={summary["best_val_miou"]}')
    logging.info(f'{"="*60}')


# ── Main function ─────────────────────────────────────────────────────────────
def main():
    script_start_time = time.time()

    # ========== Training configuration ==========
    epochs        = 50
    batch_size    = 12       # L40S 48GB VRAM
    learning_rate = 5e-5
    classes       = 3
    bilinear      = False
    base_channels = 32

    load_model       = False
    val_percent      = 0.1
    amp              = True
    use_preprocessed = False  # Bypass data aug if true
    preprocessed_dir = './preprocessed_data'
    data_dir         = None
    # ==============================

    parser = argparse.ArgumentParser(description='Train UNet model')
    parser.add_argument('--use-preprocessed', action='store_true')
    parser.add_argument('--preprocessed-dir', type=str, default=preprocessed_dir)
    parser.add_argument('--batch-size',    '-b', type=int,   default=batch_size)
    parser.add_argument('--epochs',        '-e', type=int,   default=epochs)
    parser.add_argument('--lr',                  type=float, default=learning_rate)
    parser.add_argument('--base-channels',       type=int,   default=base_channels)
    parser.add_argument('--bilinear',            action='store_true', default=bilinear)
    args = parser.parse_args()

    use_preprocessed = args.use_preprocessed
    preprocessed_dir = args.preprocessed_dir
    batch_size       = args.batch_size
    epochs           = args.epochs
    learning_rate    = args.lr
    base_channels    = args.base_channels
    bilinear         = args.bilinear

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    if torch.cuda.is_available():
        logging.info(f'GPU: {torch.cuda.get_device_name(0)}')

    model = UNet(n_channels=3, n_classes=classes, bilinear=bilinear, base_channels=base_channels)

    if load_model and isinstance(load_model, str):
        state_dict = torch.load(load_model, map_location=device)
        if 'mask_values' in state_dict:
            del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {load_model}')

    model.to(device=device)

    # ── Dataset Loading ──
    if use_preprocessed:
        logging.info('=' * 60)
        logging.info('Using preprocessed dataset (Fast mode)')
        logging.info('=' * 60)
        try:
            dataset = PreprocessedDataset(preprocessed_dir=preprocessed_dir)
            from torch.utils.data import random_split as rs
            n_val_pre   = int(len(dataset) * val_percent)
            n_train_pre = len(dataset) - n_val_pre
            train_dataset, val_dataset = rs(
                dataset, [n_train_pre, n_val_pre],
                generator=torch.Generator().manual_seed(0)
            )
        except RuntimeError as e:
            logging.error(str(e))
            logging.error('\nPlease run: python preprocess_to_pt.py')
            return
    else:
        logging.info('=' * 60)
        logging.info('Using original dataset')
        logging.info('=' * 60)

        if data_dir is None:
            data_dir = os.getenv('DATA_DIR', './data')

        BASE = Path(data_dir)
        if not BASE.exists():
            for cand in [Path('./segmentation'), Path('./segmentation/segmentation')]:
                if (cand / 'animal_train' / 'images').exists():
                    BASE = cand
                    logging.info(f'Found data at: {BASE}')
                    break

        TRAIN_DIRS = [
            (BASE / 'animal_train'  / 'images', BASE / 'animal_train'  / 'masks'),
            (BASE / 'phantom_train' / 'images', BASE / 'phantom_train' / 'masks'),
        ]

        image_paths, mask_paths = [], []
        for dir_img, dir_mask in TRAIN_DIRS:
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
        n_val         = int(len(indices) * val_percent)
        val_indices   = indices[:n_val]
        train_indices = indices[n_val:]

        common_kwargs = dict(
            target_size=512,
            use_clahe=False,
            mean=[0.4172, 0.4167, 0.4173],
            std =[0.2436, 0.2435, 0.2436],
        )
        train_dataset = CatheterPreprocessingDataset(
            image_paths=[image_paths[i] for i in train_indices],
            mask_paths =[mask_paths[i]  for i in train_indices],
            mode='train', **common_kwargs
        )
        val_dataset = CatheterPreprocessingDataset(
            image_paths=[image_paths[i] for i in val_indices],
            mask_paths =[mask_paths[i]  for i in val_indices],
            mode='val', **common_kwargs
        )

    logging.info('Starting training...')
    train_model(
        model=model,
        device=device,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        amp=amp,
        experiment_config={
            'base_channels': base_channels,
            'bilinear': bilinear,
            'val_percent': val_percent,
            'preprocessing': 'mixed(pad<=512, fg_centered_crop>512)',
            'use_clahe': False,
            'mean': [0.4172, 0.4167, 0.4173],
            'std':  [0.2436, 0.2435, 0.2436],
            'loss': 'WeightedCEDiceLoss(CE*0.3+Dice*0.7, w=[0.05,1.5,4.0])',
            'class_weights': [0.05, 1.5, 4.0],
        }
    )

    total_s = time.time() - script_start_time
    logging.info(f'Total time: {datetime.timedelta(seconds=int(total_s))}')


if __name__ == '__main__':
    main()
