# coding: utf-8
"""
SwinUNet training script (keeps existing TransUNet pipeline untouched).

Usage:
  python train_swinunet.py
  --train-subset phantom
  python train_swinunet.py --batch-size 8 --epochs 50
  python train_swinunet.py --model-name swinv2_tiny_window8_256 --decoder-channels 256 128 64 32 --epochs 50
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
import numpy as np

from evaluate import evaluate
from swinunet import SwinUNet
from utils.dice_score import WeightedCEDiceLoss
from utils.dice_score import TverskyLoss, FocalTverskyLoss
from utils.dice_score import SoftCLDiceLoss
from utils.dice_score import BCEDiceCLDiceLoss
from preprocess import CatheterPreprocessingDataset, PreprocessedDataset


# -- run dir --------------------------------------------------------------

def make_run_dir(base: Path) -> Path:
    """Create a new run folder under `base`.

    We name runs by date (YYYYMMDD). If the folder already exists we append
    _v2, _v3, ... so previous experiments are not overwritten.
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


def _mask_has_all_classes(mask_path: Path) -> bool:
    """Return True if a saved *.npy mask contains labels {0,1,2}.

    Some of our raw masks can miss a foreground class (e.g. only catheter or
    only guidewire). For certain experiments we keep only images that contain
    both foreground classes.
    """
    try:
        mask = np.load(mask_path)
    except Exception:
        return False
    labels = set(np.unique(mask).tolist())
    return {0, 1, 2}.issubset(labels)


def _filter_pairs_all_classes(image_paths, mask_paths):
    """Filter (image, mask) pairs to those whose mask has all classes."""
    kept_images, kept_masks = [], []
    dropped = 0
    for img_p, mask_p in zip(image_paths, mask_paths):
        if _mask_has_all_classes(Path(mask_p)):
            kept_images.append(img_p)
            kept_masks.append(mask_p)
        else:
            dropped += 1
    return kept_images, kept_masks, dropped


def _resolve_train_subdirs(train_subset: str):
    """Map CLI argument `--train-subset` to concrete folder names."""
    key = (train_subset or 'both').strip().lower()
    alias = {
        'both': ['animal_train', 'phantom_train'],
        'all': ['animal_train', 'phantom_train'],
        'animal': ['animal_train'],
        'animal_train': ['animal_train'],
        'phantom': ['phantom_train'],
        'phantom_train': ['phantom_train'],
    }
    if key not in alias:
        raise ValueError(
            f'Invalid --train-subset={train_subset!r}. '
            f'Use one of: both, all, animal, animal_train, phantom, phantom_train'
        )
    return alias[key]

# -- training -------------------------------------------------------------

def train_model(
    model,
    device,
    train_dataset,
    val_dataset,
    epochs: int = 50,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    save_checkpoint: bool = True,
    amp: bool = True,
    weight_decay: float = 0.01,
    gradient_clipping: float = 1.0,
    experiment_config: dict = None,
    # NEW loss config
    loss_name: str = 'wce_dice',
    tversky_alpha: float = 0.7,
    tversky_beta: float = 0.3,
    focal_tversky_gamma: float = 1.33,
    # NEW: clDice config
    cldice_iters: int = 10,
    cldice_weight: float = 1.0,
    # NEW: BCE+Dice+clDice combo weights
    ce_weight: float = 0.4,
    dice_weight: float = 0.6,
    bce_dice_cldice_weight: float = 0.5,
    class_weights: tuple = (0.05, 2.0, 5.0),
    # NEW foreground-crop config (recorded in the run summary)
    fg_crop: bool = True,
    fg_crop_sizes: tuple = (256, 320, 384),
    # NEW intensity augmentation config (recorded in the run summary)
    intensity_aug: bool = True,
):
    """Main training loop.

    Notes:
    - Supports multiple loss functions (Dice/CE variants, Tversky, clDice).
    - Supports optional deep supervision: the model can return
      (main_logits, [aux_logits_1, aux_logits_2, ...]) and we add a weighted
      auxiliary loss for each head.
    - Keeps logging/CSV/JSON summaries so runs are easy to compare later.
    """
    n_train = len(train_dataset)
    n_val = len(val_dataset)

    run_dir = make_run_dir(Path('checkpoints/checkpoints_swinunet'))
    csv_path = run_dir / 'history.csv'
    json_path = run_dir / 'summary.json'
    logging.info(f'Run dir: {run_dir}')

    csv_fieldnames = [
        'epoch', 'train_loss',
        'val_dice', 'val_miou', 'val_jaccard', 'val_accuracy',
        'val_foreground_accuracy',
        'val_dice_catheter', 'val_dice_guidewire',
        'val_iou_catheter', 'val_iou_guidewire', 'fg_pred_ratio',
        'val_selection_score',
        'learning_rate',
    ]
    csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames)
    csv_writer.writeheader()

    class_weights = list(class_weights)

    # Human-readable loss description (saved into summary.json).
    # This is useful when you come back to a run weeks later.

    if loss_name == 'tversky':
        loss_desc = f'TverskyLoss(alpha={tversky_alpha}, beta={tversky_beta}, w={class_weights}, ignore_bg=True)'
    elif loss_name == 'focal_tversky':
        loss_desc = f'FocalTverskyLoss(alpha={tversky_alpha}, beta={tversky_beta}, gamma={focal_tversky_gamma}, w={class_weights}, ignore_bg=True)'
    elif loss_name == 'cldice':
        loss_desc = f'SoftCLDiceLoss(iters={cldice_iters}, w={class_weights}, fg_only=True)'
    elif loss_name == 'tversky_cldice':
        loss_desc = (
            f'TverskyLoss(alpha={tversky_alpha}, beta={tversky_beta}) + '
            f'{cldice_weight}*SoftCLDiceLoss(iters={cldice_iters})'
        )
    elif loss_name == 'bce_dice_cldice':
        # NOTE: Here the clDice weight uses bce_dice_cldice_weight; --cldice-weight is only a compatibility alias.
        loss_desc = (
            f'BCE+Dice+{bce_dice_cldice_weight}*clDice('
            f'CE={ce_weight}, Dice={dice_weight}, iters={cldice_iters}, w={class_weights}, ignore_bg=True)'
        )
    else:
        loss_name = 'wce_dice'
        loss_desc = f'WeightedCEDiceLoss(CE*{ce_weight}+Dice*{dice_weight}, w={class_weights})'

    summary = {
        'run_dir': str(run_dir),
        'model': 'SwinUNet',
        'training': {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'gradient_clipping': gradient_clipping,
            'amp': amp,
            'optimizer': 'AdamW',
            'scheduler': 'Warmup(LinearLR, 5 epochs) + CosineAnnealingLR',
            'loss': loss_desc,
            'loss_name': loss_name,
            'tversky_alpha': tversky_alpha,
            'tversky_beta': tversky_beta,
            'focal_tversky_gamma': focal_tversky_gamma,
            'cldice_iters': cldice_iters,
            'cldice_weight': cldice_weight,
            'ce_weight': ce_weight,
            'dice_weight': dice_weight,
            'bce_dice_cldice_weight': bce_dice_cldice_weight,
            'class_weights': class_weights,
            'n_train': n_train,
            'n_val': n_val,
            'device': str(device),
            'fg_crop': fg_crop,
            'fg_crop_sizes': list(fg_crop_sizes),
            'intensity_aug': bool(intensity_aug),
            'skip_gate': bool(getattr(model, 'use_skip_gate', False)),
            'deep_supervision': bool(getattr(model, 'use_deep_supervision', False)),
            'deep_supervision_weights': list(getattr(model, 'deep_supervision_weights', [])),
        },
        'extra': experiment_config or {},
        'best_epoch': None,
        'best_val_dice': None,
        'best_val_miou': None,
        'best_val_jaccard': None,
        'best_val_accuracy': None,
        'best_selection_score': None,
        'best_fg_pred_ratio': None,
        'total_train_time': None,
        'history': [],
    }
    if torch.cuda.is_available():
        summary['training']['gpu'] = torch.cuda.get_device_name(0)
    summary['training']['fg_crop'] = fg_crop
    summary['training']['fg_crop_sizes'] = list(fg_crop_sizes)

    logging.info(
        f'Starting training: epochs={epochs} batch_size={batch_size} lr={learning_rate} '
        f'device={device.type} amp={amp}'
    )
    logging.info(
        f'Augment: geom_flip/rotate/shiftscale + intensity_aug={bool(intensity_aug)} | '
        f'clahe={"ON" if getattr(train_dataset, "use_clahe", False) else "OFF"} | '
        f'fg_crop={bool(fg_crop)} | deep_supervision={bool(getattr(model, "use_deep_supervision", False))}'
    )

    loader_args = dict(
        # A bit aggressive for small datasets, but helps GPU utilization.
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=True, **loader_args)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Scheduler: short warm-up (stabilizes early training) then cosine decay.
    warmup_epochs = min(5, max(1, epochs // 10))
    scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_epochs
    )
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=1e-5
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_epochs]
    )
    grad_scaler = torch.amp.GradScaler('cuda', enabled=amp)

    if model.n_classes == 1:
        criterion = nn.BCEWithLogitsLoss()
    else:
        if loss_name == 'tversky':
            criterion = TverskyLoss(
                alpha=tversky_alpha,
                beta=tversky_beta,
                class_weights=class_weights,
                ignore_index=0,
                include_background=False,
            )
        elif loss_name == 'focal_tversky':
            criterion = FocalTverskyLoss(
                alpha=tversky_alpha,
                beta=tversky_beta,
                gamma=focal_tversky_gamma,
                class_weights=class_weights,
                ignore_index=0,
                include_background=False,
            )
        elif loss_name == 'cldice':
            criterion = SoftCLDiceLoss(
                iters=cldice_iters,
                class_weights=class_weights,
                include_background=False,
            )
        elif loss_name == 'tversky_cldice':
            tv = TverskyLoss(
                alpha=tversky_alpha,
                beta=tversky_beta,
                class_weights=class_weights,
                ignore_index=0,
                include_background=False,
            )
            cl = SoftCLDiceLoss(
                iters=cldice_iters,
                class_weights=class_weights,
                include_background=False,
            )

            class _TVCLDice(nn.Module):
                def __init__(self, tv_loss, cl_loss, w: float):
                    super().__init__()
                    self.tv_loss = tv_loss
                    self.cl_loss = cl_loss
                    self.w = float(w)

                def forward(self, inputs, targets):
                    return self.tv_loss(inputs, targets) + self.w * self.cl_loss(inputs, targets)

            criterion = _TVCLDice(tv, cl, cldice_weight)
        elif loss_name == 'bce_dice_cldice':
            criterion = BCEDiceCLDiceLoss(
                class_weights=class_weights,
                ce_weight=ce_weight,
                dice_weight=dice_weight,
                cldice_weight=bce_dice_cldice_weight,
                cldice_iters=cldice_iters,
                include_background=False,
            )
        else:
            criterion = WeightedCEDiceLoss(
                class_weights=class_weights,
                ce_weight=ce_weight,
                dice_weight=dice_weight,
            )

    best_dice = 0.0
    best_selection_score = float('-inf')
    epoch_times = []
    train_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_start_time = time.time()

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
            images = images.to(device=device, dtype=torch.float32,
                               memory_format=torch.channels_last, non_blocking=True)
            true_masks = true_masks.to(device=device, dtype=torch.long, non_blocking=True)

            with torch.autocast(device.type, enabled=amp):
                masks_pred = model(images)

                # Deep supervision contract:
                # - normal: model(...) -> logits
                # - deep supervision: model(...) -> (logits, [aux_logits...])
                aux_preds = None
                if isinstance(masks_pred, (tuple, list)):
                    if len(masks_pred) >= 2:
                        masks_pred, aux_preds = masks_pred[0], masks_pred[1]
                    else:
                        masks_pred = masks_pred[0]

                # Safety: some backbones/decoders slightly change spatial size.
                if masks_pred.shape[-2:] != true_masks.shape[-2:]:
                    masks_pred = torch.nn.functional.interpolate(
                        masks_pred,
                        size=true_masks.shape[-2:],
                        mode='bilinear',
                        align_corners=False,
                    )
                loss = criterion(masks_pred.squeeze(1), true_masks.float()) \
                    if model.n_classes == 1 else criterion(masks_pred, true_masks)

                if aux_preds is not None:
                    # Add auxiliary losses with per-head weights.
                    # The weights live on the model so you can tune them via CLI.
                    ds_weights = getattr(model, 'deep_supervision_weights', [])
                    if len(ds_weights) != len(aux_preds):
                        # Fallback defaults for 3 aux heads.
                        ds_weights = [0.3, 0.2, 0.1][:len(aux_preds)]
                    for w, aux in zip(ds_weights, aux_preds):
                        if aux.shape[-2:] != true_masks.shape[-2:]:
                            aux = torch.nn.functional.interpolate(
                                aux,
                                size=true_masks.shape[-2:],
                                mode='bilinear',
                                align_corners=False,
                            )
                        if model.n_classes == 1:
                            loss = loss + float(w) * criterion(aux.squeeze(1), true_masks.float())
                        else:
                            loss = loss + float(w) * criterion(aux, true_masks)

            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            grad_scaler.step(optimizer)
            grad_scaler.update()

            epoch_loss += loss.item()
            elapsed = time.time() - epoch_start_time
            speed = (pbar.n * batch_size) / elapsed if elapsed > 0 else 0
            pbar.set_postfix(loss=f'{loss.item():.4f}', speed=f'{speed:.0f}img/s')

        pbar.close()

        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        avg_loss = epoch_loss / max(1, len(train_loader))
        current_lr = optimizer.param_groups[0]['lr']
        throughput = n_train / epoch_time

        metrics = evaluate(model, val_loader, device, amp)
        dice_score = metrics['dice_score']

        iou_cat = float(metrics.get('iou_catheter') or 0.0)
        iou_gw = float(metrics.get('iou_guidewire') or 0.0)
        fg_pred_ratio = float(metrics.get('fg_pred_ratio') or 0.0)

        # Composite score used to select best_model: Dice + (catheter/guidewire IoU) - foreground-ratio penalty.
        # Why this exists: pure Dice sometimes prefers thicker/over-complete masks.
        # The foreground-ratio term nudges the selection towards more realistic thickness.
        selection_score = (
            0.5 * float(dice_score) +
            0.25 * iou_cat +
            0.25 * iou_gw -
            0.10 * abs(fg_pred_ratio - 1.0)
        )

        # Diagnostic: use fg_pred_ratio to estimate the dominant failure mode.
        # >1.2 usually means over-predicting foreground (false positives / thick masks); <0.8 usually means under-predicting foreground.
        if fg_pred_ratio > 1.2:
            diag = 'DIAG=over-predict(FP↑/thick)'
        elif fg_pred_ratio < 0.8:
            diag = 'DIAG=under-predict(FN↑/break)'
        else:
            diag = 'DIAG=balanced'

        scheduler.step()

        row = {
            'epoch': epoch,
            'train_loss': round(avg_loss, 6),
            'val_dice': round(metrics['dice_score'], 6),
            'val_miou': round(metrics['miou'], 6),
            'val_jaccard': round(metrics['jaccard'], 6),
            'val_accuracy': round(metrics['accuracy'], 6),
            'val_foreground_accuracy': round(metrics.get('foreground_accuracy') or 0, 6),
            'val_dice_catheter': round(metrics.get('dice_catheter') or 0, 6),
            'val_dice_guidewire': round(metrics.get('dice_guidewire') or 0, 6),
            'val_iou_catheter': round(metrics.get('iou_catheter') or 0, 6),
            'val_iou_guidewire': round(metrics.get('iou_guidewire') or 0, 6),
            'fg_pred_ratio': round(fg_pred_ratio, 6),
            'val_selection_score': round(float(selection_score), 6),
            'learning_rate': float(current_lr),
        }
        if csv_writer is not None and hasattr(csv_writer, 'fieldnames'):
            row = {k: row.get(k, '') for k in csv_writer.fieldnames}
        csv_writer.writerow(row)
        csv_file.flush()
        summary['history'].append(row)

        best_marker = ''
        if dice_score > best_dice:
            best_dice = float(dice_score)

        if float(selection_score) > float(best_selection_score):
            best_selection_score = float(selection_score)
            torch.save(model.state_dict(), str(run_dir / 'best_model.pth'))
            summary['best_epoch'] = epoch
            summary['best_val_dice'] = round(metrics['dice_score'], 6)
            summary['best_val_miou'] = round(metrics['miou'], 6)
            summary['best_val_jaccard'] = round(metrics['jaccard'], 6)
            summary['best_val_accuracy'] = round(metrics['accuracy'], 6)
            summary['best_selection_score'] = round(float(selection_score), 6)
            summary['best_fg_pred_ratio'] = round(float(fg_pred_ratio), 6)
            best_marker = ' *'

        remaining_h = (epochs - epoch) * avg_epoch_time / 3600
        logging.info(
            f'[ {epoch:>2}/{epochs}] '
            f'loss={avg_loss:.4f} | '
            f'dice={metrics["dice_score"]:.4f} miou={metrics["miou"]:.4f} | '
            f'cat={metrics.get("dice_catheter", 0):.4f} gw={metrics.get("dice_guidewire", 0):.4f} | '
            f'fg_ratio={fg_pred_ratio:.2f} {diag} | sel={selection_score:.4f} | '
            f'lr={current_lr:.1e} | {throughput:.0f}img/s | eta={remaining_h:.1f}h'
            f'{best_marker}'
        )

        if device.type == 'cuda':
            max_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            logging.info(f'[Epoch {epoch}] Max GPU Memory Used: {max_mem_gb:.2f} GB')
            torch.cuda.reset_peak_memory_stats()

        if save_checkpoint and (epoch % 10 == 0 or epoch == epochs):
            torch.save(model.state_dict(), str(run_dir / f'checkpoint_epoch{epoch}.pth'))
            logging.info(f'checkpoint_epoch{epoch}.pth saved')

    total_s = time.time() - train_start
    summary['total_train_time'] = str(datetime.timedelta(seconds=int(total_s)))
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    csv_file.close()

    logging.info('=' * 60)
    logging.info(f'Results saved to: {run_dir}')
    logging.info(f'best_model.pth — epoch {summary["best_epoch"]}')
    logging.info(f'Best Dice={summary["best_val_dice"]}  mIoU={summary["best_val_miou"]}')
    logging.info('=' * 60)


# -- main -----------------------------------------------------------------

def main():
    """CLI entry point for SwinUNet training."""
    script_start = time.time()

    epochs = 50
    batch_size = 8
    learning_rate = 1e-4
    classes = 3
    val_percent = 0.1
    amp = True
    use_preprocessed = False
    preprocessed_dir = './preprocessed_data'
    data_dir = None
    target_size = 768

    parser = argparse.ArgumentParser(
        description='Train SwinUNet',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--use-preprocessed', action='store_true')
    parser.add_argument('--preprocessed-dir', type=str, default=preprocessed_dir)
    parser.add_argument('--batch-size', '-b', type=int, default=batch_size)
    parser.add_argument('--epochs', '-e', type=int, default=epochs)
    parser.add_argument('--lr', type=float, default=learning_rate)
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--target-size', type=int, default=target_size)
    parser.add_argument('--train-subset', type=str, default='both',
                        help='Choose training subset: both/all, animal/animal_train, phantom/phantom_train')
    parser.add_argument('--class-filter', type=str, default='require-all',
                        choices=['require-all', 'allow-missing'],
                        help='require-all: keep only masks containing 0/1/2; allow-missing: do not filter and keep all masks')
    parser.add_argument('--fg-crop', action='store_true',
                        help='Enable foreground-aware multi-scale cropping followed by resize to target_size (train only)')
    parser.add_argument('--fg-crop-sizes', type=int, nargs='+', default=[256, 320, 384],
                        help='Candidate crop scales for foreground cropping')

    # NEW: intensity augmentation (Option A)
    parser.add_argument('--intensity-aug', action='store_true',
                        help='Enable X-ray intensity augmentation: brightness/contrast, gamma, noise, and light blur (train only)')

    # NEW: loss
    loss_g = parser.add_argument_group('loss')
    loss_g.add_argument('--loss', type=str, default='wce_dice',
                        choices=['wce_dice', 'tversky', 'focal_tversky', 'cldice', 'tversky_cldice', 'bce_dice_cldice'],
                        help='Loss function for multi-class training')
    loss_g.add_argument('--tversky-alpha', type=float, default=0.3,
                        help='Tversky alpha (FP penalty)')
    loss_g.add_argument('--tversky-beta', type=float, default=0.7,
                        help='Tversky beta (FN penalty)')
    loss_g.add_argument('--focal-tversky-gamma', type=float, default=1.33,
                        help='Focal Tversky gamma (>1 focuses on hard pixels)')
    loss_g.add_argument('--cldice-iters', type=int, default=10,
                        help='Soft skeletonization iterations for clDice (typical 5~20)')

    # Compatibility: people often used --cldice-weight 0.5 in older runs.
    # Under bce_dice_cldice, it is equivalent to --bce-dice-cldice-weight.
    loss_g.add_argument('--cldice-weight', type=float, default=None,
                        help='Alias for --bce-dice-cldice-weight when --loss bce_dice_cldice; '
                             'or for --loss tversky_cldice it is the clDice weight.')

    # NEW: BCE+Dice+clDice combo weights
    loss_g.add_argument('--ce-weight', type=float, default=0.4,
                        help='CE weight for bce_dice_cldice')
    loss_g.add_argument('--dice-weight', type=float, default=0.6,
                        help='Dice weight for bce_dice_cldice')
    loss_g.add_argument('--bce-dice-cldice-weight', type=float, default=0.5,
                        help='clDice weight for bce_dice_cldice (recommended 0.1~0.7 as an auxiliary term)')
    loss_g.add_argument('--class-weights', type=float, nargs=3, default=[0.05, 2.0, 5.0],
                        help='Class weights for loss function (default: 0.05 2.0 5.0)')

    arch = parser.add_argument_group('model architecture')
    arch.add_argument('--model-name', type=str, default=None,
                      help='timm model name (default: swinv2_tiny_window8_256)')
    arch.add_argument('--pretrained', action='store_true',
                      help='Use timm pretrained weights (requires download)')
    arch.add_argument('--decoder-channels', type=int, default=None, nargs=4,
                      metavar=('D0', 'D1', 'D2', 'D3'))
    arch.add_argument('--img-size', type=int, default=None,
                      help='Override Swin backbone input size (default: target-size)')
    arch.add_argument('--skip-gate', action='store_true',
                      help='Enable SE gate on skip features in decoder (default: off)')
    arch.add_argument('--deep-supervision', action='store_true',
                      help='Enable deep supervision heads in decoder (default: off)')
    arch.add_argument('--ds-weights', '--deep-supervision-weights', type=float, nargs='+', default=None,
                      dest='ds_weights',
                      help='Deep supervision weights for aux heads (e.g., 0.3 0.2 0.1)')

    # Decoder selection.
    # - cnn: the original SwinUNet decoder (upsample + concat + conv)
    # - transformer: token-mixer decoder implemented in swinunet/transformer_decoder.py
    arch.add_argument('--decoder-type', type=str, default='cnn', choices=['cnn', 'transformer'],
                      help='Decoder type: cnn (default) or transformer')

    # Transformer-decoder knobs.
    # These are ignored when --decoder-type=cnn.
    arch.add_argument('--decoder-depths', type=int, nargs=4, default=[1, 1, 1, 1],
                      metavar=('D3', 'D2', 'D1', 'D0'),
                      help='Transformer decoder depths for 4 upsampling stages (default: 1 1 1 1)')
    arch.add_argument('--decoder-num-heads', type=int, nargs=4, default=[8, 8, 4, 4],
                      metavar=('H3', 'H2', 'H1', 'H0'),
                      help='Transformer decoder attention heads per stage (default: 8 8 4 4)')
    arch.add_argument('--decoder-mlp-ratio', type=float, default=4.0,
                      help='Transformer decoder MLP expansion ratio (default: 4.0)')
    arch.add_argument('--decoder-drop', type=float, default=0.0,
                      help='Transformer decoder dropout (residual dropout, default: 0.0)')
    arch.add_argument('--decoder-attn-drop', type=float, default=0.0,
                      help='Transformer decoder attention dropout (default: 0.0)')
    arch.add_argument('--decoder-proj-drop', type=float, default=0.0,
                      help='Transformer decoder MLP/projection dropout (default: 0.0)')

    args = parser.parse_args()

    # Route the clDice weight based on the selected loss.
    # - tversky_cldice: use args.cldice_weight (default 1.0, overridden by --cldice-weight)
    # - bce_dice_cldice: use args.bce_dice_cldice_weight (overridden by the --cldice-weight alias)
    if args.loss == 'tversky_cldice':
        cldice_weight = float(args.cldice_weight) if args.cldice_weight is not None else float(args.cldice_weight or 1.0)
        # Keep the change minimal; explicitly enforce the default of 1.0 below.
        if args.cldice_weight is None:
            cldice_weight = 1.0
    else:
        cldice_weight = 1.0

    bce_dice_cldice_weight = float(args.bce_dice_cldice_weight)
    if args.loss == 'bce_dice_cldice' and args.cldice_weight is not None:
        bce_dice_cldice_weight = float(args.cldice_weight)

    use_preprocessed = args.use_preprocessed
    preprocessed_dir = args.preprocessed_dir
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.lr
    target_size = args.target_size
    if args.data_dir:
        data_dir = args.data_dir

    decoder_channels = tuple(args.decoder_channels) if args.decoder_channels else None
    img_size = args.img_size or target_size
    require_all_classes = args.class_filter == 'require-all'
    fg_crop = args.fg_crop
    fg_crop_sizes = args.fg_crop_sizes
    intensity_aug = bool(args.intensity_aug)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    if torch.cuda.is_available():
        logging.info(f'GPU: {torch.cuda.get_device_name(0)}')

    model = SwinUNet(
        n_channels=3,
        n_classes=classes,
        model_name=args.model_name,
        pretrained=args.pretrained,
        decoder_channels=decoder_channels,
        img_size=img_size,
        use_skip_gate=bool(args.skip_gate),
        use_deep_supervision=bool(args.deep_supervision),
        decoder_type=str(args.decoder_type),
        decoder_depths=tuple(int(x) for x in args.decoder_depths),
        decoder_num_heads=tuple(int(x) for x in args.decoder_num_heads),
        decoder_mlp_ratio=float(args.decoder_mlp_ratio),
        decoder_drop=float(args.decoder_drop),
        decoder_attn_drop=float(args.decoder_attn_drop),
        decoder_proj_drop=float(args.decoder_proj_drop),
    )
    model.deep_supervision_weights = args.ds_weights or [0.3, 0.2, 0.1]

    # Quick parameter count sanity-check (handy when you swap backbones).
    param_m = sum(p.numel() for p in model.parameters()) / 1e6
    logging.info(f'SwinUNet model={args.model_name or model.DEFAULT_MODEL} params={param_m:.1f}M')
    model.to(device=device)

    if use_preprocessed:
        logging.info('=' * 60)
        logging.info('Using preprocessed dataset')
        logging.info('=' * 60)
        if (args.train_subset or 'both').strip().lower() not in ['both', 'all']:
            logging.warning('--train-subset is ignored when --use-preprocessed is enabled')
        try:
            dataset = PreprocessedDataset(preprocessed_dir=preprocessed_dir)
            from torch.utils.data import random_split as rs
            n_val_n = int(len(dataset) * val_percent)
            n_train_n = len(dataset) - n_val_n
            train_dataset, val_dataset = rs(
                dataset, [n_train_n, n_val_n],
                generator=torch.Generator().manual_seed(0),
            )
        except RuntimeError as e:
            logging.error(str(e))
            return
    else:
        logging.info('=' * 60)
        logging.info('Using raw dataset')
        logging.info('=' * 60)

        if data_dir is None:
            data_dir = os.getenv('DATA_DIR', './segmentation')

        base = Path(data_dir)
        if not base.exists():
            for cand in [Path('./segmentation'), Path('../unet/segmentation')]:
                if (cand / 'animal_train' / 'images').exists():
                    base = cand
                    logging.info(f'Found data at: {base}')
                    break

        selected_subdirs = _resolve_train_subdirs(args.train_subset)
        logging.info(f'Training subset: {args.train_subset} -> {selected_subdirs}')

        image_paths, mask_paths = [], []
        per_subset_counts = {}
        for sub in selected_subdirs:
            dir_img = base / sub / 'images'
            dir_mask = base / sub / 'masks'
            if not dir_img.exists():
                logging.warning(f'Directory not found: {dir_img}')
                continue
            before_n = len(image_paths)
            for img_p in sorted(
                list(dir_img.glob('*.png')) +
                list(dir_img.glob('*.jpg')) +
                list(dir_img.glob('*.jpeg'))
            ):
                mask_p = dir_mask / (img_p.stem + '.npy')
                if mask_p.exists():
                    image_paths.append(str(img_p))
                    mask_paths.append(str(mask_p))
            per_subset_counts[sub] = len(image_paths) - before_n

        if not image_paths:
            logging.error(f'No image/mask pairs found for --train-subset={args.train_subset} under {base}')
            return

        for sub in selected_subdirs:
            logging.info(f'Pairs from {sub}: {per_subset_counts.get(sub, 0)}')

        logging.info(f'Total image/mask pairs: {len(image_paths)}')

        if require_all_classes:
            logging.info('Filtering to masks that contain classes {0,1,2}...')
            image_paths, mask_paths, dropped = _filter_pairs_all_classes(image_paths, mask_paths)
            logging.info(f'Filtered pairs kept: {len(image_paths)} (dropped {dropped})')

        indices = list(range(len(image_paths)))
        random.Random(0).shuffle(indices)
        n_val_n = int(len(indices) * val_percent)
        val_indices = indices[:n_val_n]
        train_indices = indices[n_val_n:]

        kw = dict(target_size=target_size, use_clahe=False,
                  mean=[0.4172, 0.4167, 0.4173], std=[0.2436, 0.2435, 0.2436],
                  fg_crop=fg_crop, fg_crop_sizes=fg_crop_sizes,
                  intensity_aug=intensity_aug)
        # NOTE: mean/std are dataset statistics (RGB). Keep in sync with your preprocessing.
        train_dataset = CatheterPreprocessingDataset(
            image_paths=[image_paths[i] for i in train_indices],
            mask_paths=[mask_paths[i] for i in train_indices],
            mode='train', **kw)
        val_dataset = CatheterPreprocessingDataset(
            image_paths=[image_paths[i] for i in val_indices],
            mask_paths=[mask_paths[i] for i in val_indices],
            mode='val', **kw)

    train_model(
        model=model, device=device,
        train_dataset=train_dataset, val_dataset=val_dataset,
        epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, amp=amp,
        loss_name=args.loss,
        tversky_alpha=args.tversky_alpha,
        tversky_beta=args.tversky_beta,
        focal_tversky_gamma=args.focal_tversky_gamma,
        cldice_iters=args.cldice_iters,
        cldice_weight=cldice_weight,
        ce_weight=float(args.ce_weight),
        dice_weight=float(args.dice_weight),
        bce_dice_cldice_weight=bce_dice_cldice_weight,
        class_weights=tuple(args.class_weights),
        fg_crop=fg_crop,
        fg_crop_sizes=fg_crop_sizes,
        intensity_aug=intensity_aug,
        experiment_config={
            'model': 'SwinUNet',
            'model_name': args.model_name or model.DEFAULT_MODEL,
            'train_subset': args.train_subset,
            'decoder_channels': list(decoder_channels) if decoder_channels else None,
            'img_size': img_size,
            'val_percent': val_percent,
            'target_size': target_size,
            'preprocessing': 'mixed(pad<=target_size, fg_centered_crop>target_size)',
            'fg_crop': fg_crop,
            'fg_crop_sizes': fg_crop_sizes,
            'intensity_aug': intensity_aug,
            'clahe': False,
            'skip_gate': bool(args.skip_gate),
            'deep_supervision': bool(args.deep_supervision),
            'deep_supervision_weights': list(model.deep_supervision_weights),
        }
    )

    logging.info(f'Total time: {datetime.timedelta(seconds=int(time.time() - script_start))}')


if __name__ == '__main__':
    main()

