# coding: utf-8
"""
Standalone evaluation script for testing a trained TransUNet model
with per-class metrics.

Usage:
    python test_transunet.py --model checkpoints_transunet/20260413/best_model.pth
    python test_transunet.py --model checkpoints_transunet/20260311/best_model.pth --preset vit-s
    python test_transunet.py --model checkpoints_transunet/20260311/best_model.pth \\
        --hidden-size 512 --num-heads 6 --num-layers 6
    python test_transunet.py --model checkpoints_transunet/20260318/best_model.pth --batch-size 12 --auto-detect

"""

import argparse
import logging
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from transunet import TransUNet
from preprocess import CatheterPreprocessingDataset


def main():
    # ---------------------------------------------------------------------
    # Argument parsing
    # ---------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='Evaluate trained TransUNet model')
    parser.add_argument('--model', '-m', type=str, required=True,
                        help='Path to trained model (.pth file)')
    parser.add_argument('--batch-size', '-b', type=int, default=4,
                        help='Batch size for evaluation (TransUNet uses more VRAM, default=4)')
    parser.add_argument('--classes', '-c', type=int, default=3,
                        help='Number of classes (including background)')
    parser.add_argument('--data-dir', type=str, default='./segmentation',
                        help='Path to data directory')

    # Model architecture arguments (should match train_transunet.py).
    parser.add_argument('--preset', type=str, default=None,
                        choices=['vit-b', 'vit-s'],
                        help='Use a named preset (vit-b or vit-s). '
                             'Overrides hidden-size / num-heads / num-layers.')
    parser.add_argument('--hidden-size', type=int, default=None,
                        help='Transformer hidden dim (768=ViT-B, 384=ViT-S)')
    parser.add_argument('--num-heads', type=int, default=None,
                        help='Attention heads (12=ViT-B, 6=ViT-S)')
    parser.add_argument('--num-layers', type=int, default=None,
                        help='Transformer layers (12=ViT-B, 6=ViT-S)')
    parser.add_argument('--decoder-channels', type=int, nargs=4,
                        default=None,
                        metavar=('D0', 'D1', 'D2', 'D3'),
                        help='Decoder channel sizes (default: 384 192 96 48 for ViT-B)')
    parser.add_argument('--auto-detect', action='store_true',
                        help='Auto-detect architecture from summary.json in checkpoint dir')

    # Optional softmax probability thresholds (helps suppress false positives).
    parser.add_argument('--threshold-catheter', type=float, default=0.0,
                        help='Softmax threshold for catheter (class=1). 0 disables thresholding.')
    parser.add_argument('--threshold-guidewire', type=float, default=0.0,
                        help='Softmax threshold for guidewire (class=2). 0 disables thresholding.')

    args = parser.parse_args()

    # Try to auto-detect architecture from summary.json (saved next to checkpoints).
    if args.auto_detect or (args.hidden_size is None and args.preset is None):
        model_path = Path(args.model)
        summary_path = model_path.parent / 'summary.json'
        if summary_path.exists():
            logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
            logging.info('Auto-detecting architecture from: %s', summary_path)
            try:
                with open(summary_path, 'r', encoding='utf-8') as f:
                    summary = json.load(f)
                # Read architecture parameters from either `training` or `extra`.
                config = summary.get('training', {})
                config.update(summary.get('extra', {}))  # `extra` overrides `training`

                if args.hidden_size is None:
                    args.hidden_size = config.get('hidden_size', 768)
                if args.num_heads is None:
                    args.num_heads = config.get('num_heads', 12)
                if args.num_layers is None:
                    args.num_layers = config.get('num_layers', 12)
                if args.decoder_channels is None:
                    dc = config.get('decoder_channels', [384, 192, 96, 48])
                    args.decoder_channels = dc
                logging.info(
                    'Loaded: hidden_size=%s, num_heads=%s, num_layers=%s, decoder_channels=%s',
                    args.hidden_size, args.num_heads, args.num_layers, args.decoder_channels,
                )
            except Exception as e:
                logging.warning(f'Failed to parse summary.json: {e}')
                logging.warning('Falling back to manual arguments or defaults.')

    # Fill defaults (if still unspecified).
    if args.hidden_size is None:
        args.hidden_size = 768
    if args.num_heads is None:
        args.num_heads = 12
    if args.num_layers is None:
        args.num_layers = 12
    if args.decoder_channels is None:
        args.decoder_channels = [384, 192, 96, 48]

    # Logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    if device.type == 'cuda':
        logging.info(f'GPU: {torch.cuda.get_device_name(0)}')

    # Build model
    if args.preset is not None:
        logging.info(f'Loading TransUNet with preset: {args.preset}')
        model = TransUNet.from_preset(args.preset, n_classes=args.classes)
    else:
        decoder_channels = tuple(args.decoder_channels)
        logging.info(
            f'Loading TransUNet: hidden_size={args.hidden_size}, '
            f'num_heads={args.num_heads}, num_layers={args.num_layers}, '
            f'decoder_channels={decoder_channels}'
        )
        model = TransUNet(
            n_channels=3,
            n_classes=args.classes,
            hidden_size=args.hidden_size,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            decoder_channels=decoder_channels,
        )

    # ---------------------------------------------------------------------
    # Load weights
    # ---------------------------------------------------------------------
    logging.info(f'Loading weights from: {args.model}')
    state_dict = torch.load(args.model, map_location=device)
    if 'mask_values' in state_dict:
        del state_dict['mask_values']

    # Safety check: validate that the checkpoint matches the requested architecture.
    try:
        model.load_state_dict(state_dict, strict=True)
        logging.info('Model loaded successfully (strict=True)')
    except RuntimeError as e:
        logging.error('=' * 60)
        logging.error('CRITICAL: Model architecture mismatch')
        logging.error('=' * 60)
        logging.error(str(e))
        logging.error('')
        logging.error('This usually means your test command uses DIFFERENT hyperparameters')
        logging.error('than the training run. Check your training command and ensure:')
        logging.error('  --hidden-size, --num-heads, --num-layers, --decoder-channels')
        logging.error('are IDENTICAL to the values used during training.')
        logging.error('=' * 60)
        raise

    model.to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    logging.info(f'Model parameters: {total_params:.1f}M')

    # ---------------------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------------------
    BASE = Path(args.data_dir)
    test_dirs = [
        (BASE / 'animal_test'  / 'images', BASE / 'animal_test'  / 'masks'),
        (BASE / 'phantom_test' / 'images', BASE / 'phantom_test' / 'masks'),
    ]

    image_paths, mask_paths = [], []
    for dir_img, dir_mask in test_dirs:
        if not dir_img.exists():
            logging.warning(f'Directory not found: {dir_img}')
            continue
        all_images = sorted(
            list(dir_img.glob('*.png')) +
            list(dir_img.glob('*.jpg')) +
            list(dir_img.glob('*.jpeg'))
        )
        logging.info(f'Scanning images from {dir_img}...')
        for img_p in all_images:
            mask_p = dir_mask / (img_p.stem + '.npy')
            if mask_p.exists():
                image_paths.append(str(img_p))
                mask_paths.append(str(mask_p))

    if not image_paths:
        logging.error('No test data found! Please check your data directory.')
        return
    logging.info(f'Total test image/mask pairs: {len(image_paths)}')

    test_dataset = CatheterPreprocessingDataset(
        image_paths=image_paths,
        mask_paths=mask_paths,
        target_size=512,
        use_clahe=False,
        mean=[0.4172, 0.4167, 0.4173],
        std=[0.2436, 0.2435, 0.2436],
        mode='val',  # Critical: use 'val' mode to disable augmentation.
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == 'cuda'),
    )

    # ---------------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------------
    logging.info('=' * 60)
    logging.info('Starting per-class evaluation...')
    logging.info('=' * 60)

    model.eval()
    num_classes = args.classes

    total_inter  = torch.zeros(num_classes, device=device)
    total_union  = torch.zeros(num_classes, device=device)
    total_target = torch.zeros(num_classes, device=device)
    total_pred   = torch.zeros(num_classes, device=device)
    correct_pixels   = 0
    total_pixels_all = 0

    # Extra statistic: predicted foreground ratio (pred_fg_pixels / true_fg_pixels).
    total_pred_fg = 0
    total_true_fg = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating', unit='batch'):
            images     = batch['image'].to(device)
            true_masks = batch['mask'].to(device)

            with torch.autocast(device.type, enabled=(device.type == 'cuda')):
                masks_pred = model(images)

            # logits -> preds
            if (args.threshold_catheter and args.threshold_catheter > 0) or \
               (args.threshold_guidewire and args.threshold_guidewire > 0):
                # thresholded softmax for foreground classes
                probs = torch.softmax(masks_pred, dim=1)
                p0 = probs[:, 0]
                p1 = probs[:, 1]
                p2 = probs[:, 2] if num_classes > 2 else None

                preds = torch.zeros_like(true_masks)
                # catheter
                cat_mask = p1 > float(args.threshold_catheter)
                preds[cat_mask] = 1
                # guidewire
                if num_classes > 2 and p2 is not None:
                    gw_mask = p2 > float(args.threshold_guidewire)
                    preds[gw_mask] = 2

                # conflict resolution: keep higher prob between 1 and 2 when both pass threshold
                if num_classes > 2 and p2 is not None:
                    both = cat_mask & gw_mask
                    if both.any():
                        pick_gw = p2[both] > p1[both]
                        # default already 2 for gw_mask; set back to 1 where catheter wins
                        idx = both.nonzero(as_tuple=True)
                        preds[idx] = torch.where(pick_gw, torch.tensor(2, device=device), torch.tensor(1, device=device))

                # background wins only if neither foreground passes threshold
                bg = (~cat_mask) & (~gw_mask if (num_classes > 2 and p2 is not None) else torch.ones_like(cat_mask, dtype=torch.bool))
                preds[bg] = 0
            else:
                preds = torch.argmax(masks_pred, dim=1)

            correct_pixels   += (preds == true_masks).sum().item()
            total_pixels_all += true_masks.numel()

            total_pred_fg += (preds > 0).sum().item()
            total_true_fg += (true_masks > 0).sum().item()

            for c in range(num_classes):
                pred_c   = (preds == c)
                target_c = (true_masks == c)
                total_inter[c]  += (pred_c & target_c).sum()
                total_union[c]  += (pred_c | target_c).sum()
                total_target[c] += target_c.sum()
                total_pred[c]   += pred_c.sum()

    # ---------------------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------------------
    eps = 1e-6
    class_dice    = (2 * total_inter) / (total_pred + total_target + eps)
    class_iou     = total_inter / (total_union + eps)
    class_jaccard = class_iou
    class_acc     = total_inter / (total_target + eps)    # per-class recall

    # Per-class precision (directly reflects false positives).
    class_precision = total_inter / (total_pred + eps)

    # Foreground-only averages (exclude background=0)
    mean_dice    = class_dice[1:].mean().item()
    mean_iou     = class_iou[1:].mean().item()
    mean_jaccard = class_jaccard.mean().item()  # includes background

    fg_mean_acc  = class_acc[1:].mean().item()
    global_acc   = correct_pixels / total_pixels_all

    fg_pred_ratio = total_pred_fg / max(total_true_fg, 1)

    class_names = {0: 'Background', 1: 'Catheter', 2: 'Guidewire'}
    W = 28

    # ---------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------
    logging.info('=' * 60)
    logging.info('OVERALL METRICS (foreground average, excluding background)')
    logging.info('=' * 60)
    logging.info(f'  {"Global Pixel Accuracy":<{W}}: {global_acc:.4f}  (background included, ref only)')
    logging.info(f'  {"Mean Dice Score":<{W}}: {mean_dice:.4f}')
    logging.info(f'  {"Mean Jaccard Index":<{W}}: {mean_jaccard:.4f}  (includes background)')
    logging.info(f'  {"Mean IoU (mIoU)":<{W}}: {mean_iou:.4f}')
    logging.info(f'  {"Foreground Mean Accuracy":<{W}}: {fg_mean_acc:.4f}  (avg over classes 1..C-1)')
    logging.info(f'  {"pred_fg_ratio":<{W}}: {fg_pred_ratio:.3f}  (pred_fg / true_fg, ~1 is healthy)')
    logging.info('=' * 60)

    # Comparison against a reference baseline (reported in the original paper).
    baselines = {
        'Dice Score':    (0.5652, mean_dice),
        'Jaccard Index': (0.5593, mean_jaccard),
        'mIoU':          (0.3413, mean_iou),
        'Accuracy':      (0.5561, fg_mean_acc),
    }
    logging.info('  Comparison with baseline TransUNet (CathAction, Table IV):')
    logging.info(f'  {"Metric":<{W}}  {"Baseline":>10}  {"Yours":>10}  {"Delta":>10}')
    logging.info(f'  {"-"*58}')
    for metric, (base, val) in baselines.items():
        d = (val - base) / base * 100
        logging.info(f'  {metric:<{W}}  {base:>10.4f}  {val:>10.4f}  {d:>+10.1f}%')
    logging.info('=' * 60)

    # Simple qualitative rating for quick sanity checks.
    if mean_dice > 0.63:
        logging.info('  Rating: Excellent (above SegViT-level performance)')
    elif mean_dice > 0.60:
        logging.info('  Rating: Good (competitive with SwinUNet / SegViT)')
    elif mean_dice > 0.56:
        logging.info('  Rating: Acceptable (on par with baseline TransUNet)')
    else:
        logging.info('  Rating: Below baseline (consider adjusting loss / lr)')
    logging.info('=' * 60)

    # ---------------------------------------------------------------------
    # Per-class breakdown
    # ---------------------------------------------------------------------
    logging.info('')
    logging.info('=' * 60)
    logging.info('PER-CLASS DETAIL')
    logging.info('=' * 60)
    logging.info(f'  {"Class":<{W}}  {"Accuracy":>10}  {"Precision":>10}  {"Dice":>10}  {"Jaccard":>10}  {"IoU":>10}  {"Pixel%":>8}')
    logging.info(f'  {"-"*90}')
    for c in range(num_classes):
        name   = class_names.get(c, f'Class {c}')
        pct    = (total_target[c] / total_pixels_all * 100).item()
        acc_c  = class_acc[c].item()
        pre_c  = class_precision[c].item()
        dice_c = class_dice[c].item()
        jacc_c = class_jaccard[c].item()
        iou_c  = class_iou[c].item()
        logging.info(
            f'  {name:<{W}}  {acc_c:>10.4f}  {pre_c:>10.4f}  {dice_c:>10.4f}  '
            f'{jacc_c:>10.4f}  {iou_c:>10.4f}  {pct:>7.2f}%'
        )
    logging.info('=' * 60)


if __name__ == '__main__':
    main()

