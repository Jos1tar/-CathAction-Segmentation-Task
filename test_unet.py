"""
Standalone evaluation script for testing a trained model with per-class metrics.

Usage:
    python test_model.py --model checkpoints/20260310/best_model.pth --base-channels 64

"""

import argparse
import logging
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from unet import UNet
from preprocess import CatheterPreprocessingDataset


def main():
    # ---------------------------------------------------------------------
    # Argument parsing
    # ---------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='Evaluate trained U-Net model')
    parser.add_argument('--model', '-m', type=str, required=True,
                        help='Path to trained model (.pth file)')
    parser.add_argument('--batch-size', '-b', type=int, default=8,
                        help='Batch size for evaluation')
    parser.add_argument('--classes', '-c', type=int, default=3,
                        help='Number of classes (including background)')
    parser.add_argument('--base-channels', type=int, default=64,
                        help='Base number of channels in U-Net')
    parser.add_argument('--data-dir', type=str, default='./segmentation',
                        help='Path to data directory')
    args = parser.parse_args()

    # ---------------------------------------------------------------------
    # Logging
    # ---------------------------------------------------------------------
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # ---------------------------------------------------------------------
    # Device
    # ---------------------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    # ---------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------
    logging.info(f'Loading model from {args.model}')
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=False,
                 base_channels=args.base_channels)

    state_dict = torch.load(args.model, map_location=device)
    # Remove 'mask_values' if it exists (legacy key)
    if 'mask_values' in state_dict:
        del state_dict['mask_values']

    model.load_state_dict(state_dict)
    model.to(device)
    logging.info('Model loaded successfully')

    # ---------------------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------------------
    BASE = Path(args.data_dir)

    # Resolve test directories.
    test_dirs = [
        (BASE / "animal_test" / "images", BASE / "animal_test" / "masks"),
        (BASE / "phantom_test" / "images", BASE / "phantom_test" / "masks"),
    ]

    image_paths = []
    mask_paths = []

    for dir_img, dir_mask in test_dirs:
        if not dir_img.exists():
            logging.warning(f"Directory not found: {dir_img}")
            continue

        all_images = sorted(
            list(dir_img.glob('*.png')) +
            list(dir_img.glob('*.jpg')) +
            list(dir_img.glob('*.jpeg'))
        )

        logging.info(f"Scanning images from {dir_img}...")
        for img_p in all_images:
            mask_p = dir_mask / (img_p.stem + '.npy')
            if mask_p.exists():
                image_paths.append(str(img_p))
                mask_paths.append(str(mask_p))

    if len(image_paths) == 0:
        logging.error("No test data found! Please check your data directory.")
        return

    logging.info(f"Total test image/mask pairs: {len(image_paths)}")

    # Build dataset.
    test_dataset = CatheterPreprocessingDataset(
        image_paths=image_paths,
        mask_paths=mask_paths,
        target_size=512,
        use_clahe=False,
        mean=[0.4172, 0.4167, 0.4173],
        std=[0.2436, 0.2435, 0.2436],
        mode='val'  # Critical: disable augmentation for testing.
    )

    # DataLoader.
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # ---------------------------------------------------------------------
    # Custom multi-class evaluation
    # ---------------------------------------------------------------------
    logging.info('=' * 60)
    logging.info('Starting custom per-class evaluation...')
    logging.info('=' * 60)

    model.eval()
    num_classes = args.classes

    # Accumulators for metrics.
    total_inter = torch.zeros(num_classes, device=device)
    total_union = torch.zeros(num_classes, device=device)
    total_target = torch.zeros(num_classes, device=device)
    total_pred = torch.zeros(num_classes, device=device)

    correct_pixels = 0
    total_pixels_all = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", unit="batch"):
            images, true_masks = batch['image'].to(device), batch['mask'].to(device)

            with torch.autocast(device.type, enabled=True):
                masks_pred = model(images)

            # Predicted labels in {0, 1, 2}.
            if num_classes > 1:
                preds = torch.argmax(masks_pred, dim=1)
            else:
                preds = (torch.sigmoid(masks_pred) > 0.5).long().squeeze(1)

            # Global pixel accuracy.
            correct_pixels += (preds == true_masks).sum().item()
            total_pixels_all += true_masks.numel()

            # Per-class intersection/union and pixel counts.
            for c in range(num_classes):
                pred_c = (preds == c)
                target_c = (true_masks == c)

                total_inter[c] += (pred_c & target_c).sum()
                total_union[c] += (pred_c | target_c).sum()
                total_target[c] += target_c.sum()
                total_pred[c] += pred_c.sum()

    # Compute metrics (eps avoids division by zero).
    eps = 1e-6
    class_dice    = (2 * total_inter) / (total_pred + total_target + eps)
    class_iou     = total_inter / (total_union + eps)
    class_jaccard = class_iou                              # Jaccard = IoU
    class_acc     = total_inter / (total_target + eps)     # per-class recall

    # Foreground averages (exclude background=0).
    mean_dice = class_dice[1:].mean().item()
    mean_iou  = class_iou[1:].mean().item()

    # Jaccard includes background (mean over all classes).
    mean_jaccard = class_jaccard.mean().item()

    # Foreground mean accuracy (avg over classes 1..C-1 to avoid background dominance).
    fg_mean_acc = class_acc[1:].mean().item()

    # Keep global pixel accuracy for reference only (background included).
    global_acc = correct_pixels / total_pixels_all

    class_names = {0: "Background", 1: "Catheter", 2: "Guidewire"}
    W = 28   # column width for alignment

    # =====================================================================
    # 1) Overall summary
    # =====================================================================
    logging.info('=' * 60)
    logging.info('OVERALL METRICS (foreground average, excluding background)')
    logging.info('=' * 60)
    logging.info(f'  {"Global Pixel Accuracy":<{W}}: {global_acc:.4f}  (background included, ref only)')
    logging.info(f'  {"Mean Dice Score":<{W}}: {mean_dice:.4f}')
    logging.info(f'  {"Mean Jaccard Index":<{W}}: {mean_jaccard:.4f}  (includes background)')
    logging.info(f'  {"Mean IoU (mIoU)":<{W}}: {mean_iou:.4f}')
    logging.info(f'  {"Foreground Mean Accuracy":<{W}}: {fg_mean_acc:.4f}  (avg over classes 1..C-1)')
    logging.info('=' * 60)

    # Compare with the reported baseline U-Net (CathAction, Table IV).
    baseline_dice    = 0.5169
    baseline_jaccard = 0.5751
    baseline_miou    = 0.3117
    baseline_acc     = 0.6326
    logging.info('  Comparison with baseline U-Net (CathAction, Table IV):')
    logging.info(f'  {"Metric":<{W}}  {"Baseline":>10}  {"Yours":>10}  {"Delta":>10}')
    logging.info(f'  {"-"*58}')

    def _delta(val, base):
        d = (val - base) / base * 100
        return f'{d:+.1f}%'

    logging.info(f'  {"Dice Score":<{W}}  {baseline_dice:>10.4f}  {mean_dice:>10.4f}  {_delta(mean_dice, baseline_dice):>10}')
    logging.info(f'  {"Jaccard Index":<{W}}  {baseline_jaccard:>10.4f}  {mean_jaccard:>10.4f}  {_delta(mean_jaccard, baseline_jaccard):>10}')
    logging.info(f'  {"mIoU":<{W}}  {baseline_miou:>10.4f}  {mean_iou:>10.4f}  {_delta(mean_iou, baseline_miou):>10}')
    logging.info(f'  {"Accuracy":<{W}}  {baseline_acc:>10.4f}  {fg_mean_acc:>10.4f}  {_delta(fg_mean_acc, baseline_acc):>10}')
    logging.info('=' * 60)

    # Simple qualitative rating.
    if mean_dice > 0.60:
        logging.info('  Rating: Excellent (above transformer-level performance)')
    elif mean_dice > 0.55:
        logging.info('  Rating: Good (competitive with TransUNet / SwinUNet)')
    elif mean_dice > 0.50:
        logging.info('  Rating: Acceptable (on par with baseline U-Net)')
    else:
        logging.info('  Rating: Below baseline (consider adjusting loss weights / lr)')
    logging.info('=' * 60)

    # =====================================================================
    # 2) Per-class detail
    # =====================================================================
    logging.info('')
    logging.info('=' * 60)
    logging.info('PER-CLASS DETAIL')
    logging.info('=' * 60)
    logging.info(f'  {"Class":<{W}}  {"Accuracy":>10}  {"Dice":>10}  {"Jaccard":>10}  {"IoU":>10}  {"Pixel%":>8}')
    logging.info(f'  {"-"*78}')
    for c in range(num_classes):
        name    = class_names.get(c, f'Class {c}')
        pct     = (total_target[c] / total_pixels_all * 100).item()
        acc_c   = class_acc[c].item()
        dice_c  = class_dice[c].item()
        jacc_c  = class_jaccard[c].item()
        iou_c   = class_iou[c].item()
        logging.info(f'  {name:<{W}}  {acc_c:>10.4f}  {dice_c:>10.4f}  {jacc_c:>10.4f}  {iou_c:>10.4f}  {pct:>7.2f}%')
    logging.info('=' * 60)


if __name__ == '__main__':
    main()