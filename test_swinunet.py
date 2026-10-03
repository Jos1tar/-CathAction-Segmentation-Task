# coding: utf-8
"""\

python test_swinunet.py \
  --model checkpoints/checkpoints_swinunet/20260501_v3/best_model.pth \
  --auto-detect \
  --test-set animal,phantom\
  --eval012-mode filter-contains \
  --decoder-type transformer \
  --decoder-mlp-ratio 2.0



python test_swinunet.py --model checkpoints_swinunet/20260414/best_model.pth --auto-detect --test-set all
python test_swinunet.py --model checkpoints_swinunet/20260414/best_model.pth --auto-detect --test-set animal
python test_swinunet.py --model checkpoints_swinunet/20260414/best_model.pth --auto-detect --test-set phantom_test
python test_swinunet.py --model checkpoints_swinunet/20260414/best_model.pth --auto-detect --test-set animal,phantom


Linux example:
  python test_swinunet.py \
    --model checkpoints_swinunet/20260410_v2/best_model.pth \
    --auto-detect \
    --eval012-mode filter-contains \
    --threshold-catheter 0.45 \
    --threshold-guidewire 0.70 \
    --temp-cat 1.2 \
    --temp-gw 1.6 \
    --min-area-catheter 50 \
    --min-area-guidewire 120 \
    --overlap-policy prob
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from preprocess import CatheterPreprocessingDataset
from swinunet import SwinUNet
from utils.postprocess import PostprocessConfig, logits_to_mask_postprocessed


def _default_logger():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def _auto_detect_from_summary(ckpt_path: Path) -> dict:
    summary_path = ckpt_path.parent / 'summary.json'
    if not summary_path.exists():
        return {}
    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        cfg = summary.get('training', {})
        cfg.update(summary.get('extra', {}))
        return cfg
    except Exception:
        return {}


def _filter_by_eval012_mode(image_paths, mask_paths, mode: str):
    if mode not in ['filter-contains', 'filter-exact']:
        return image_paths, mask_paths

    need_subset = mode == 'filter-contains'
    need_exact = mode == 'filter-exact'

    kept_images, kept_masks = [], []
    for img_p, msk_p in zip(image_paths, mask_paths):
        try:
            uniq = np.unique(np.load(msk_p))
        except Exception as e:
            logging.warning(f'Failed to read mask {msk_p}: {e}; skipping')
            continue
        uniq_set = set(int(x) for x in uniq.tolist())
        if need_subset and {0, 1, 2}.issubset(uniq_set):
            kept_images.append(img_p)
            kept_masks.append(msk_p)
        elif need_exact and uniq_set == {0, 1, 2}:
            kept_images.append(img_p)
            kept_masks.append(msk_p)

    dropped = len(image_paths) - len(kept_images)
    logging.info(f'Filter={mode}: kept {len(kept_images)} samples, dropped {dropped}')
    return kept_images, kept_masks


def _resolve_test_dirs(base: Path, test_set: str):
    available = []
    if base.exists():
        for p in sorted(base.iterdir()):
            if not p.is_dir() or not p.name.endswith('_test'):
                continue
            img_dir = p / 'images'
            mask_dir = p / 'masks'
            if img_dir.exists() and mask_dir.exists():
                available.append((p.name, img_dir, mask_dir))

    if not available:
        raise SystemExit(f'No valid test sets found under: {base.resolve()}')

    if test_set.lower() == 'all':
        return available, [name for name, _, _ in available]

    alias_to_name = {}
    for name, _, _ in available:
        name_l = name.lower()
        alias_to_name[name_l] = name
        if name_l.endswith('_test'):
            alias_to_name[name_l[:-5]] = name

    requested = [x.strip().lower() for x in test_set.split(',') if x.strip()]
    unknown = [x for x in requested if x not in alias_to_name]
    if unknown:
        choices = ', '.join(sorted(alias_to_name.keys()))
        raise SystemExit(f'Unknown --test-set: {unknown}. Available: {choices}')

    selected_names = []
    for key in requested:
        mapped = alias_to_name[key]
        if mapped not in selected_names:
            selected_names.append(mapped)

    selected = [item for item in available if item[0] in selected_names]
    return selected, selected_names


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained SwinUNet model')

    parser.add_argument('--model', '-m', type=str, required=True, help='Path to trained model (.pth)')
    parser.add_argument('--batch-size', '-b', type=int, default=4)
    parser.add_argument('--classes', '-c', type=int, default=3)
    parser.add_argument('--data-dir', type=str, default='./segmentation')
    parser.add_argument('--test-set', type=str, default='all',
                        help='Which test set(s) under data-dir to evaluate. Use all, or comma-separated names (e.g. animal_test,phantom_test or animal,phantom).')

    parser.add_argument('--eval012-mode', type=str, default='map',
                        choices=['off', 'map', 'filter-contains', 'filter-exact'],
                        help='off: raw labels; map: map labels>2->0; filter-contains/exact: subset then map>2->0')

    # arch
    parser.add_argument('--model-name', type=str, default=None)
    parser.add_argument('--decoder-type', type=str, default=None, choices=['cnn', 'transformer'])
    parser.add_argument('--decoder-channels', type=int, nargs=4, default=None)
    parser.add_argument('--decoder-mlp-ratio', type=float, default=None)
    parser.add_argument('--img-size', type=int, default=None)
    parser.add_argument('--auto-detect', action='store_true')
    parser.add_argument('--skip-gate', action=argparse.BooleanOptionalAction, default=None,
                        help='Use SE gate on skip features (default: from summary if auto-detect)')
    parser.add_argument('--deep-supervision', action=argparse.BooleanOptionalAction, default=None,
                        help='Enable deep supervision heads (default: from summary if auto-detect)')

    # postprocess knobs
    parser.add_argument('--threshold-catheter', type=float, default=0.0)
    parser.add_argument('--threshold-guidewire', type=float, default=0.0)
    parser.add_argument('--temp-cat', type=float, default=1.0)
    parser.add_argument('--temp-gw', type=float, default=1.0)

    parser.add_argument('--min-area-catheter', type=int, default=0)
    parser.add_argument('--min-area-guidewire', type=int, default=0)

    parser.add_argument('--skeletonize-guidewire', action='store_true')
    parser.add_argument('--gw-dilate-iter', type=int, default=1)
    parser.add_argument('--gw-dilate-ksize', type=int, default=3)
    parser.add_argument('--overlap-policy', type=str, default='prob', choices=['prob', 'gw', 'cat'])
    parser.add_argument('--tta', action='store_true', help='Enable Test Time Augmentation (h-flip, v-flip, vh-flip)')

    args = parser.parse_args()
    _default_logger()

    ckpt_path = Path(args.model)
    if not ckpt_path.exists():
        raise SystemExit(f'Model checkpoint not found: {ckpt_path.resolve()}')

    # auto-detect arch
    if args.auto_detect:
        cfg = _auto_detect_from_summary(ckpt_path)
        if args.model_name is None:
            args.model_name = cfg.get('model_name')
        if args.decoder_type is None:
            args.decoder_type = cfg.get('decoder_type', 'cnn')
        if args.decoder_channels is None:
            args.decoder_channels = cfg.get('decoder_channels', [256, 128, 64, 32])
        if args.decoder_mlp_ratio is None:
            args.decoder_mlp_ratio = cfg.get('decoder_mlp_ratio', 4.0)
        if args.img_size is None:
            args.img_size = cfg.get('img_size', 512)
        if args.skip_gate is None:
            args.skip_gate = cfg.get('skip_gate', False)
        if args.deep_supervision is None:
            args.deep_supervision = cfg.get('deep_supervision', False)
        logging.info(
            'Auto-detect: model_name=%s, decoder_type=%s, decoder_channels=%s, img_size=%s',
            args.model_name, args.decoder_type, args.decoder_channels, args.img_size,
        )

    if args.model_name is None:
        args.model_name = 'swinv2_tiny_window8_256'
    if args.decoder_type is None:
        args.decoder_type = 'cnn'
    if args.decoder_channels is None:
        args.decoder_channels = [256, 128, 64, 32]
    if args.decoder_mlp_ratio is None:
        args.decoder_mlp_ratio = 4.0
    if args.img_size is None:
        args.img_size = 512
    if args.skip_gate is None:
        args.skip_gate = False
    if args.deep_supervision is None:
        args.deep_supervision = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    if device.type == 'cuda':
        logging.info(f'GPU: {torch.cuda.get_device_name(0)}')

    model = SwinUNet(
        n_channels=3,
        n_classes=args.classes,
        model_name=args.model_name,
        decoder_type=args.decoder_type,
        decoder_channels=tuple(args.decoder_channels),
        decoder_mlp_ratio=args.decoder_mlp_ratio,
        img_size=int(args.img_size),
        use_skip_gate=bool(args.skip_gate),
        use_deep_supervision=bool(args.deep_supervision),
    ).to(device)

    # safe load
    try:
        state_dict = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(str(ckpt_path), map_location=device)

    if isinstance(state_dict, dict) and 'mask_values' in state_dict:
        del state_dict['mask_values']
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # data
    base = Path(args.data_dir)
    test_dirs, selected_sets = _resolve_test_dirs(base, args.test_set)
    logging.info(f'Selected test sets: {selected_sets}')

    image_paths, mask_paths = [], []
    for set_name, dir_img, dir_mask in test_dirs:
        imgs = sorted(list(dir_img.glob('*.png')) + list(dir_img.glob('*.jpg')) + list(dir_img.glob('*.jpeg')))
        logging.info(f'Scanning images from [{set_name}] {dir_img}...')
        for img_p in imgs:
            msk_p = dir_mask / (img_p.stem + '.npy')
            if msk_p.exists():
                image_paths.append(str(img_p))
                mask_paths.append(str(msk_p))

    if not image_paths:
        raise SystemExit('❌ No test image/mask pairs found. Check --data-dir.')

    image_paths, mask_paths = _filter_by_eval012_mode(image_paths, mask_paths, args.eval012_mode)
    logging.info(f'Total test image/mask pairs: {len(image_paths)}')

    test_dataset = CatheterPreprocessingDataset(
        image_paths=image_paths,
        mask_paths=mask_paths,
        target_size=512,
        use_clahe=False,
        mean=[0.4172, 0.4167, 0.4173],
        std=[0.2436, 0.2435, 0.2436],
        mode='val',
    )

    loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == 'cuda'),
    )

    eval_map_to_012 = args.eval012_mode in ['map', 'filter-contains', 'filter-exact']
    num_classes = 3 if eval_map_to_012 else int(args.classes)

    total_inter = torch.zeros(num_classes, device=device)
    total_union = torch.zeros(num_classes, device=device)
    total_target = torch.zeros(num_classes, device=device)
    total_pred = torch.zeros(num_classes, device=device)

    correct_pixels = 0
    total_pixels_all = 0

    total_pred_fg = 0
    total_true_fg = 0
    total_pred_cat = 0
    total_true_cat = 0
    total_pred_gw = 0
    total_true_gw = 0

    pp_cfg = PostprocessConfig(
        t_cat=float(args.threshold_catheter),
        t_gw=float(args.threshold_guidewire),
        temp_cat=float(args.temp_cat),
        temp_gw=float(args.temp_gw),
        min_area_cat=int(args.min_area_catheter),
        min_area_gw=int(args.min_area_guidewire),
        skeletonize_gw=bool(args.skeletonize_guidewire),
        gw_dilate_iter=int(args.gw_dilate_iter),
        gw_dilate_ksize=int(args.gw_dilate_ksize),
        overlap_policy=str(args.overlap_policy),
    )

    logging.info('=' * 60)
    logging.info('Starting per-class evaluation...')
    logging.info('=' * 60)

    with torch.inference_mode():
        for batch in tqdm(loader, desc='Evaluating', unit='batch'):
            images = batch['image'].to(device, non_blocking=True)
            true_masks = batch['mask'].to(device, non_blocking=True)

            with torch.autocast(device.type, enabled=(device.type == 'cuda')):
                if args.tta:
                    logits_orig = model(images)
                    if isinstance(logits_orig, (tuple, list)): logits_orig = logits_orig[0]

                    logits_h = model(torch.flip(images, dims=[3]))
                    if isinstance(logits_h, (tuple, list)): logits_h = logits_h[0]
                    logits_h = torch.flip(logits_h, dims=[3])

                    logits_v = model(torch.flip(images, dims=[2]))
                    if isinstance(logits_v, (tuple, list)): logits_v = logits_v[0]
                    logits_v = torch.flip(logits_v, dims=[2])

                    logits_hv = model(torch.flip(images, dims=[2, 3]))
                    if isinstance(logits_hv, (tuple, list)): logits_hv = logits_hv[0]
                    logits_hv = torch.flip(logits_hv, dims=[2, 3])

                    logits = (logits_orig + logits_h + logits_v + logits_hv) / 4.0
                else:
                    logits = model(images)
                    if isinstance(logits, (tuple, list)):
                        logits = logits[0]

            if logits.shape[-2:] != true_masks.shape[-2:]:
                logits = F.interpolate(logits, size=true_masks.shape[-2:], mode='bilinear', align_corners=False)

            preds_np = logits_to_mask_postprocessed(logits, pp_cfg)
            preds = torch.from_numpy(preds_np).to(device=device, dtype=true_masks.dtype)

            if eval_map_to_012:
                true_masks = true_masks.clone()
                true_masks[true_masks > 2] = 0

            correct_pixels += (preds == true_masks).sum().item()
            total_pixels_all += true_masks.numel()

            total_pred_fg += (preds > 0).sum().item()
            total_true_fg += (true_masks > 0).sum().item()
            total_pred_cat += (preds == 1).sum().item()
            total_true_cat += (true_masks == 1).sum().item()
            total_pred_gw += (preds == 2).sum().item()
            total_true_gw += (true_masks == 2).sum().item()

            for c in range(num_classes):
                pred_c = (preds == c)
                targ_c = (true_masks == c)
                total_inter[c] += (pred_c & targ_c).sum()
                total_union[c] += (pred_c | targ_c).sum()
                total_target[c] += targ_c.sum()
                total_pred[c] += pred_c.sum()

    eps = 1e-6
    total_target_f = total_target.float()
    total_pred_f = total_pred.float()
    class_dice = (2 * total_inter.float()) / (total_pred_f + total_target_f + eps)
    class_iou = total_inter.float() / (total_union.float() + eps)

    mean_dice = class_dice[1:].mean().item() if num_classes > 1 else float(class_dice.mean().item())
    miou = class_iou[1:].mean().item() if num_classes > 1 else float(class_iou.mean().item())
    mean_jaccard = class_iou.mean().item()

    global_acc = correct_pixels / max(total_pixels_all, 1)
    fg_acc = (total_inter[1:].float() / (total_target_f[1:] + eps)).mean().item() if num_classes > 1 else global_acc

    pred_fg_ratio = total_pred_fg / max(total_true_fg, 1)
    pred_cat_ratio = total_pred_cat / max(total_true_cat, 1)
    pred_gw_ratio = total_pred_gw / max(total_true_gw, 1)

    # Baseline comparison (CathAction Table IV SwinUNet)
    b_dice = 0.6126
    b_jaccard = 0.5954
    b_miou = 0.3953
    b_acc = 0.7660

    d_dice = (mean_dice - b_dice) / b_dice * 100
    d_jaccard = (mean_jaccard - b_jaccard) / b_jaccard * 100
    d_miou = (miou - b_miou) / b_miou * 100
    d_acc = (fg_acc - b_acc) / b_acc * 100

    logging.info('  Comparison with baseline SwinUNet [3] (CathAction, Table IV):')
    logging.info('  Metric                          Baseline       Yours       Delta')
    logging.info('  ----------------------------------------------------------')
    logging.info(f'  Dice Score                        {b_dice:.4f}      {mean_dice:.4f}      {d_dice:>+6.1f}%')
    logging.info(f'  Jaccard Index                     {b_jaccard:.4f}      {mean_jaccard:.4f}      {d_jaccard:>+6.1f}%')
    logging.info(f'  mIoU                              {b_miou:.4f}      {miou:.4f}      {d_miou:>+6.1f}%')
    logging.info(f'  Accuracy                          {b_acc:.4f}      {fg_acc:.4f}      {d_acc:>+6.1f}%')
    logging.info('============================================================')
    logging.info('============================================================')
    logging.info('OVERALL METRICS (foreground average, excluding background)')
    logging.info('============================================================')
    logging.info(f'  Global Pixel Accuracy       : {global_acc:.4f}  (background included, ref only)')
    logging.info(f'  Mean Dice Score             : {mean_dice:.4f}')
    logging.info(f'  Mean Jaccard Index          : {mean_jaccard:.4f}  (includes background)')
    logging.info(f'  Mean IoU (mIoU)             : {miou:.4f}')
    logging.info(f'  Foreground Mean Accuracy    : {fg_acc:.4f}  (avg over classes 1..C-1)')
    logging.info(f'  pred_fg_ratio               : {pred_fg_ratio:.3f}  (pred_fg / true_fg, ~1 is healthy)')
    logging.info('============================================================')

    logging.info('')
    logging.info('============================================================')
    logging.info('PER-CLASS DETAIL')
    logging.info('============================================================')
    logging.info('  Class                           Accuracy   Precision        Dice     Jaccard         IoU    Pixel%')
    logging.info('  ------------------------------------------------------------------------------------------')

    total_pixels = float(total_pixels_all)
    names = ['Background', 'Catheter', 'Guidewire']
    for c in range(num_classes):
        inter = float(total_inter[c].item())
        pred = float(total_pred[c].item())
        targ = float(total_target[c].item())

        acc_c = inter / (targ + eps)
        prec_c = inter / (pred + eps)
        dice_c = float(class_dice[c].item())
        iou_c = float(class_iou[c].item())
        pix_pct = (targ / max(total_pixels, 1.0)) * 100.0

        name = names[c] if c < len(names) else f'Class{c}'
        logging.info(
            f'  {name:<30s} {acc_c:>11.4f} {prec_c:>11.4f} {dice_c:>11.4f} {iou_c:>11.4f} {iou_c:>11.4f} {pix_pct:>8.2f}%'
        )


if __name__ == '__main__':
    main()

