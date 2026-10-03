# coding: utf-8
"""Post-processing utilities for catheter/guidewire segmentation.

Goals (no retraining):
- Reduce false positives (small noisy blobs) via connected-component filtering.
- Make guidewire predictions thinner and more "wire-like" via skeletonization,
  then optionally dilate back to a small width.

This module is intentionally independent from model code so it can be reused in
VAL threshold scanning and TEST evaluation.

Dependencies:
- NumPy (already in most stacks)
- OpenCV (cv2) is used for fast connected components and morphology.
- scikit-image is OPTIONAL. If present, we use `skimage.morphology.skeletonize`
  for better skeletons. If not present, we fall back to OpenCV thinning when
  available, otherwise skip skeletonization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PostprocessConfig:
    # per-class softmax thresholds
    t_cat: float = 0.0
    t_gw: float = 0.0

    # NEW: per-class temperature scaling on logits before softmax
    # T=1.0 means no scaling. T>1 makes probabilities flatter (more conservative),
    # often reduces false positives; T<1 makes them sharper (more aggressive).
    temp_cat: float = 1.0
    temp_gw: float = 1.0

    # connected-component filtering (remove small blobs)
    # set to 0 to disable
    min_area_cat: int = 0
    min_area_gw: int = 0

    # guidewire skeletonization + dilation
    skeletonize_gw: bool = False
    # dilation iterations after skeletonization (0 disables)
    gw_dilate_iter: int = 1
    # dilation kernel size (odd number recommended)
    gw_dilate_ksize: int = 3

    # priority when catheter & guidewire overlap after thresholding
    # 'prob' => pick higher prob per-pixel; 'gw' => force guidewire; 'cat' => force catheter
    overlap_policy: str = 'prob'


def _require_cv2():
    try:
        import cv2  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            'OpenCV (cv2) is required for post-processing. '
            'Please install: pip install opencv-python'
        ) from e
    return cv2


def _remove_small_components(mask01: np.ndarray, min_area: int) -> np.ndarray:
    """Remove connected components with area < min_area.

    mask01: uint8/bool HxW with values {0,1}
    returns: uint8 HxW {0,1}
    """
    if min_area <= 0:
        return (mask01 > 0).astype(np.uint8)

    cv2 = _require_cv2()
    mask_u8 = (mask01 > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    if num_labels <= 1:
        return mask_u8

    out = np.zeros_like(mask_u8)
    # label 0 is background
    for lab in range(1, num_labels):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area >= min_area:
            out[labels == lab] = 1
    return out


def _skeletonize(binary01: np.ndarray) -> np.ndarray:
    """Skeletonize a binary mask.

    Uses scikit-image if available; otherwise tries OpenCV ximgproc thinning.
    If neither exists, returns the input unchanged.
    """
    bin_u8 = (binary01 > 0).astype(np.uint8)
    if bin_u8.max() == 0:
        return bin_u8

    # 1) prefer skimage
    try:
        from skimage.morphology import skeletonize as sk_skeletonize  # type: ignore

        sk = sk_skeletonize(bin_u8.astype(bool))
        return sk.astype(np.uint8)
    except Exception:
        pass

    # 2) fallback to OpenCV thinning if available
    try:
        cv2 = _require_cv2()
        if hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'thinning'):
            return cv2.ximgproc.thinning(bin_u8 * 255, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN) // 255
    except Exception:
        pass

    # 3) no skeletonization available
    return bin_u8


def _dilate(binary01: np.ndarray, ksize: int, iterations: int) -> np.ndarray:
    if iterations <= 0:
        return (binary01 > 0).astype(np.uint8)
    cv2 = _require_cv2()
    k = max(1, int(ksize))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    out = cv2.dilate((binary01 > 0).astype(np.uint8), kernel, iterations=int(iterations))
    return (out > 0).astype(np.uint8)


def logits_to_mask_postprocessed(
    logits: 'np.ndarray | "torch.Tensor"',
    config: PostprocessConfig,
    *,
    return_debug: bool = False,
):
    """Convert logits (C,H,W) or (B,C,H,W) to label mask with post-processing.

    This function expects *multiclass* logits with C>=3 (bg/cat/gw).

    Returns:
      mask: (H,W) or (B,H,W) uint8 labels in {0,1,2}
    """
    # allow torch tensor input without importing torch globally
    is_torch = False
    try:
        import torch  # type: ignore

        if isinstance(logits, torch.Tensor):
            is_torch = True
            lg = logits.detach().float().cpu()
            # -> numpy
            arr = lg.numpy()
        else:
            arr = np.asarray(logits)
    except Exception:
        arr = np.asarray(logits)

    input_had_batch = True
    if arr.ndim == 3:
        input_had_batch = False
        arr = arr[None, ...]  # B=1

    if arr.ndim != 4:
        raise ValueError(f'Expected logits with shape (B,C,H,W) or (C,H,W), got {arr.shape}')

    b, c, h, w = arr.shape
    if c < 3:
        raise ValueError(f'Postprocess expects >=3 classes (bg/cat/gw), got C={c}')

    # --- per-class temperature scaling (logit calibration) ---
    # We only scale foreground logits (cat/gw). Background stays unchanged.
    # NOTE: this is applied BEFORE softmax, so it changes probabilities.
    temp_cat = float(getattr(config, 'temp_cat', 1.0) or 1.0)
    temp_gw = float(getattr(config, 'temp_gw', 1.0) or 1.0)
    # safety: avoid division by 0 / negative
    temp_cat = max(temp_cat, 1e-6)
    temp_gw = max(temp_gw, 1e-6)

    arr = arr.copy()
    arr[:, 1, :, :] = arr[:, 1, :, :] / temp_cat
    arr[:, 2, :, :] = arr[:, 2, :, :] / temp_gw

    # softmax
    ex = np.exp(arr - arr.max(axis=1, keepdims=True))
    probs = ex / (ex.sum(axis=1, keepdims=True) + 1e-12)
    p0 = probs[:, 0]
    p1 = probs[:, 1]
    p2 = probs[:, 2]

    t_cat = float(config.t_cat or 0.0)
    t_gw = float(config.t_gw or 0.0)

    cat = ((p1 > t_cat) if t_cat > 0 else (p1 >= p0)).astype(np.uint8)
    gw = ((p2 > t_gw) if t_gw > 0 else (p2 >= p0)).astype(np.uint8)

    # resolve overlap
    both = (cat > 0) & (gw > 0)
    if both.any():
        if config.overlap_policy == 'gw':
            cat[both] = 0
        elif config.overlap_policy == 'cat':
            gw[both] = 0
        else:  # 'prob'
            pick_gw = np.asarray(p2[both] > p1[both])
            cat[both] = 0
            gw[both] = 0
            gw[both] = pick_gw.astype(np.uint8)
            cat[both] = (~pick_gw).astype(np.uint8)

    # connected component filtering
    cat_pp = np.stack([_remove_small_components(cat[i], int(config.min_area_cat)) for i in range(b)], axis=0)
    gw_pp = np.stack([_remove_small_components(gw[i], int(config.min_area_gw)) for i in range(b)], axis=0)

    # skeletonize + dilate guidewire
    if bool(config.skeletonize_gw):
        gw_sk = np.stack([_skeletonize(gw_pp[i]) for i in range(b)], axis=0)
        gw_pp = np.stack([
            _dilate(gw_sk[i], int(config.gw_dilate_ksize), int(config.gw_dilate_iter))
            for i in range(b)
        ], axis=0)

    # compose labels, guidewire overrides catheter by default
    out = np.zeros((b, h, w), dtype=np.uint8)
    out[cat_pp > 0] = 1
    out[gw_pp > 0] = 2

    def _maybe_squeeze(x: np.ndarray) -> np.ndarray:
        return x[0] if (not input_had_batch and x.shape[0] == 1) else x

    if return_debug:
        dbg = {
            'cat_pixels': int(cat_pp.sum()),
            'gw_pixels': int(gw_pp.sum()),
            'temp_cat': float(temp_cat),
            'temp_gw': float(temp_gw),
        }
        return _maybe_squeeze(out), dbg

    return _maybe_squeeze(out)
