# coding: utf-8
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import dice_coeff


# class index: 0=background, 1=catheter, 2=guidewire
CLASS_NAMES = ['background', 'catheter', 'guidewire']


def _dice_per_class(pred_onehot, true_onehot, c, epsilon=1e-6):
    """Dice coefficient for class c."""
    p = pred_onehot[:, c].reshape(-1)
    t = true_onehot[:, c].reshape(-1)
    inter = (p * t).sum()
    return (2 * inter + epsilon) / (p.sum() + t.sum() + epsilon)


def _iou_per_class(pred_onehot, true_onehot, c, epsilon=1e-6):
    """IoU (Jaccard) for class c."""
    p = pred_onehot[:, c].reshape(-1)
    t = true_onehot[:, c].reshape(-1)
    inter = (p * t).sum()
    union = p.sum() + t.sum() - inter
    union = torch.where(union == 0, inter, union)
    return (inter + epsilon) / (union + epsilon)


def global_pixel_accuracy(pred_classes, true_classes):
    """Global pixel accuracy including background. Matches paper Table IV."""
    correct = (pred_classes == true_classes).sum()
    return correct.float() / true_classes.numel()


def foreground_pixel_accuracy(pred_classes, true_classes, ignore_index=0):
    """Foreground-only pixel accuracy (excludes background class 0).
    Supplementary metric to avoid inflation from background dominance."""
    foreground_mask = (true_classes != ignore_index)
    if foreground_mask.sum() == 0:
        return torch.tensor(1.0)
    correct = ((pred_classes == true_classes) & foreground_mask).sum()
    return correct.float() / foreground_mask.sum().float()


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    """
    Evaluate the model on the validation set.

    Primary metrics (matches paper Table IV):
        dice_score  -- mean foreground Dice (ignores background, avg of class 1+2)
        jaccard     -- mean foreground Jaccard (ignores background)
        miou        -- mean IoU over all classes including background (paper protocol)
        accuracy    -- global pixel accuracy including background (paper protocol)

    Supplementary per-class metrics:
        dice_catheter / dice_guidewire
        iou_catheter  / iou_guidewire
        foreground_accuracy -- accuracy on foreground pixels only
    """
    # Force eval mode and verify it stays enabled.
    net.eval()
    assert not net.training, "Model should be in eval mode!"

    num_val_batches = len(dataloader)
    n_classes = net.n_classes

    device_ = next(net.parameters()).device
    total_dice        = torch.tensor(0.0, device=device_)
    total_jaccard     = torch.tensor(0.0, device=device_)
    total_miou        = torch.tensor(0.0, device=device_)
    total_accuracy    = torch.tensor(0.0, device=device_)
    total_fg_accuracy = torch.tensor(0.0, device=device_)

    per_class_dice    = [0.0] * n_classes
    per_class_jaccard = [0.0] * n_classes

    # Track foreground pixels to verify the model is actually predicting foreground.
    total_pred_fg = 0
    total_true_fg = 0

    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches,
                          desc='Validation', unit='batch', leave=False):
            image     = batch['image'].to(device=device, dtype=torch.float32,
                                          memory_format=torch.channels_last)
            mask_true = batch['mask'].to(device=device, dtype=torch.long)

            mask_pred = net(image)
            if isinstance(mask_pred, (tuple, list)):
                mask_pred = mask_pred[0]
            if mask_pred.shape[-2:] != mask_true.shape[-2:]:
                mask_pred = F.interpolate(
                    mask_pred,
                    size=mask_true.shape[-2:],
                    mode='bilinear',
                    align_corners=False,
                )

            if n_classes == 1:
                # binary segmentation
                mask_pred_bin = (F.sigmoid(mask_pred) > 0.5).float()
                mask_true_flt = mask_true.float()
                inter = (mask_pred_bin * mask_true_flt).sum()
                union = mask_pred_bin.sum() + mask_true_flt.sum() - inter
                iou   = (inter + 1e-6) / (torch.where(union == 0, inter, union) + 1e-6)

                total_dice     += dice_coeff(mask_pred_bin, mask_true_flt, reduce_batch_first=False)
                total_jaccard  += iou
                total_accuracy += foreground_pixel_accuracy((mask_pred_bin > 0.5).long(), mask_true)

            else:
                # multi-class: background / catheter / guidewire
                assert mask_true.min() >= 0 and mask_true.max() < n_classes

                mask_pred_cls    = mask_pred.argmax(dim=1)

                # Track foreground pixels.
                total_pred_fg += (mask_pred_cls > 0).sum().item()
                total_true_fg += (mask_true > 0).sum().item()

                mask_true_onehot = F.one_hot(mask_true,     n_classes).permute(0, 3, 1, 2).float()
                mask_pred_onehot = F.one_hot(mask_pred_cls, n_classes).permute(0, 3, 1, 2).float()

                # foreground Dice / Jaccard (ignore background, matches paper DiceScore/JaccardIndex)
                fg_dice    = 0.0
                fg_jaccard = 0.0
                n_fg = n_classes - 1
                for c in range(1, n_classes):
                    d = _dice_per_class(mask_pred_onehot, mask_true_onehot, c)
                    j = _iou_per_class( mask_pred_onehot, mask_true_onehot, c)
                    fg_dice    += d
                    fg_jaccard += j
                    per_class_dice[c]    += d.item()
                    per_class_jaccard[c] += j.item()
                total_dice    += fg_dice    / n_fg
                total_jaccard += fg_jaccard / n_fg

                # all-class mIoU including background (matches paper mIoU)
                all_iou = 0.0
                for c in range(n_classes):
                    all_iou += _iou_per_class(mask_pred_onehot, mask_true_onehot, c)
                total_miou += all_iou / n_classes

                # global pixel accuracy (matches paper Accuracy)
                total_accuracy    += global_pixel_accuracy(mask_pred_cls, mask_true)
                # foreground-only accuracy (supplementary)
                total_fg_accuracy += foreground_pixel_accuracy(mask_pred_cls, mask_true)

    net.train()

    n = max(num_val_batches, 1)

    # Compute the foreground prediction ratio as a sanity check.
    fg_pred_ratio = total_pred_fg / max(total_true_fg, 1)

    return {
        # primary metrics matching paper Table IV
        'dice_score': total_dice.item()        / n,
        'jaccard':    total_jaccard.item()     / n,
        'miou':       total_miou.item()        / n,
        'accuracy':   total_accuracy.item()    / n,
        # supplementary per-class metrics
        'foreground_accuracy': total_fg_accuracy.item() / n,
        'dice_catheter':  per_class_dice[1]    / n if n_classes > 1 else None,
        'dice_guidewire': per_class_dice[2]    / n if n_classes > 2 else None,
        'iou_catheter':   per_class_jaccard[1] / n if n_classes > 1 else None,
        'iou_guidewire':  per_class_jaccard[2] / n if n_classes > 2 else None,
        # Foreground pixel statistics for validation and debugging.
        'pred_fg_pixels': total_pred_fg,
        'true_fg_pixels': total_true_fg,
        'fg_pred_ratio':  fg_pred_ratio,  # Should be close to 1.0; values < 0.5 suggest severe under-prediction of foreground.
    }
