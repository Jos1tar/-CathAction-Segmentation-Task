import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class WeightedCEDiceLoss(nn.Module):
    """
    0.3 * CrossEntropyLoss(class_weights) + 0.7 * MulticlassDiceLoss(class_weights)

    class_weights: [bg=0.05, catheter=1.5, guidewire=4.0]
    - bg 0.05  : slightly higher than before (0.01) to penalise false-positive noise
    - catheter 1.5: compensate for v5 catheter Dice drop (0.559->0.501)
    - guidewire 4.0: slightly relaxed from 5.0, guidewire already improved from 0.30->0.49
    Dice weight raised to 0.7 to enforce shape/contour purity on thin lines.
    """
    DEFAULT_WEIGHTS = [0.05, 1.5, 4.0]

    def __init__(self, class_weights=None, ce_weight: float = 0.3, dice_weight: float = 0.7,
                 epsilon: float = 1e-6):
        super().__init__()
        self.ce_w    = ce_weight
        self.dice_w  = dice_weight
        self.epsilon = epsilon

        w = torch.tensor(class_weights or self.DEFAULT_WEIGHTS, dtype=torch.float32)
        self.register_buffer('class_weights', w)

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        inputs : [B, C, H, W]  logits
        targets: [B, H, W]     long
        """
        n_classes = inputs.shape[1]

        # ── CrossEntropy with class weights ──────────────────────────────
        w = self.class_weights.to(inputs.device)
        ce_loss = F.cross_entropy(inputs, targets, weight=w)

        # ── Weighted Multiclass Dice ─────────────────────────────────────
        probs      = F.softmax(inputs, dim=1)                          # [B,C,H,W]
        targets_oh = F.one_hot(targets, n_classes).permute(0, 3, 1, 2).float()  # [B,C,H,W]

        dice_per_class = []
        for c in range(n_classes):
            p = probs[:, c]
            t = targets_oh[:, c]
            inter = (p * t).sum()
            union = p.sum() + t.sum()
            dice_c = (2 * inter + self.epsilon) / (union + self.epsilon)
            dice_per_class.append(dice_c)

        # Weighted average: normalize weights to sum to 1.
        w_norm   = self.class_weights / self.class_weights.sum()
        dice_loss_val = 1.0 - sum(w_norm[c] * dice_per_class[c] for c in range(n_classes))

        return self.ce_w * ce_loss + self.dice_w * dice_loss_val


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss.

    This is mainly here for extreme class imbalance (e.g. thin catheter/guidewire).
    Intuition: based on CrossEntropy, but it down-weights pixels the model already
    predicts confidently, so training focuses more on the hard foreground pixels.

    Args:
        gamma: focusing parameter. Larger -> more focus on hard examples.
        alpha: optional per-class weights, shape=[num_classes]. If None, no extra
            class weighting is applied.
        reduction: 'mean' | 'sum' | 'none'
    """
    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = 'mean'):
        super().__init__()
        self.gamma     = gamma
        self.reduction = reduction
        # Register alpha as a buffer so it follows .to(device) automatically.
        if alpha is not None:
            self.register_buffer('alpha', torch.as_tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        inputs : [B, C, H, W] raw logits (no softmax)
        targets: [B, H, W] class indices (long)
        """
        # Per-pixel log-probabilities.
        log_prob = F.log_softmax(inputs, dim=1)          # [B, C, H, W]
        prob     = log_prob.exp()                         # [B, C, H, W]

        # Pick log_prob/prob at the ground-truth class for each pixel.
        targets_expanded = targets.unsqueeze(1)           # [B, 1, H, W]
        log_pt = log_prob.gather(1, targets_expanded).squeeze(1)   # [B, H, W]
        pt     = prob.gather(1, targets_expanded).squeeze(1)       # [B, H, W]

        # Focal weight: (1 - pt)^gamma. Higher pt (more confident) -> smaller weight.
        focal_weight = (1.0 - pt) ** self.gamma           # [B, H, W]
        loss = -focal_weight * log_pt                      # [B, H, W]

        # Optional: per-class weights.
        if self.alpha is not None:
            alpha_t = self.alpha[targets]                  # [B, H, W]
            loss = alpha_t * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class TverskyLoss(nn.Module):
    """Multi-class Tversky loss.

    Tversky index per class:
        TI = (TP + eps) / (TP + alpha*FP + beta*FN + eps)
        loss = 1 - TI

    Notes:
    - alpha controls FP penalty; beta controls FN penalty.
    - For thin objects where recall matters, set beta > alpha.
    - Supports optional class weights and ignoring background.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        epsilon: float = 1e-6,
        class_weights=None,
        ignore_index: int = 0,
        include_background: bool = False,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eps = float(epsilon)
        self.ignore_index = int(ignore_index)
        self.include_background = bool(include_background)

        if class_weights is not None:
            w = torch.as_tensor(class_weights, dtype=torch.float32)
            self.register_buffer('class_weights', w)
        else:
            self.class_weights = None

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """inputs: [B,C,H,W] logits; targets: [B,H,W] long"""
        n_classes = inputs.shape[1]
        probs = F.softmax(inputs, dim=1)
        targets_oh = F.one_hot(targets, n_classes).permute(0, 3, 1, 2).float()

        # Ensure weights (if any) are on the same device
        cw = self.class_weights.to(device=inputs.device) if self.class_weights is not None else None

        if self.include_background:
            class_ids = list(range(n_classes))
        else:
            class_ids = [c for c in range(n_classes) if c != self.ignore_index]
            if not class_ids:
                return inputs.new_tensor(0.0)

        losses = []
        weights = []
        for c in class_ids:
            p = probs[:, c]
            t = targets_oh[:, c]

            tp = (p * t).sum()
            fp = (p * (1 - t)).sum()
            fn = ((1 - p) * t).sum()

            ti = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
            losses.append(1.0 - ti)

            if cw is not None:
                weights.append(cw[c])
            else:
                weights.append(inputs.new_tensor(1.0))

        w = torch.stack(weights)
        w = w / (w.sum() + self.eps)
        l = torch.stack(losses)
        return (w * l).sum()


class FocalTverskyLoss(nn.Module):
    """Focal Tversky loss (multi-class).

    loss = (1 - TI) ** gamma

    - gamma > 1 focuses on hard pixels (useful for tiny guidewire).
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 1.33,
        epsilon: float = 1e-6,
        class_weights=None,
        ignore_index: int = 0,
        include_background: bool = False,
    ):
        super().__init__()
        self.base = TverskyLoss(
            alpha=alpha,
            beta=beta,
            epsilon=epsilon,
            class_weights=class_weights,
            ignore_index=ignore_index,
            include_background=include_background,
        )
        self.gamma = float(gamma)

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        # base returns a weighted sum of (1 - TI) per class, so we can't just power it.
        # We recompute per-class TI here for correctness.
        n_classes = inputs.shape[1]
        probs = F.softmax(inputs, dim=1)
        targets_oh = F.one_hot(targets, n_classes).permute(0, 3, 1, 2).float()

        if self.base.include_background:
            class_ids = list(range(n_classes))
        else:
            class_ids = [c for c in range(n_classes) if c != self.base.ignore_index]
            if not class_ids:
                return inputs.new_tensor(0.0)

        losses = []
        weights = []
        cw = self.base.class_weights.to(device=inputs.device) if getattr(self.base, 'class_weights', None) is not None else None
        for c in class_ids:
            p = probs[:, c]
            t = targets_oh[:, c]

            tp = (p * t).sum()
            fp = (p * (1 - t)).sum()
            fn = ((1 - p) * t).sum()

            ti = (tp + self.base.eps) / (tp + self.base.alpha * fp + self.base.beta * fn + self.base.eps)
            losses.append((1.0 - ti) ** self.gamma)

            if cw is not None:
                weights.append(cw[c])
            else:
                weights.append(inputs.new_tensor(1.0))

        w = torch.stack(weights)
        w = w / (w.sum() + self.base.eps)
        l = torch.stack(losses)
        return (w * l).sum()


def _soft_erode(img: Tensor) -> Tensor:
    """Differentiable soft erosion for 2D maps.
    img: [B,1,H,W] in [0,1]
    """
    p1 = -F.max_pool2d(-img, kernel_size=(3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-img, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)


def _soft_dilate(img: Tensor) -> Tensor:
    """Differentiable soft dilation for 2D maps."""
    p1 = F.max_pool2d(img, kernel_size=(3, 1), stride=1, padding=(1, 0))
    p2 = F.max_pool2d(img, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.max(p1, p2)


def _soft_open(img: Tensor) -> Tensor:
    return _soft_dilate(_soft_erode(img))


def soft_skel(img: Tensor, iters: int = 10) -> Tensor:
    """Soft skeletonization (differentiable) from clDice paper implementation style.

    img: [B,1,H,W] probability-like map in [0,1]
    """
    img = img.clamp(0, 1)
    opened = _soft_open(img)
    skel = F.relu(img - opened)
    for _ in range(iters):
        img = _soft_erode(img)
        opened = _soft_open(img)
        delta = F.relu(img - opened)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def _soft_cldice(p: Tensor, t: Tensor, iters: int = 10, eps: float = 1e-6) -> Tensor:
    """Soft clDice between two binary/probability maps.

    p,t: [B,1,H,W] in [0,1]
    Returns: scalar tensor
    """
    skel_p = soft_skel(p, iters=iters)
    skel_t = soft_skel(t, iters=iters)

    # topology precision / sensitivity
    tprec = (skel_p * t).sum(dim=(1, 2, 3)) / (skel_p.sum(dim=(1, 2, 3)) + eps)
    tsens = (skel_t * p).sum(dim=(1, 2, 3)) / (skel_t.sum(dim=(1, 2, 3)) + eps)
    cl = (2 * tprec * tsens) / (tprec + tsens + eps)
    return cl.mean()


class SoftCLDiceLoss(nn.Module):
    """Soft clDice loss for (multi-)class segmentation.

    - For multi-class: computes clDice on selected foreground classes and averages (optionally weighted).
    - Assumes class 0 is background.

    Inputs:
        - inputs:  [B, C, H, W] logits
        - targets: [B, H, W] long
    """

    def __init__(
        self,
        iters: int = 10,
        class_weights=None,
        include_background: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.iters = int(iters)
        self.include_background = bool(include_background)
        self.eps = float(eps)
        if class_weights is not None:
            w = torch.as_tensor(class_weights, dtype=torch.float32)
            self.register_buffer('class_weights', w)
        else:
            self.class_weights = None

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        n_classes = inputs.shape[1]
        probs = F.softmax(inputs, dim=1)
        targets_oh = F.one_hot(targets, n_classes).permute(0, 3, 1, 2).float()

        if self.include_background:
            class_ids = list(range(n_classes))
        else:
            class_ids = list(range(1, n_classes))
            if not class_ids:
                return inputs.new_tensor(0.0)

        weights = None
        if self.class_weights is not None:
            weights = self.class_weights.to(device=inputs.device)

        cl_list = []
        w_list = []
        for c in class_ids:
            p = probs[:, c:c+1]  # [B,1,H,W]
            t = targets_oh[:, c:c+1]
            cl = _soft_cldice(p, t, iters=self.iters, eps=self.eps)
            cl_list.append(cl)
            w_list.append(weights[c] if weights is not None else inputs.new_tensor(1.0))

        w = torch.stack(w_list)
        w = w / (w.sum() + self.eps)
        cl_mean = (w * torch.stack(cl_list)).sum()
        return 1.0 - cl_mean


class BCEDiceCLDiceLoss(nn.Module):
    """CrossEntropy + Dice + clDice (as an auxiliary term).

    This combo is mainly for thin/elongated structures (e.g. guidewire), where
    plain losses can lead to breaks (under-segmentation) or overly thick masks.

    - CE: pixel-wise supervision
    - Dice: overlap/foreground emphasis
    - clDice: connectivity/topology encouragement (usually with a smaller weight: 0.1~0.7)

    Inputs:
        - inputs:  [B, C, H, W] logits
        - targets: [B, H, W] long
    """

    def __init__(
        self,
        class_weights=None,
        ce_weight: float = 0.3,
        dice_weight: float = 0.7,
        cldice_weight: float = 0.5,
        cldice_iters: int = 10,
        include_background: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)
        self.cldice_weight = float(cldice_weight)
        self.eps = float(eps)

        if class_weights is not None:
            w = torch.as_tensor(class_weights, dtype=torch.float32)
            self.register_buffer('class_weights', w)
        else:
            self.class_weights = None

        self.cldice = SoftCLDiceLoss(
            iters=int(cldice_iters),
            class_weights=class_weights,
            include_background=bool(include_background),
            eps=float(eps),
        )
        self.include_background = bool(include_background)

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        n_classes = inputs.shape[1]

        # CE
        w = self.class_weights.to(inputs.device) if self.class_weights is not None else None
        ce = F.cross_entropy(inputs, targets, weight=w)

        # Dice (soft, per-class)
        probs = F.softmax(inputs, dim=1)
        targets_oh = F.one_hot(targets, n_classes).permute(0, 3, 1, 2).float()

        class_ids = list(range(n_classes)) if self.include_background else list(range(1, n_classes))
        if not class_ids:
            dice_term = inputs.new_tensor(0.0)
        else:
            dice_list = []
            w_list = []
            for c in class_ids:
                p = probs[:, c]
                t = targets_oh[:, c]
                inter = (p * t).sum(dim=(1, 2))
                union = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
                dc = (2 * inter + self.eps) / (union + self.eps)  # [B]
                dice_list.append(dc.mean())

                if w is not None:
                    w_list.append(w[c])
                else:
                    w_list.append(inputs.new_tensor(1.0))

            ww = torch.stack(w_list)
            ww = ww / (ww.sum() + self.eps)
            dice_mean = (ww * torch.stack(dice_list)).sum()
            dice_term = 1.0 - dice_mean

        # clDice
        cl = self.cldice(inputs, targets)

        return self.ce_weight * ce + self.dice_weight * dice_term + self.cldice_weight * cl

