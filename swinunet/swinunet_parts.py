# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(1, channels // reduction)
        # A classic Squeeze-and-Excitation (SE) gate:
        # 1) squeeze: global average pool -> [B, C, 1, 1]
        # 2) excite: small MLP -> per-channel weights in (0, 1)
        # We use it as a lightweight way to reweight skip features.
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(self.avg(x))
        return x * w


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_skip_gate: bool = False):
        super().__init__()
        self.skip_channels = skip_channels
        # "skip gate" here means: apply a channel-wise SE gate on the skip tensor
        # before concatenation. It's intentionally simple/cheap.
        self.use_skip_gate = bool(use_skip_gate) and skip_channels > 0
        self.skip_gate = SEBlock(skip_channels) if self.use_skip_gate else None
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            ConvBNReLU(in_channels + skip_channels, out_channels),
            ConvBNReLU(out_channels, out_channels),
        )

    def forward(self, x, skip=None):
        # Decoder step: upsample, fuse with skip (if any), then refine with convs.
        x = self.up(x)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                # This happens when the backbone uses slightly different downsample rules.
                # Interpolating the upsampled branch is usually the least painful fix.
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            if self.skip_gate is not None:
                # Gate ONLY the skip features (not the upsampled features).
                skip = self.skip_gate(skip)
            x = torch.cat([skip, x], dim=1)
        return self.conv(x)

