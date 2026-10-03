# coding: utf-8
"""
TransUNet building blocks.

Architecture (R50-ViT-B/16 style, adapted for 512x512 input):
  CNN Encoder  : ResNet-50 stem  -> layer1 -> layer2 -> layer3
  Transformer  : 12 Transformer encoder layers (ViT-B/16 config)
  CNN Decoder  : cascaded up-sampling with skip connections
  Output head  : 1x1 conv -> n_classes

Reference: Chen et al., "TransUNet: Transformers Make Strong Encoders for
           Medical Image Segmentation", arXiv:2102.04306
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Hybrid CNN Encoder (ResNet-50 backbone)
# ─────────────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """Standard ResNet bottleneck block."""
    expansion = 4

    def __init__(self, in_ch, mid_ch, stride=1, downsample=None):
        super().__init__()
        out_ch = mid_ch * self.expansion
        self.conv1      = nn.Conv2d(in_ch,  mid_ch, 1, bias=False)
        self.bn1        = nn.BatchNorm2d(mid_ch)
        self.conv2      = nn.Conv2d(mid_ch, mid_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2        = nn.BatchNorm2d(mid_ch)
        self.conv3      = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        self.bn3        = nn.BatchNorm2d(out_ch)
        self.relu       = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


def _make_layer(in_ch, mid_ch, blocks, stride=1):
    out_ch     = mid_ch * ResidualBlock.expansion
    downsample = None
    if stride != 1 or in_ch != out_ch:
        downsample = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
        )
    layers = [ResidualBlock(in_ch, mid_ch, stride, downsample)]
    for _ in range(1, blocks):
        layers.append(ResidualBlock(out_ch, mid_ch))
    return nn.Sequential(*layers)


class HybridEncoder(nn.Module):
    """
    ResNet-50 hybrid encoder.

    Input  : [B, 3, H, W]
    Returns: (patches [B, N, hidden_size],
              skip0 [B,  64, H/2,  W/2],   ← after stem only (before maxpool)
              skip1 [B, 256, H/4,  W/4],   ← after layer1
              skip2 [B, 512, H/8,  W/8],   ← after layer2
              pH, pW)  # patch-grid spatial dims (H/16, W/16)

    Spatial sizes for 512-input:
      skip0: 256×256  → used by dec3 (256 → 512)
      skip1: 128×128  → used by dec2 (128 → 256)
      skip2:  64×64   → used by dec1 ( 64 → 128)
      feat:   32×32   → enters dec1
    """

    def __init__(self, hidden_size: int = 768, pretrained: bool = False):
        super().__init__()
        # Stem: /2
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)   # /4  → 64ch

        self.layer1 = _make_layer(64,  64,  blocks=3)            # /4  → 256ch
        self.layer2 = _make_layer(256, 128, blocks=4, stride=2)  # /8  → 512ch
        self.layer3 = _make_layer(512, 256, blocks=6, stride=2)  # /16 → 1024ch

        # project 1024 → hidden_size
        self.proj = nn.Conv2d(1024, hidden_size, 1, bias=False)

        if pretrained:
            self._load_pretrained_resnet()

    def _load_pretrained_resnet(self):
        import torchvision.models as models
        import logging
        logging.info("Loading ImageNet pretrained weights into HybridEncoder (ResNet-50)...")
        # Load torchvision resnet50
        resnet = models.resnet50(pretrained=True)
        # Copy stem
        self.stem[0].load_state_dict(resnet.conv1.state_dict())
        self.stem[1].load_state_dict(resnet.bn1.state_dict())
        # Copy layers
        self.layer1.load_state_dict(resnet.layer1.state_dict())
        self.layer2.load_state_dict(resnet.layer2.state_dict())
        self.layer3.load_state_dict(resnet.layer3.state_dict())
        logging.info("HybridEncoder weights successfully loaded.")

    def forward(self, x):
        x0 = self.stem(x)                # [B,  64, H/2,  W/2]   skip0 ← before maxpool
        x1 = self.layer1(self.maxpool(x0))  # [B, 256, H/4,  W/4]   skip1
        x2 = self.layer2(x1)             # [B, 512, H/8,  W/8]   skip2
        x3 = self.layer3(x2)             # [B,1024, H/16, W/16]

        feat = self.proj(x3)             # [B, hidden, H/16, W/16]
        B, C, pH, pW = feat.shape
        patches = feat.flatten(2).transpose(1, 2)  # [B, pH*pW, hidden]
        return patches, x0, x1, x2, pH, pW


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Transformer Encoder
# ─────────────────────────────────────────────────────────────────────────────

class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.0):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = hidden_size // num_heads
        self.scale     = self.head_dim ** -0.5
        self.qkv  = nn.Linear(hidden_size, hidden_size * 3)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = self.drop((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(B, N, C))


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn  = Attention(hidden_size, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_size)
        mlp_dim    = int(hidden_size * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_size), nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, hidden_size=768, num_heads=12, num_layers=12,
                 mlp_ratio=4.0, dropout=0.1, max_patches=1024):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, max_patches, hidden_size))
        self.dropout   = nn.Dropout(dropout)
        self.layers    = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B, N, C = x.shape
        if N != self.pos_embed.shape[1]:
            pos = F.interpolate(
                self.pos_embed.transpose(1, 2), size=N,
                mode='linear', align_corners=False,
            ).transpose(1, 2)
        else:
            pos = self.pos_embed
        x = self.dropout(x + pos)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  CNN Decoder with skip connections
# ─────────────────────────────────────────────────────────────────────────────

class DecoderBlock(nn.Module):
    """ConvTranspose2d ×2  +  optional skip concat  +  DoubleConv."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up  = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        mid_ch   = (in_ch // 2) + skip_ch
        self.conv = nn.Sequential(
            nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            dH = skip.shape[2] - x.shape[2]
            dW = skip.shape[3] - x.shape[3]
            x  = F.pad(x, [dW // 2, dW - dW // 2, dH // 2, dH - dH // 2])
            x  = torch.cat([skip, x], dim=1)
        return self.conv(x)

