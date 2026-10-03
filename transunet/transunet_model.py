# coding: utf-8
"""
TransUNet — full model assembly (R50-ViT-B/16, 512x512 input).

Data flow (spatial sizes for 512×512 input):
  [B,3,512,512]
  -> HybridEncoder
       skip0 [B, 64,  256, 256]  (after stem, before maxpool)
       skip1 [B,256,  128, 128]  (after layer1)
       skip2 [B,512,   64,  64]  (after layer2)
       feat  [B,hidden, 32,  32]  (after layer3 + proj)
  -> TransformerEncoder (num_layers, hidden=hidden_size, heads=num_heads)
  -> reshape [B, hidden_size, 32, 32]
  -> dec1(hidden_size -> d[0], skip=x2[512],  32-> 64)
  -> dec2(d[0]        -> d[1], skip=x1[256],  64->128)
  -> dec3(d[1]        -> d[2], skip=x0[ 64], 128->256)
  -> dec4(d[2]        -> d[3], no skip,       256->512)
  -> 1x1 conv -> [B, n_classes, 512, 512]

Structural presets (analogous to UNet base_channels):
  ViT-B/16 (default, ~118M): hidden_size=768, num_layers=12, num_heads=12,
                              decoder_channels=(384, 192, 96, 48)
  ViT-S/16 (light,   ~35M) : hidden_size=384, num_layers=6,  num_heads=6,
                              decoder_channels=(192, 96,  48, 24)
"""

import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from .transunet_parts import HybridEncoder, TransformerEncoder, DecoderBlock


class TransUNet(nn.Module):
    """
    TransUNet for multi-class segmentation.

    Args:
        n_channels       : input channels (3 for RGB X-ray)
        n_classes        : output classes (3: background/catheter/guidewire)
        hidden_size      : Transformer hidden dim — controls model width
                           768 = ViT-B (default),  384 = ViT-S (lighter)
        num_heads        : attention heads — must divide hidden_size evenly
                           12 for ViT-B,  6 for ViT-S
        num_layers       : Transformer depth — controls model depth
                           12 for ViT-B,  6 for ViT-S
        decoder_channels : output channels of each of the 4 decoder stages
                           (d0, d1, d2, d3) — analogous to UNet base_channels
                           default (384,192,96,48) for ViT-B
                           use   (192, 96,48,24) for ViT-S
        img_size         : expected input spatial size (512)
    """

    # Predefined presets for convenience
    PRESETS = {
        'vit-b': dict(hidden_size=768, num_heads=12, num_layers=12,
                      decoder_channels=(384, 192, 96, 48)),   # ~118M
        'vit-s': dict(hidden_size=384, num_heads=6,  num_layers=6,
                      decoder_channels=(192,  96, 48, 24)),   # ~35M
    }

    def __init__(self,
                 n_channels:       int                = 3,
                 n_classes:        int                = 3,
                 hidden_size:      int                = 768,
                 num_heads:        int                = 12,
                 num_layers:       int                = 12,
                 decoder_channels: Tuple[int,...]     = (384, 192, 96, 48),
                 img_size:         int                = 512,
                 pretrained:       bool               = False):
        super().__init__()
        self.n_channels  = n_channels
        self.n_classes   = n_classes
        self.hidden_size = hidden_size

        assert len(decoder_channels) == 4, "decoder_channels must have exactly 4 values"
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        d0, d1, d2, d3 = decoder_channels

        # CNN hybrid encoder (skip channel sizes are fixed by ResNet-50 design):
        #   x0 ->  64ch  (stem + maxpool)
        #   x1 -> 256ch  (after layer1)
        #   x2 -> 512ch  (after layer2)
        self.encoder = HybridEncoder(hidden_size=hidden_size, pretrained=pretrained)

        # Transformer encoder (32×32 = 1024 patches for a 512-input)
        max_patches = (img_size // 16) ** 2
        self.transformer = TransformerEncoder(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            max_patches=max_patches,
        )

        # CNN decoder — tunable via decoder_channels
        # skip channels (fixed by ResNet-50 design):
        #   skip2 -> 512ch at H/8   (after layer2)
        #   skip1 -> 256ch at H/4   (after layer1)
        #   skip0 ->  64ch at H/2   (after stem, BEFORE maxpool)  ← key fix
        self.dec1 = DecoderBlock(hidden_size, 512, d0)  # H/16->H/8,  skip=skip2(512ch)
        self.dec2 = DecoderBlock(d0,          256, d1)  # H/8 ->H/4,  skip=skip1(256ch)
        self.dec3 = DecoderBlock(d1,           64, d2)  # H/4 ->H/2,  skip=skip0( 64ch)
        self.dec4 = DecoderBlock(d2,            0, d3)  # H/2 ->H/1,  no skip

        self.outc = nn.Conv2d(d3, n_classes, kernel_size=1)

    @classmethod
    def from_preset(cls, preset: str, **kwargs) -> 'TransUNet':
        """
        Convenience constructor using a named preset.

        Usage:
            model = TransUNet.from_preset('vit-b', n_classes=3)
            model = TransUNet.from_preset('vit-s', n_classes=3)
        """
        if preset not in cls.PRESETS:
            raise ValueError(f"Unknown preset '{preset}'. Choose from: {list(cls.PRESETS)}")
        cfg = {**cls.PRESETS[preset], **kwargs}
        return cls(**cfg)

    def forward(self, x):
        B = x.shape[0]
        orig_size = x.shape[2:]

        patches, skip0, skip1, skip2, pH, pW = self.encoder(x)
        encoded = self.transformer(patches)
        feat = encoded.transpose(1, 2).reshape(B, self.hidden_size, pH, pW)

        x = self.dec1(feat, skip2)
        x = self.dec2(x,    skip1)
        x = self.dec3(x,    skip0)
        x = self.dec4(x,    None)
        logits = self.outc(x)

        if logits.shape[2:] != orig_size:
            logits = F.interpolate(logits, size=orig_size, mode='bilinear', align_corners=False)
        return logits
