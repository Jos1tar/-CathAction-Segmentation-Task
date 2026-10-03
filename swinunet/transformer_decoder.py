# coding: utf-8
"""Transformer-based decoder blocks for SwinUNet.

This is intentionally lightweight and shape-friendly:
- We keep everything in NCHW.
- Each upsampling step aligns spatial size to the corresponding skip feature.
- Fusion is done with a small transformer stack working on flattened HW tokens.

The goal is not to perfectly reproduce any specific paper, but to provide a
reasonable "transformer-ish" alternative to the CNN decoder while keeping the
rest of the training pipeline unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .swinunet_parts import SEBlock


class FeedForward(nn.Module):
    """Simple MLP used inside transformer blocks."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block (MHSA + MLP).

    We use batch_first=True so tokens are [B, N, C].
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(drop)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim=dim, mlp_ratio=mlp_ratio, drop=proj_drop)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        xn = self.norm1(x)
        attn_out, _ = self.attn(xn, xn, xn, need_weights=False)
        x = x + self.drop1(attn_out)

        # MLP
        x = x + self.drop2(self.mlp(self.norm2(x)))
        return x


def _nchw_to_tokens(x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """Convert NCHW -> tokens [B, HW, C]."""
    b, c, h, w = x.shape
    x = x.flatten(2).transpose(1, 2).contiguous()
    return x, h, w


def _tokens_to_nchw(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Convert tokens [B, HW, C] -> NCHW."""
    b, n, c = x.shape
    if n != h * w:
        raise ValueError(f"Token count {n} does not match h*w={h*w}")
    return x.transpose(1, 2).contiguous().view(b, c, h, w)


@dataclass
class TransformerUpBlockCfg:
    """Configuration for one decoder stage."""

    in_channels: int
    skip_channels: int
    out_channels: int
    depth: int
    num_heads: int
    mlp_ratio: float = 4.0
    drop: float = 0.0
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    use_skip_gate: bool = False


class TransformerUpBlock(nn.Module):
    """Upsample + (optional) gated skip + transformer fusion.

    Compared with the CNN UpBlock:
    - We still upsample by 2.
    - We still optionally gate ONLY the skip feature (SE style).
    - Instead of conv refinement, we run a small transformer stack over tokens.

    Fusion strategy used here: concat tokens from (upsampled x) and (skip), then
    project back to out_channels.
    """

    def __init__(self, cfg: TransformerUpBlockCfg):
        super().__init__()
        self.cfg = cfg
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.use_skip_gate = bool(cfg.use_skip_gate) and cfg.skip_channels > 0
        self.skip_gate = SEBlock(cfg.skip_channels) if self.use_skip_gate else None

        fused_dim = cfg.in_channels + cfg.skip_channels
        self.fuse_proj = nn.Linear(fused_dim, cfg.out_channels)

        # Keep it small by default; decoder doesn't need to be huge.
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=cfg.out_channels,
                    num_heads=cfg.num_heads,
                    mlp_ratio=cfg.mlp_ratio,
                    attn_drop=cfg.attn_drop,
                    proj_drop=cfg.proj_drop,
                    drop=cfg.drop,
                )
                for _ in range(int(cfg.depth))
            ]
        )

        # A tiny post-norm helps keep activations stable.
        self.norm = nn.LayerNorm(cfg.out_channels)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.up(x)

        if skip is not None:
            # Same safety net as the CNN decoder: always align to skip resolution.
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            if self.skip_gate is not None:
                skip = self.skip_gate(skip)
        else:
            # No skip: treat it as an empty tensor in the concat.
            skip = None

        xt, h, w = _nchw_to_tokens(x)
        if skip is not None:
            st, _, _ = _nchw_to_tokens(skip)
            z = torch.cat([xt, st], dim=-1)
        else:
            z = xt

        z = self.fuse_proj(z)
        for blk in self.blocks:
            z = blk(z)
        z = self.norm(z)
        out = _tokens_to_nchw(z, h=h, w=w)
        return out


class TransformerDecoder(nn.Module):
    """4-stage transformer decoder for SwinUNet.

    Returns (x0, x1, x2, x3) where x0 is the final highest-resolution feature.
    """

    def __init__(
        self,
        enc_channels: Union[Sequence[int], Tuple[int, int, int, int]],
        decoder_channels: tuple[int, int, int, int] = (256, 128, 64, 32),
        depths: tuple[int, int, int, int] = (1, 1, 1, 1),
        num_heads: tuple[int, int, int, int] = (8, 8, 4, 4),
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_skip_gate: bool = False,
    ):
        super().__init__()
        if len(enc_channels) != 4:
            raise ValueError("enc_channels must have 4 elements")
        if len(decoder_channels) != 4:
            raise ValueError("decoder_channels must have 4 elements")
        if len(depths) != 4 or len(num_heads) != 4:
            raise ValueError("depths and num_heads must have 4 elements")

        # Stage mapping matches the existing CNN decoder:
        #   up3: feat3 -> feat2 => D0
        #   up2: D0   -> feat1 => D1
        #   up1: D1   -> feat0 => D2
        #   up0: D2   -> None  => D3
        self.up3 = TransformerUpBlock(
            TransformerUpBlockCfg(
                in_channels=enc_channels[3],
                skip_channels=enc_channels[2],
                out_channels=decoder_channels[0],
                depth=depths[0],
                num_heads=num_heads[0],
                mlp_ratio=mlp_ratio,
                drop=drop,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                use_skip_gate=use_skip_gate,
            )
        )
        self.up2 = TransformerUpBlock(
            TransformerUpBlockCfg(
                in_channels=decoder_channels[0],
                skip_channels=enc_channels[1],
                out_channels=decoder_channels[1],
                depth=depths[1],
                num_heads=num_heads[1],
                mlp_ratio=mlp_ratio,
                drop=drop,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                use_skip_gate=use_skip_gate,
            )
        )
        self.up1 = TransformerUpBlock(
            TransformerUpBlockCfg(
                in_channels=decoder_channels[1],
                skip_channels=enc_channels[0],
                out_channels=decoder_channels[2],
                depth=depths[2],
                num_heads=num_heads[2],
                mlp_ratio=mlp_ratio,
                drop=drop,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                use_skip_gate=use_skip_gate,
            )
        )
        self.up0 = TransformerUpBlock(
            TransformerUpBlockCfg(
                in_channels=decoder_channels[2],
                skip_channels=0,
                out_channels=decoder_channels[3],
                depth=depths[3],
                num_heads=num_heads[3],
                mlp_ratio=mlp_ratio,
                drop=drop,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                use_skip_gate=use_skip_gate,
            )
        )

    def forward(self, feats: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(feats) != 4:
            raise ValueError("Expected feats with 4 feature maps")
        x3 = self.up3(feats[3], feats[2])
        x2 = self.up2(x3, feats[1])
        x1 = self.up1(x2, feats[0])
        x0 = self.up0(x1, None)
        return x0, x1, x2, x3

