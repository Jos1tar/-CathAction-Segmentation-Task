# coding: utf-8
import torch.nn as nn
import torch.nn.functional as F

from .swinunet_parts import UpBlock
from .transformer_decoder import TransformerDecoder


class SwinUNet(nn.Module):
    """
    SwinUNet style model using a Swin Transformer encoder from timm and a U-Net decoder.
    Default model expects inputs divisible by patch_size * window_size.
    """

    DEFAULT_MODEL = 'swinv2_tiny_window8_256'

    def __init__(
        self,
        n_channels=3,
        n_classes=3,
        model_name=None,
        pretrained=False,
        decoder_channels=None,
        img_size=None,
        use_skip_gate: bool = False,
        use_deep_supervision: bool = False,
        decoder_type: str = 'cnn',
        # Transformer decoder knobs (kept optional so old experiments still run).
        decoder_depths: tuple[int, int, int, int] = (1, 1, 1, 1),
        decoder_num_heads: tuple[int, int, int, int] = (8, 8, 4, 4),
        decoder_mlp_ratio: float = 4.0,
        decoder_drop: float = 0.0,
        decoder_attn_drop: float = 0.0,
        decoder_proj_drop: float = 0.0,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.use_skip_gate = bool(use_skip_gate)
        self.use_deep_supervision = bool(use_deep_supervision)
        self.decoder_type = (decoder_type or 'cnn').strip().lower()

        try:
            import timm
        except Exception as exc:
            raise ImportError(
                'SwinUNet requires timm. Install with: pip install timm'
            ) from exc

        model_name = model_name or self.DEFAULT_MODEL
        create_kwargs = dict(
            pretrained=pretrained,
            features_only=True,
            in_chans=n_channels,
            out_indices=(0, 1, 2, 3),
        )
        if img_size is not None:
            create_kwargs['img_size'] = img_size
        try:
            self.backbone = timm.create_model(model_name, **create_kwargs)
        except TypeError:
            # Some timm models may not accept img_size; retry without it.
            create_kwargs.pop('img_size', None)
            self.backbone = timm.create_model(model_name, **create_kwargs)

        enc_channels = list(self.backbone.feature_info.channels())
        if len(enc_channels) != 4:
            raise ValueError(f'Expected 4 encoder stages, got {len(enc_channels)}')

        if decoder_channels is None:
            decoder_channels = (256, 128, 64, 32)
        if len(decoder_channels) != 4:
            raise ValueError('decoder_channels must have 4 integers')

        # Decoder blocks.
        # - "cnn": classic U-Net style upsample + concat + conv.
        # - "transformer": upsample + (optional) gated skip + token-mixer blocks.
        if self.decoder_type not in {'cnn', 'transformer'}:
            raise ValueError("decoder_type must be one of: 'cnn', 'transformer'")

        if self.decoder_type == 'cnn':
            # If `use_skip_gate=True`, each UpBlock will run a small gate (SE-style)
            # on the skip feature BEFORE concatenating it with the upsampled feature.
            self.up3 = UpBlock(enc_channels[3], enc_channels[2], decoder_channels[0], use_skip_gate=use_skip_gate)
            self.up2 = UpBlock(decoder_channels[0], enc_channels[1], decoder_channels[1], use_skip_gate=use_skip_gate)
            self.up1 = UpBlock(decoder_channels[1], enc_channels[0], decoder_channels[2], use_skip_gate=use_skip_gate)
            self.up0 = UpBlock(decoder_channels[2], 0,               decoder_channels[3], use_skip_gate=use_skip_gate)
            self.decoder = None
        else:
            # Transformer decoder is packed into a single module so SwinUNet stays tidy.
            self.decoder = TransformerDecoder(
                enc_channels=enc_channels,
                decoder_channels=tuple(decoder_channels),
                depths=tuple(decoder_depths),
                num_heads=tuple(decoder_num_heads),
                mlp_ratio=float(decoder_mlp_ratio),
                drop=float(decoder_drop),
                attn_drop=float(decoder_attn_drop),
                proj_drop=float(decoder_proj_drop),
                use_skip_gate=use_skip_gate,
            )
            self.up3 = self.up2 = self.up1 = self.up0 = None

        self.out_conv = nn.Conv2d(decoder_channels[3], n_classes, kernel_size=1)

        if self.use_deep_supervision:
            # Deep supervision = extra prediction heads on intermediate decoder features.
            # During training we add auxiliary losses to encourage early decoder stages
            # to be semantically meaningful (often helps thin structures).
            self.aux3 = nn.Conv2d(decoder_channels[0], n_classes, kernel_size=1)
            self.aux2 = nn.Conv2d(decoder_channels[1], n_classes, kernel_size=1)
            self.aux1 = nn.Conv2d(decoder_channels[2], n_classes, kernel_size=1)

        self.patch_size = self._infer_patch_size()
        self.window_size = self._infer_window_size()
        self.input_multiple = self.patch_size * self.window_size

    def _infer_patch_size(self):
        patch_size = getattr(self.backbone, 'patch_size', 4)
        if isinstance(patch_size, (tuple, list)):
            patch_size = patch_size[0]
        return int(patch_size)

    def _infer_window_size(self):
        window_size = getattr(self.backbone, 'window_size', 8)
        if isinstance(window_size, (tuple, list)):
            window_size = window_size[0]
        return int(window_size)

    def _pad_input(self, x):
        # Swin backbones typically want H/W divisible by (patch_size * window_size).
        # We pad once at the input and crop back at the end.
        h, w = x.shape[-2:]
        m = self.input_multiple
        pad_h = (m - h % m) % m
        pad_w = (m - w % m) % m
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return x, (h, w)

    def _normalize_feature(self, feat, expected_channels):
        # Some timm Swin models emit NHWC; convert to NCHW if needed.
        # We keep the rest of the pipeline in PyTorch standard NCHW.
        if feat.shape[1] == expected_channels:
            return feat
        if feat.shape[-1] == expected_channels:
            return feat.permute(0, 3, 1, 2).contiguous()
        raise ValueError(
            f'Unexpected feature shape {tuple(feat.shape)} for channels={expected_channels}'
        )

    def forward(self, x):
        x, (h, w) = self._pad_input(x)
        feats = self.backbone(x)
        feats = [self._normalize_feature(f, c) for f, c in zip(feats, self.backbone.feature_info.channels())]

        if self.decoder_type == 'cnn':
            x3 = self.up3(feats[3], feats[2])
            x2 = self.up2(x3,       feats[1])
            x1 = self.up1(x2,       feats[0])
            x0 = self.up0(x1,       None)
        else:
            # Transformer decoder returns (x0, x1, x2, x3) to match deep supervision heads.
            x0, x1, x2, x3 = self.decoder(feats)
        out = self.out_conv(x0)

        if out.shape[-2:] != (h, w):
            out = out[..., :h, :w]

        if self.use_deep_supervision:
            # When deep supervision is enabled we return a tuple:
            #   (main_logits, [aux_logits_stage1, aux_logits_stage2, aux_logits_stage3])
            # The training script is responsible for combining the losses.
            aux3 = self.aux3(x3)
            aux2 = self.aux2(x2)
            aux1 = self.aux1(x1)
            aux_outs = [aux1, aux2, aux3]
            for i in range(len(aux_outs)):
                if aux_outs[i].shape[-2:] != (h, w):
                    aux_outs[i] = F.interpolate(aux_outs[i], size=(h, w), mode='bilinear', align_corners=False)
            return out, aux_outs

        return out
