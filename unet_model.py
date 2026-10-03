""" Full assembly of the parts to form the complete network """

from .unet_parts import *
import torch.nn as nn
import torch

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, base_channels=64):
        """
        Build a plain U-Net.

        Args:
            n_channels: input channels (e.g. 3 for RGB)
            n_classes: number of output classes
            bilinear: use bilinear upsampling instead of transposed conv
            base_channels: base width of the network (capacity knob). Typical values:
                - 32: small
                - 64: standard
                - 128+: large
            Channel depth doubles as we go down the encoder.
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.base_channels = base_channels

        # Compute channels per stage from the base width.
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        self.inc = (DoubleConv(n_channels, c1))
        self.down1 = (Down(c1, c2))
        self.down2 = (Down(c2, c3))
        self.down3 = (Down(c3, c4))
        factor = 2 if bilinear else 1
        self.down4 = (Down(c4, c5 // factor))

        self.up1 = (Up(c5, c4 // factor, bilinear))
        self.up2 = (Up(c4, c3 // factor, bilinear))
        self.up3 = (Up(c3, c2 // factor, bilinear))
        self.up4 = (Up(c2, c1, bilinear))
        self.outc = (OutConv(c1, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)