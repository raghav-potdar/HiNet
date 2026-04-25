"""Decoupled reveal network: CBAM-UNet with a ResidualDense bottleneck.

Takes a (possibly noised) stego image in [0, 1] and outputs the recovered
secret image in [0, 1]. Non-invertible by design so it can specialize on
noise robustness without the constraints of HiNet's reverse pass.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.cbam import CBAM
from src.models.dense_block import ResidualDenseBlock_out


class ConvBlock(nn.Module):
    """Two 3x3 conv + BN + LeakyReLU, followed by CBAM."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.attn = CBAM(out_ch)

    def forward(self, x):
        return self.attn(self.body(x))


class RevealNet(nn.Module):
    """3-stage CBAM-UNet with ResidualDense bottleneck.

    Channels: [base_ch, base_ch*2, base_ch*4] (default 64 / 128 / 256).
    Input:  (B, in_ch, H, W) in [0, 1].
    Output: (B, out_ch, H, W) in [0, 1] (sigmoid head).
    """

    def __init__(self, in_ch=3, out_ch=3, base_ch=64):
        super().__init__()
        c1, c2, c3 = base_ch, base_ch * 2, base_ch * 4

        self.enc1 = ConvBlock(in_ch, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            ResidualDenseBlock_out(c3, c3),
            CBAM(c3),
        )

        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear",
                               align_corners=False)
        self.dec3 = ConvBlock(c3 + c3, c2)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear",
                               align_corners=False)
        self.dec2 = ConvBlock(c2 + c2, c1)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear",
                               align_corners=False)
        self.dec1 = ConvBlock(c1 + c1, c1 // 2)

        self.head = nn.Conv2d(c1 // 2, out_ch, kernel_size=1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))

        b = self.bottleneck(self.pool(s3))

        d3 = self.dec3(torch.cat([self.up3(b), s3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), s2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), s1], dim=1))

        return torch.sigmoid(self.head(d1))
