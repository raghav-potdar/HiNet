"""Differentiable noise augmentation layer adapted from Nexus-Steg.

Applied in pixel domain on stego images to make hidden information
robust against common image processing operations.
"""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffJPEG(nn.Module):
    """Differentiable JPEG approximation using 8x8 DCT blocks with STE."""

    def __init__(self, quality=90):
        super().__init__()
        self.quality = quality
        scale = 5000.0 / quality if quality < 50 else 200.0 - 2.0 * quality
        self.register_buffer(
            "quant_table",
            self._jpeg_quant_table() * (scale / 100.0),
        )

    @staticmethod
    def _jpeg_quant_table():
        return torch.tensor([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99],
        ], dtype=torch.float32)

    def forward(self, x):
        if not self.training:
            return x
        img_01 = (x + 1.0) / 2.0
        b, c, h, w = img_01.shape
        ph = (8 - h % 8) % 8
        pw = (8 - w % 8) % 8
        if ph > 0 or pw > 0:
            img_01 = F.pad(img_01, (0, pw, 0, ph), mode="reflect")
        _, _, h2, w2 = img_01.shape
        blocks = img_01.unfold(2, 8, 8).unfold(3, 8, 8)
        blocks = blocks.contiguous().view(b, c, -1, 8, 8)

        quant = self.quant_table.to(x.device).clamp(min=1.0)
        quantized = blocks / quant.view(1, 1, 1, 8, 8)
        quantized = quantized + (quantized.round() - quantized).detach()  # STE
        blocks_out = quantized * quant.view(1, 1, 1, 8, 8)

        nh, nw = h2 // 8, w2 // 8
        out = blocks_out.view(b, c, nh, nw, 8, 8)
        out = out.permute(0, 1, 2, 4, 3, 5).contiguous().view(b, c, h2, w2)
        if ph > 0 or pw > 0:
            out = out[:, :, :h, :w]
        return out * 2.0 - 1.0


class GaussianBlur(nn.Module):
    def __init__(self, kernel_size=5, sigma=1.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        ax = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * self.sigma ** 2))
        kernel = kernel / kernel.sum()
        self.register_buffer("kernel", kernel.view(1, 1, kernel_size, kernel_size))

    def forward(self, x):
        if not self.training:
            return x
        pad = self.kernel_size // 2
        k = self.kernel.expand(x.shape[1], -1, -1, -1).to(x.device)
        return F.conv2d(x, k, padding=pad, groups=x.shape[1])


class GaussianNoise(nn.Module):
    def __init__(self, std=0.03):
        super().__init__()
        self.std = std

    def forward(self, x):
        if not self.training:
            return x
        return x + torch.randn_like(x) * self.std


class PixelDropout(nn.Module):
    def __init__(self, p=0.05):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training:
            return x
        mask = (torch.rand(x.shape[0], 1, x.shape[2], x.shape[3],
                           device=x.device) > self.p).float()
        return x * mask


class RandomResizing(nn.Module):
    def __init__(self, scale_range=(0.5, 0.9)):
        super().__init__()
        self.scale_range = scale_range

    def forward(self, x):
        if not self.training:
            return x
        _, _, h, w = x.shape
        scale = random.uniform(*self.scale_range)
        new_h, new_w = int(h * scale), int(w * scale)
        down = F.interpolate(x, size=(new_h, new_w), mode="bilinear",
                             align_corners=False)
        return F.interpolate(down, size=(h, w), mode="bilinear",
                             align_corners=False)


class DifferentiableNoiseLayer(nn.Module):
    """Randomly applies one of several differentiable augmentations during training."""

    def __init__(self):
        super().__init__()
        self.jpeg90 = DiffJPEG(quality=90)
        self.jpeg50 = DiffJPEG(quality=50)
        self.blur = GaussianBlur(kernel_size=5, sigma=1.0)
        self.noise = GaussianNoise(std=0.03)
        self.dropout = PixelDropout(p=0.05)
        self.resize = RandomResizing(scale_range=(0.5, 0.9))

        self._augmentations = [
            "identity",
            "jpeg90",
            "jpeg50",
            "blur",
            "noise",
            "dropout",
            "resize",
            "combined",
        ]

    def forward(self, x):
        if not self.training:
            return x

        aug = random.choice(self._augmentations)
        if aug == "identity":
            return x
        elif aug == "jpeg90":
            return self.jpeg90(x)
        elif aug == "jpeg50":
            return self.jpeg50(x)
        elif aug == "blur":
            return self.blur(x)
        elif aug == "noise":
            return self.noise(x)
        elif aug == "dropout":
            return self.dropout(x)
        elif aug == "resize":
            return self.resize(x)
        elif aug == "combined":
            out = self.resize(x)
            out = self.jpeg90(out)
            out = self.noise(out)
            return out
        return x
