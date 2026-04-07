"""Differentiable noise augmentation for training robustness.

Applied in pixel domain ([0, 1] range) on stego images between the
forward (hiding) and reverse (revealing) passes so the model learns
to survive common image processing operations (JPEG, blur, resize, etc.).

JPEG simulation uses the Diff-JPEG library (WACV 2024):
    https://github.com/necla-ml/Diff-JPEG
"""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from diff_jpeg import DiffJPEGCoding


class DiffJPEG(nn.Module):
    """Differentiable JPEG via Diff-JPEG (Reich et al., WACV 2024).

    Wraps DiffJPEGCoding with [0,1] <-> [0,255] scaling and
    randomized quality per forward call.
    """

    def __init__(self, quality_range=(50, 95)):
        super().__init__()
        self.quality_range = quality_range
        self.jpeg = DiffJPEGCoding(ste=True)

    def forward(self, x):
        if not self.training:
            return x
        b = x.shape[0]
        q = torch.empty(b, device=x.device).uniform_(*self.quality_range)
        out = self.jpeg(x * 255.0, q)
        return out / 255.0


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

    def __init__(self, jpeg_quality_range=(50, 95)):
        super().__init__()
        self.jpeg = DiffJPEG(quality_range=jpeg_quality_range)
        self.blur = GaussianBlur(kernel_size=5, sigma=1.0)
        self.noise = GaussianNoise(std=0.03)
        self.dropout = PixelDropout(p=0.05)
        self.resize = RandomResizing(scale_range=(0.5, 0.9))

        self._augmentations = [
            "identity",
            "jpeg",
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
        elif aug == "jpeg":
            return self.jpeg(x)
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
            out = self.jpeg(out)
            out = self.noise(out)
            return out
        return x
