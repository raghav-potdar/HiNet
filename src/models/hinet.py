import torch
import torch.nn as nn

from src.models.inv_block import INV_block


class Hinet(nn.Module):
    """Stack of 16 invertible blocks."""

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([INV_block() for _ in range(16)])

    def forward(self, x, rev=False):
        if not rev:
            for block in self.blocks:
                x = block(x)
        else:
            for block in reversed(self.blocks):
                x = block(x, rev=True)
        return x


class HiNetModel(nn.Module):
    """Top-level wrapper around Hinet."""

    def __init__(self):
        super().__init__()
        self.model = Hinet()

    def forward(self, x, rev=False):
        return self.model(x, rev=rev)


def init_model(model, init_scale=0.01):
    """Random init (scale 0.01), zero out conv5 layers for stable start."""
    for key, param in model.named_parameters():
        if param.requires_grad:
            param.data = init_scale * torch.randn(param.data.shape, device=param.device)
            if key.split(".")[-2] == "conv5":
                param.data.fill_(0.0)
