import torch
import torch.nn as nn

from src.models.dense_block import ResidualDenseBlock_out


class INV_block(nn.Module):
    """Invertible block with additive + affine coupling.

    Splits 24-channel wavelet input into two 12-channel halves (cover / secret
    wavelet sub-bands) and applies a reversible transform.
    """

    def __init__(self, subnet_constructor=ResidualDenseBlock_out,
                 clamp=2.0, in_1=3, in_2=3):
        super().__init__()
        self.split_len1 = in_1 * 4  # 12 channels (cover wavelet)
        self.split_len2 = in_2 * 4  # 12 channels (secret wavelet)
        self.clamp = clamp

        self.r = subnet_constructor(self.split_len1, self.split_len2)
        self.y = subnet_constructor(self.split_len1, self.split_len2)
        self.f = subnet_constructor(self.split_len2, self.split_len1)

    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1),
                   x.narrow(1, self.split_len1, self.split_len2))

        if not rev:
            t2 = self.f(x2)
            y1 = x1 + t2
            s1, t1 = self.r(y1), self.y(y1)
            y2 = self.e(s1) * x2 + t1
        else:
            s1, t1 = self.r(x1), self.y(x1)
            y2 = (x2 - t1) / self.e(s1)
            t2 = self.f(y2)
            y1 = x1 - t2

        return torch.cat((y1, y2), 1)
