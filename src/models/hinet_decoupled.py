"""Decoupled HiNet: invertible hide + separate non-invertible reveal.

Keeps the original HiNet stack for hiding (preserves its imperceptibility)
and pairs it with a CBAM-UNet RevealNet that operates directly on the
pixel-domain stego image. Designed for noise-robust training where the
reveal path needs flexibility the invertible constraint cannot provide.
"""

import torch.nn as nn

from src.models.hinet import Hinet
from src.models.reveal_net import RevealNet


class HiNetDecoupled(nn.Module):
    def __init__(self, reveal_base_ch=64):
        super().__init__()
        self.hide = Hinet()
        self.reveal = RevealNet(in_ch=3, out_ch=3, base_ch=reveal_base_ch)

    def hide_forward(self, dwt_input):
        """Run the invertible hide stack. Returns concatenated
        [output_steg (4*C ch) | output_z (rest)] in wavelet domain."""
        return self.hide(dwt_input)

    def reveal_forward(self, steg_pixel):
        """Recover the secret directly from a pixel-domain stego image."""
        return self.reveal(steg_pixel)
