import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.models.dwt import DWT, IWT


def compute_psnr(x, y):
    """PSNR for tensors in [0, 1] range (equivalent to 255-scale formula)."""
    mse = F.mse_loss(x, y)
    if mse < 1e-10:
        return torch.tensor(100.0)
    return 10.0 * torch.log10(1.0 / mse)


class SSIMCalculator:
    def __init__(self, window_size=11, channels=3):
        self.window_size = window_size
        self.channels = channels
        self._window = self._create_window(window_size, channels)

    @staticmethod
    def _gaussian(window_size, sigma=1.5):
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        return g / g.sum()

    def _create_window(self, window_size, channels):
        _1d = self._gaussian(window_size)
        _2d = _1d.unsqueeze(1) @ _1d.unsqueeze(0)
        return _2d.unsqueeze(0).unsqueeze(0).expand(channels, 1, -1, -1).contiguous()

    def __call__(self, img1, img2):
        window = self._window.to(img1.device, dtype=img1.dtype)
        c = img1.shape[1]
        if c != self.channels:
            window = self._create_window(self.window_size, c).to(img1.device, dtype=img1.dtype)

        pad = self.window_size // 2
        mu1 = F.conv2d(img1, window, padding=pad, groups=c)
        mu2 = F.conv2d(img2, window, padding=pad, groups=c)
        mu1_sq, mu2_sq = mu1 ** 2, mu2 ** 2
        mu12 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=c) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=c) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=c) - mu12

        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()


class HiNetTrainer:
    """Trainer matching the original HiNet paper exactly.

    Original paper parameters:
        lr = 10^(-4.5) = 3.16e-5
        betas = (0.5, 0.999)
        weight_decay = 1e-5
        lambda_reconstruction = 5, lambda_guide = 1, lambda_low_frequency = 1
        scheduler = StepLR(step_size=1000, gamma=0.5)
        loss = MSE with reduction='sum'
        no noise layer, no AMP, no grad clipping
    """

    def __init__(self, model, device,
                 lr=3.16e-5, betas=(0.5, 0.999), weight_decay=1e-5,
                 lambda_guide=1.0, lambda_reconstruction=5.0,
                 lambda_low_frequency=1.0):
        self.model = model
        self.device = device
        self.dwt = DWT()
        self.iwt = IWT()
        self.ssim_calc = SSIMCalculator()

        self.lambda_guide = lambda_guide
        self.lambda_reconstruction = lambda_reconstruction
        self.lambda_low_frequency = lambda_low_frequency
        self.channels_in = 3

        params = list(filter(lambda p: p.requires_grad, model.parameters()))
        self.optimizer = torch.optim.Adam(
            params, lr=lr, betas=betas, eps=1e-6, weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1000, gamma=0.5,
        )

    def _gauss_noise(self, shape):
        return torch.randn(shape, device=self.device)

    def train_step(self, cover, secret):
        self.model.train()
        cover = cover.to(self.device)
        secret = secret.to(self.device)

        cover_input = self.dwt(cover)
        secret_input = self.dwt(secret)
        input_img = torch.cat((cover_input, secret_input), 1)

        output = self.model(input_img)
        output_steg = output.narrow(1, 0, 4 * self.channels_in)
        output_z = output.narrow(1, 4 * self.channels_in,
                                 output.shape[1] - 4 * self.channels_in)
        steg_img = self.iwt(output_steg)

        output_z_gauss = self._gauss_noise(output_z.shape)
        output_rev = torch.cat((output_steg, output_z_gauss), 1)
        output_image = self.model(output_rev, rev=True)

        secret_rev_wav = output_image.narrow(
            1, 4 * self.channels_in,
            output_image.shape[1] - 4 * self.channels_in,
        )
        secret_rev = self.iwt(secret_rev_wav)

        g_loss = F.mse_loss(steg_img, cover, reduction="sum")
        r_loss = F.mse_loss(secret_rev, secret, reduction="sum")
        steg_low = output_steg.narrow(1, 0, self.channels_in)
        cover_low = cover_input.narrow(1, 0, self.channels_in)
        l_loss = F.mse_loss(steg_low, cover_low, reduction="sum")

        total_loss = (self.lambda_guide * g_loss +
                      self.lambda_reconstruction * r_loss +
                      self.lambda_low_frequency * l_loss)

        total_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return {
            "total_loss": total_loss.item(),
            "g_loss": g_loss.item(),
            "r_loss": r_loss.item(),
            "l_loss": l_loss.item(),
        }

    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        psnr_stego_list = []
        psnr_secret_list = []
        ssim_stego_list = []
        ssim_secret_list = []
        sample = None

        for cover, secret in val_loader:
            cover = cover.to(self.device)
            secret = secret.to(self.device)

            cover_input = self.dwt(cover)
            secret_input = self.dwt(secret)
            input_img = torch.cat((cover_input, secret_input), 1)

            output = self.model(input_img)
            output_steg = output.narrow(1, 0, 4 * self.channels_in)
            output_z = output.narrow(1, 4 * self.channels_in,
                                     output.shape[1] - 4 * self.channels_in)
            steg_img = self.iwt(output_steg)

            output_z_gauss = self._gauss_noise(output_z.shape)
            output_rev = torch.cat((output_steg, output_z_gauss), 1)
            output_image = self.model(output_rev, rev=True)
            secret_rev_wav = output_image.narrow(
                1, 4 * self.channels_in,
                output_image.shape[1] - 4 * self.channels_in,
            )
            secret_rev = self.iwt(secret_rev_wav)

            steg_clamped = steg_img.clamp(0, 1)
            cover_clamped = cover.clamp(0, 1)
            secret_clamped = secret.clamp(0, 1)
            rev_clamped = secret_rev.clamp(0, 1)

            psnr_stego_list.append(compute_psnr(steg_clamped, cover_clamped).item())
            psnr_secret_list.append(compute_psnr(rev_clamped, secret_clamped).item())
            ssim_stego_list.append(self.ssim_calc(steg_clamped, cover_clamped).item())
            ssim_secret_list.append(self.ssim_calc(rev_clamped, secret_clamped).item())

            if sample is None:
                sample = (
                    cover_clamped[0].cpu(),
                    secret_clamped[0].cpu(),
                    steg_clamped[0].cpu(),
                    rev_clamped[0].cpu(),
                )

        return {
            "psnr_stego": np.mean(psnr_stego_list) if psnr_stego_list else 0.0,
            "psnr_secret": np.mean(psnr_secret_list) if psnr_secret_list else 0.0,
            "ssim_stego": np.mean(ssim_stego_list) if ssim_stego_list else 0.0,
            "ssim_secret": np.mean(ssim_secret_list) if ssim_secret_list else 0.0,
        }, sample
