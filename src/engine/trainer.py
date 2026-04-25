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
    """HiNet trainer with optional gradient safety.

    Original paper parameters:
        lr = 10^(-4.5) = 3.16e-5
        betas = (0.5, 0.999)
        weight_decay = 1e-5
        lambda_reconstruction = 5, lambda_guide = 1, lambda_low_frequency = 1
        scheduler = StepLR(step_size=1000, gamma=0.5)
        loss = MSE with reduction='sum'

    Gradient safety (enabled by default, disable with max_grad_norm=None):
        Per-step gradient clipping prevents catastrophic single-batch spikes.
    """

    def __init__(self, model, device,
                 lr=3.16e-5, betas=(0.5, 0.999), weight_decay=1e-5,
                 lambda_guide=1.0, lambda_reconstruction=5.0,
                 lambda_low_frequency=1.0,
                 max_grad_norm=10.0,
                 noise_layer=None):
        self.model = model
        self.device = device
        self.dwt = DWT()
        self.iwt = IWT()
        self.ssim_calc = SSIMCalculator()
        self.max_grad_norm = max_grad_norm
        self.noise_layer = noise_layer

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

        if self.noise_layer is not None:
            steg_noised = self.noise_layer(steg_img)
            output_steg_noised = self.dwt(steg_noised)
        else:
            output_steg_noised = output_steg

        output_z_gauss = self._gauss_noise(output_z.shape)
        output_rev = torch.cat((output_steg_noised, output_z_gauss), 1)
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

        raw_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=float("inf"),
        ).item()

        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.max_grad_norm,
            )

        self.optimizer.step()
        self.optimizer.zero_grad()

        return {
            "total_loss": total_loss.item(),
            "g_loss": g_loss.item(),
            "r_loss": r_loss.item(),
            "l_loss": l_loss.item(),
            "raw_grad_norm": raw_grad_norm,
            "grad_norm": min(raw_grad_norm, self.max_grad_norm)
                         if self.max_grad_norm is not None else raw_grad_norm,
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


class DecoupledTrainer:
    """Trainer for HiNetDecoupled (invertible hide + CBAM-UNet reveal).

    Two independent Adam optimizers so the hide net can be fine-tuned at a
    low LR while the reveal net trains from scratch at a higher LR. The
    reveal path operates on the (possibly noised) pixel-domain stego image
    directly, bypassing HiNet's reverse pass.
    """

    def __init__(self, model, device,
                 hide_lr=1e-6, reveal_lr=1e-4,
                 betas=(0.5, 0.999), weight_decay=1e-5,
                 lambda_guide=1.0, lambda_reconstruction=5.0,
                 lambda_low_frequency=1.0,
                 max_grad_norm=10.0,
                 noise_layer=None,
                 freeze_hide_epochs=0,
                 step_size=1000, gamma=0.5):
        self.model = model
        self.device = device
        self.dwt = DWT()
        self.iwt = IWT()
        self.ssim_calc = SSIMCalculator()
        self.max_grad_norm = max_grad_norm
        self.noise_layer = noise_layer
        self.freeze_hide_epochs = freeze_hide_epochs
        self.current_epoch = 0
        self.hide_frozen = False

        self.lambda_guide = lambda_guide
        self.lambda_reconstruction = lambda_reconstruction
        self.lambda_low_frequency = lambda_low_frequency
        self.channels_in = 3

        hide_params = list(filter(lambda p: p.requires_grad,
                                  self.model.hide.parameters()))
        reveal_params = list(filter(lambda p: p.requires_grad,
                                    self.model.reveal.parameters()))

        self.opt_hide = torch.optim.Adam(
            hide_params, lr=hide_lr, betas=betas, eps=1e-6,
            weight_decay=weight_decay,
        )
        self.opt_reveal = torch.optim.Adam(
            reveal_params, lr=reveal_lr, betas=betas, eps=1e-6,
            weight_decay=weight_decay,
        )

        self.sched_hide = torch.optim.lr_scheduler.StepLR(
            self.opt_hide, step_size=step_size, gamma=gamma,
        )
        self.sched_reveal = torch.optim.lr_scheduler.StepLR(
            self.opt_reveal, step_size=step_size, gamma=gamma,
        )

        if freeze_hide_epochs > 0:
            self._freeze_hide(True)

        # Compatibility shims so outer code that inspects .optimizer /
        # .scheduler still works (watchdog logic in main.py).
        self.optimizer = self.opt_reveal
        self.scheduler = self.sched_reveal

    def _freeze_hide(self, freeze):
        for p in self.model.hide.parameters():
            p.requires_grad = not freeze
        self.hide_frozen = freeze

    def set_epoch(self, epoch):
        """Called at the start of each training epoch. Unfreezes the hide
        net once freeze_hide_epochs have elapsed."""
        self.current_epoch = epoch
        if self.hide_frozen and epoch >= self.freeze_hide_epochs:
            self._freeze_hide(False)
            print(f"[DecoupledTrainer] Unfreezing hide at epoch {epoch}")

    def train_step(self, cover, secret):
        self.model.train()
        cover = cover.to(self.device)
        secret = secret.to(self.device)

        cover_input = self.dwt(cover)
        secret_input = self.dwt(secret)
        input_img = torch.cat((cover_input, secret_input), 1)

        output = self.model.hide_forward(input_img)
        output_steg = output.narrow(1, 0, 4 * self.channels_in)
        steg_img = self.iwt(output_steg)

        if self.noise_layer is not None:
            steg_noised = self.noise_layer(steg_img)
        else:
            steg_noised = steg_img

        steg_noised_clamped = steg_noised.clamp(0.0, 1.0)
        secret_rev = self.model.reveal_forward(steg_noised_clamped)

        g_loss = F.mse_loss(steg_img, cover, reduction="sum")
        r_loss = F.mse_loss(secret_rev, secret, reduction="sum")
        steg_low = output_steg.narrow(1, 0, self.channels_in)
        cover_low = cover_input.narrow(1, 0, self.channels_in)
        l_loss = F.mse_loss(steg_low, cover_low, reduction="sum")

        total_loss = (self.lambda_guide * g_loss +
                      self.lambda_reconstruction * r_loss +
                      self.lambda_low_frequency * l_loss)

        self.opt_hide.zero_grad()
        self.opt_reveal.zero_grad()
        total_loss.backward()

        raw_grad_norm_reveal = torch.nn.utils.clip_grad_norm_(
            self.model.reveal.parameters(), max_norm=float("inf"),
        ).item()
        if not self.hide_frozen:
            raw_grad_norm_hide = torch.nn.utils.clip_grad_norm_(
                self.model.hide.parameters(), max_norm=float("inf"),
            ).item()
        else:
            raw_grad_norm_hide = 0.0

        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.reveal.parameters(), max_norm=self.max_grad_norm,
            )
            if not self.hide_frozen:
                torch.nn.utils.clip_grad_norm_(
                    self.model.hide.parameters(), max_norm=self.max_grad_norm,
                )

        if not self.hide_frozen:
            self.opt_hide.step()
        self.opt_reveal.step()

        raw_grad_norm = max(raw_grad_norm_hide, raw_grad_norm_reveal)
        grad_norm = (min(raw_grad_norm, self.max_grad_norm)
                     if self.max_grad_norm is not None else raw_grad_norm)

        return {
            "total_loss": total_loss.item(),
            "g_loss": g_loss.item(),
            "r_loss": r_loss.item(),
            "l_loss": l_loss.item(),
            "raw_grad_norm": raw_grad_norm,
            "grad_norm": grad_norm,
            "raw_grad_norm_hide": raw_grad_norm_hide,
            "raw_grad_norm_reveal": raw_grad_norm_reveal,
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

            output = self.model.hide_forward(input_img)
            output_steg = output.narrow(1, 0, 4 * self.channels_in)
            steg_img = self.iwt(output_steg)
            steg_clamped = steg_img.clamp(0, 1)

            secret_rev = self.model.reveal_forward(steg_clamped)

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
