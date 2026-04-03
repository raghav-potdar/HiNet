#!/usr/bin/env python
"""Post-training evaluation with attack robustness tests.

Usage:
    python evaluate.py --checkpoint checkpoints/hinet_best.pth
    python evaluate.py --checkpoint checkpoints/hinet_best.pth --val_dir path/to/images
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from PIL import Image, ImageFilter
from tqdm import tqdm
import io

from src.core.device import DeviceManager
from src.data.pipeline import DataPipeline
from src.engine.trainer import compute_psnr, SSIMCalculator
from src.models.dwt import DWT, IWT
from src.models.hinet import HiNetModel


class Evaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.dwt = DWT()
        self.iwt = IWT()
        self.ssim_calc = SSIMCalculator()
        self.channels_in = 3

    def _gauss_noise(self, shape):
        return torch.randn(shape, device=self.device)

    @torch.no_grad()
    def hide_and_reveal(self, cover, secret):
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

        return steg_img.clamp(0, 1), secret_rev.clamp(0, 1)

    @torch.no_grad()
    def reveal_from_stego(self, stego):
        """Reveal secret from a (potentially attacked) stego image."""
        stego = stego.to(self.device)
        steg_wav = self.dwt(stego)
        z_shape = (stego.shape[0], 4 * self.channels_in,
                   stego.shape[2] // 2, stego.shape[3] // 2)
        output_z_gauss = self._gauss_noise(z_shape)
        output_rev = torch.cat((steg_wav, output_z_gauss), 1)
        output_image = self.model(output_rev, rev=True)
        secret_rev_wav = output_image.narrow(
            1, 4 * self.channels_in,
            output_image.shape[1] - 4 * self.channels_in,
        )
        return self.iwt(secret_rev_wav).clamp(0, 1)

    def compute_metrics(self, pred, target):
        return {
            "psnr": compute_psnr(pred, target).item(),
            "ssim": self.ssim_calc(pred, target).item(),
        }


def apply_jpeg_attack(stego_tensor, quality):
    """Apply real JPEG compression via PIL."""
    results = []
    for i in range(stego_tensor.shape[0]):
        img = stego_tensor[i].cpu().clamp(0, 1)
        img_pil = Image.fromarray(
            (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        )
        buf = io.BytesIO()
        img_pil.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        img_jpeg = Image.open(buf).convert("RGB")
        img_t = torch.from_numpy(
            np.array(img_jpeg).astype(np.float32) / 255.0
        ).permute(2, 0, 1)
        results.append(img_t)
    return torch.stack(results).to(stego_tensor.device)


def apply_blur_attack(stego_tensor, radius=2):
    results = []
    for i in range(stego_tensor.shape[0]):
        img = stego_tensor[i].cpu().clamp(0, 1)
        img_pil = Image.fromarray(
            (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        )
        img_blur = img_pil.filter(ImageFilter.GaussianBlur(radius=radius))
        img_t = torch.from_numpy(
            np.array(img_blur).astype(np.float32) / 255.0
        ).permute(2, 0, 1)
        results.append(img_t)
    return torch.stack(results).to(stego_tensor.device)


def apply_noise_attack(stego_tensor, std=0.03):
    noise = torch.randn_like(stego_tensor) * std
    return (stego_tensor + noise).clamp(0, 1)


def apply_resize_attack(stego_tensor, scale=0.5):
    _, _, h, w = stego_tensor.shape
    new_h, new_w = int(h * scale), int(w * scale)
    down = F.interpolate(stego_tensor, size=(new_h, new_w),
                         mode="bilinear", align_corners=False)
    up = F.interpolate(down, size=(h, w),
                       mode="bilinear", align_corners=False)
    return up


ATTACKS = {
    "clean": lambda x: x,
    "jpeg_90": lambda x: apply_jpeg_attack(x, 90),
    "jpeg_50": lambda x: apply_jpeg_attack(x, 50),
    "blur": lambda x: apply_blur_attack(x, radius=2),
    "noise": lambda x: apply_noise_attack(x, std=0.03),
    "resize_50": lambda x: apply_resize_attack(x, scale=0.5),
    "resize_75": lambda x: apply_resize_attack(x, scale=0.75),
    "social": lambda x: apply_jpeg_attack(
        apply_resize_attack(x, scale=0.75), quality=70
    ),
}


def main():
    parser = argparse.ArgumentParser(description="HiNet Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--val_dir", type=str,
                        default="datasets/DIV2K_valid_HR")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    dm = DeviceManager()
    device = dm.device

    model = HiNetModel().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["net"])
    model.eval()
    print(f"[Eval] Loaded checkpoint: {args.checkpoint}")

    pipeline = DataPipeline(crop_size=224, val_crop_size=224)
    val_loader = pipeline.get_val_loader(args.val_dir, batch_size=args.batch_size)

    evaluator = Evaluator(model, device)

    os.makedirs("results/evaluation", exist_ok=True)

    report_lines = []
    report_lines.append("HiNet Evaluation Report")
    report_lines.append("=" * 50)

    for attack_name, attack_fn in ATTACKS.items():
        psnr_list, ssim_list = [], []
        sample_saved = False

        for cover, secret in tqdm(val_loader, desc=attack_name):
            cover = cover.to(device)
            secret = secret.to(device)

            stego, _ = evaluator.hide_and_reveal(cover, secret)
            attacked_stego = attack_fn(stego)
            revealed = evaluator.reveal_from_stego(attacked_stego)

            secret_clamped = secret.clamp(0, 1)
            m = evaluator.compute_metrics(revealed, secret_clamped)
            psnr_list.append(m["psnr"])
            ssim_list.append(m["ssim"])

            if not sample_saved:
                strip = torch.cat([
                    cover[0].cpu().clamp(0, 1),
                    secret[0].cpu().clamp(0, 1),
                    stego[0].cpu().clamp(0, 1),
                    attacked_stego[0].cpu().clamp(0, 1),
                    revealed[0].cpu().clamp(0, 1),
                ], dim=2)
                vutils.save_image(strip, f"results/evaluation/{attack_name}.png")
                sample_saved = True

        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)

        if avg_psnr > 28:
            status = "PASS"
        elif avg_psnr > 20:
            status = "WARN"
        else:
            status = "FAIL"

        line = (f"  {attack_name:12s} | PSNR={avg_psnr:6.2f}dB | "
                f"SSIM={avg_ssim:.4f} | [{status}]")
        print(line)
        report_lines.append(line)

    report_lines.append("=" * 50)
    report_path = "results/evaluation/report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
