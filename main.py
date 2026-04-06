#!/usr/bin/env python
"""HiNet training entry point — supports multi-stage training.

Default parameters (original HiNet paper):
    lr = 10^(-4.5) = 3.16e-5, betas = (0.5, 0.999), weight_decay = 1e-5
    lambda_reconstruction = 5, lambda_guide = 1, lambda_low_frequency = 1
    batch_size = 16, crop_size = 224, val_crop = 1024
    scheduler = StepLR(step=1000, gamma=0.5)

Multi-stage workflow (from HiNet README training demo):
    # Stage 1: lr = 10^(-4.5), train 500 epochs
    python main.py --epochs 500 --lr 3.16e-5

    # Stage 2: resume, lr = 10^(-5.0)
    python main.py --epochs 1190 --lr 1e-5 \
        --resume checkpoints/hinet_epoch_500.pth --start_epoch 500

    # Stage 3: resume, lr = 10^(-5.2)
    python main.py --epochs 500 --lr 6.31e-6 \
        --resume checkpoints/hinet_epoch_1690.pth --start_epoch 1690
"""

import argparse
import csv
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from tqdm import tqdm

from src.core.device import DeviceManager
from src.data.pipeline import DataPipeline
from src.engine.trainer import HiNetTrainer
from src.models.hinet import HiNetModel, init_model


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_image_grid(tensors, path, nrow=4):
    grid = vutils.make_grid(torch.stack(tensors), nrow=nrow, padding=2)
    vutils.save_image(grid, path)


class HiNetApp:
    def __init__(self, args):
        self.args = args
        set_seed(args.seed)

        self.dm = DeviceManager()
        self.device = self.dm.device

        if args.batch_size is None:
            args.batch_size = 16 if self.dm.is_cuda else 4
        print(f"[Config] batch_size={args.batch_size}, epochs={args.epochs}, "
              f"lr={args.lr:.2e}, seed={args.seed}, val_freq={args.val_freq}, "
              f"start_epoch={args.start_epoch}")
        if args.resume:
            print(f"[Config] Resuming from {args.resume}, start_epoch={args.start_epoch}")

        self.pipeline = DataPipeline(
            crop_size=args.crop_size,
            val_crop_size=args.val_crop_size,
        )
        self.train_loader, self.val_loader = self.pipeline.get_loaders(
            args.train_dir, args.val_dir,
            batch_size=args.batch_size,
            val_batch_size=args.val_batch_size,
            num_workers=self.dm.get_optimal_workers(),
        )

        self.model = HiNetModel().to(self.device)
        init_model(self.model)
        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"[Model] Parameters: {param_count:,}")

        mgn = None if args.no_grad_safety else args.max_grad_norm
        self.trainer = HiNetTrainer(self.model, self.device, lr=args.lr,
                                    max_grad_norm=mgn)
        if mgn is not None:
            print(f"[GradSafety] clip={mgn}, watchdog_factor={args.grad_watch_factor}")
        else:
            print("[GradSafety] Disabled (--no_grad_safety)")

        if args.resume:
            ckpt = torch.load(args.resume, map_location=self.device)
            self.model.load_state_dict(ckpt["net"])
            self.trainer.optimizer.load_state_dict(ckpt["opt"])
            for pg in self.trainer.optimizer.param_groups:
                pg["lr"] = args.lr
            print(f"[Resume] Loaded checkpoint (epoch {ckpt.get('epoch', '?')}), "
                  f"LR set to {args.lr:.2e}")

        os.makedirs("results", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)

    def run_sanity(self):
        """Sanity check: verify initial losses are in expected range."""
        print("\n" + "=" * 60)
        print("SANITY CHECK")
        print("=" * 60)

        cover, secret = next(iter(self.train_loader))
        cover, secret = cover.to(self.device), secret.to(self.device)

        imgs = []
        n_show = min(4, cover.shape[0])
        for i in range(n_show):
            imgs.append(cover[i].cpu().clamp(0, 1))
        for i in range(n_show):
            imgs.append(secret[i].cpu().clamp(0, 1))
        save_image_grid(imgs, "results/sanity_inputs.png", nrow=n_show)
        print("[Sanity] Saved results/sanity_inputs.png")

        print(f"  Cover range:  [{cover.min():.3f}, {cover.max():.3f}]")
        print(f"  Secret range: [{secret.min():.3f}, {secret.max():.3f}]")

        self.model.eval()
        with torch.no_grad():
            from src.models.dwt import DWT, IWT
            dwt, iwt = DWT(), IWT()
            cover_input = dwt(cover)
            secret_input = dwt(secret)
            input_img = torch.cat((cover_input, secret_input), 1)

            output = self.model(input_img)
            output_steg = output.narrow(1, 0, 12)
            output_z = output.narrow(1, 12, output.shape[1] - 12)
            steg_img = iwt(output_steg)

            output_z_gauss = torch.randn_like(output_z)
            output_rev = torch.cat((output_steg, output_z_gauss), 1)
            output_image = self.model(output_rev, rev=True)
            secret_rev = iwt(output_image.narrow(1, 12, output_image.shape[1] - 12))

            g_loss = F.mse_loss(steg_img, cover, reduction="sum").item()
            r_loss = F.mse_loss(secret_rev, secret, reduction="sum").item()
            steg_low = output_steg.narrow(1, 0, 3)
            cover_low = cover_input.narrow(1, 0, 3)
            l_loss = F.mse_loss(steg_low, cover_low, reduction="sum").item()
            total_loss = g_loss + 5.0 * r_loss + l_loss

        print(f"\n  g_loss (guide):          {g_loss:.4f}")
        print(f"  r_loss (reconstruction): {r_loss:.4f}")
        print(f"  l_loss (low frequency):  {l_loss:.4f}")
        print(f"  total_loss:              {total_loss:.4f}")

        checks_passed = True
        if g_loss > 1000:
            print("  [WARN] g_loss very high — cover/stego mismatch at init")
            checks_passed = False
        if r_loss < 0.001:
            print("  [WARN] r_loss suspiciously low — network may be trivially copying")
            checks_passed = False

        if checks_passed:
            print("\n  [PASS] Initial losses look reasonable")
        else:
            print("\n  [WARN] Some checks flagged — review above")
        print("=" * 60)

    def run_overfit_one_batch(self):
        """Overfit a single batch to verify model capacity."""
        print("\n" + "=" * 60)
        print("OVERFIT ONE BATCH (Karpathy Recipe)")
        print("=" * 60)

        cover, secret = next(iter(self.train_loader))

        for pg in self.trainer.optimizer.param_groups:
            pg["lr"] = 1e-4

        n_steps = 500
        pbar = tqdm(range(n_steps), desc="Overfit")
        final_loss = None

        for step in pbar:
            metrics = self.trainer.train_step(cover, secret)
            final_loss = metrics["r_loss"]
            pbar.set_postfix({
                "total": f"{metrics['total_loss']:.4f}",
                "r_loss": f"{metrics['r_loss']:.4f}",
                "g_loss": f"{metrics['g_loss']:.4f}",
            })

        print(f"\n  Final r_loss: {final_loss:.6f}")
        if final_loss < 0.01:
            print("  [PASS] Model has sufficient capacity")
        elif final_loss < 0.10:
            print("  [WARN] r_loss somewhat high — may still work, monitor during training")
        else:
            print("  [FAIL] r_loss too high — architecture or LR problem")
        print("=" * 60)

    def run(self):
        """Full training loop with multi-stage resume and gradient safety."""
        start = self.args.start_epoch
        end = start + self.args.epochs
        use_watchdog = not self.args.no_grad_safety

        print("\n" + "=" * 60)
        print(f"TRAINING  (epochs {start + 1} -> {end})")
        print("=" * 60)

        csv_path = "results/training_log.csv"
        if self.args.resume and os.path.exists(csv_path):
            csv_file = open(csv_path, "a", newline="")
            csv_writer = csv.writer(csv_file)
        else:
            csv_file = open(csv_path, "w", newline="")
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([
                "epoch", "train_loss", "g_loss", "r_loss", "l_loss",
                "raw_grad_norm", "grad_norm",
                "val_psnr_stego", "val_ssim_stego",
                "val_psnr_secret", "val_ssim_secret", "lr",
            ])

        best_psnr_secret = 0.0
        gnorm_ema = None
        ema_alpha = 0.1

        for epoch in range(start, end):
            global_ep = epoch + 1
            self.model.train()
            epoch_losses = []

            desc = f"Epoch {global_ep}/{end}"
            pbar = tqdm(self.train_loader, desc=desc, leave=False)

            for cover, secret in pbar:
                metrics = self.trainer.train_step(cover, secret)
                epoch_losses.append(metrics)
                pbar.set_postfix({
                    "loss": f"{metrics['total_loss']:.1f}",
                    "r": f"{metrics['r_loss']:.1f}",
                    "g": f"{metrics['g_loss']:.1f}",
                    "gnorm": f"{metrics['raw_grad_norm']:.0f}",
                })

            self.trainer.scheduler.step()

            avg = {k: np.mean([m[k] for m in epoch_losses])
                   for k in epoch_losses[0]}

            lr = self.trainer.optimizer.param_groups[0]["lr"]
            raw_gn = avg["raw_grad_norm"]

            # --- Gradient norm watchdog ---
            if use_watchdog:
                if gnorm_ema is None:
                    gnorm_ema = raw_gn
                else:
                    threshold = self.args.grad_watch_factor * gnorm_ema
                    if raw_gn > threshold and global_ep > start + 5:
                        emer_path = f"checkpoints/hinet_emergency_epoch_{global_ep}.pth"
                        torch.save({
                            "epoch": epoch,
                            "net": self.model.state_dict(),
                            "opt": self.trainer.optimizer.state_dict(),
                        }, emer_path)
                        old_lr = lr
                        for pg in self.trainer.optimizer.param_groups:
                            pg["lr"] = pg["lr"] * 0.5
                        lr = self.trainer.optimizer.param_groups[0]["lr"]
                        print(f"  [WATCHDOG] Epoch {global_ep}: raw_grad_norm "
                              f"{raw_gn:.0f} > {threshold:.0f} "
                              f"({self.args.grad_watch_factor}x EMA) | "
                              f"LR {old_lr:.2e} -> {lr:.2e} | "
                              f"Saved {emer_path}")
                    gnorm_ema = ema_alpha * raw_gn + (1 - ema_alpha) * gnorm_ema

            val_metrics = None
            sample = None
            if global_ep % self.args.val_freq == 0:
                val_metrics, sample = self.trainer.validate(self.val_loader)

                print(f"  Epoch {global_ep:4d} | loss={avg['total_loss']:.1f} | "
                      f"gnorm={raw_gn:.0f} | "
                      f"PSNR(stego)={val_metrics['psnr_stego']:.2f}dB | "
                      f"PSNR(secret)={val_metrics['psnr_secret']:.2f}dB | "
                      f"SSIM(s)={val_metrics['ssim_secret']:.4f}")

                if val_metrics["psnr_secret"] > best_psnr_secret:
                    best_psnr_secret = val_metrics["psnr_secret"]
                    torch.save({
                        "epoch": epoch,
                        "net": self.model.state_dict(),
                        "opt": self.trainer.optimizer.state_dict(),
                        "psnr_secret": best_psnr_secret,
                    }, "checkpoints/hinet_best.pth")
                    print(f"    -> Best PSNR(secret)={best_psnr_secret:.2f}dB saved")

                if sample is not None:
                    cover_s, secret_s, steg_s, rev_s = sample
                    save_image_grid(
                        [cover_s, secret_s, steg_s, rev_s],
                        f"results/epoch_{global_ep}.png", nrow=4,
                    )
            else:
                print(f"  Epoch {global_ep:4d} | loss={avg['total_loss']:.1f} | "
                      f"g={avg['g_loss']:.1f} r={avg['r_loss']:.1f} l={avg['l_loss']:.1f} | "
                      f"gnorm={raw_gn:.0f} | lr={lr:.2e}")

            csv_writer.writerow([
                global_ep,
                f"{avg['total_loss']:.6f}",
                f"{avg['g_loss']:.6f}",
                f"{avg['r_loss']:.6f}",
                f"{avg['l_loss']:.6f}",
                f"{avg['raw_grad_norm']:.4f}",
                f"{avg['grad_norm']:.4f}",
                f"{val_metrics['psnr_stego']:.2f}" if val_metrics else "",
                f"{val_metrics['ssim_stego']:.4f}" if val_metrics else "",
                f"{val_metrics['psnr_secret']:.2f}" if val_metrics else "",
                f"{val_metrics['ssim_secret']:.4f}" if val_metrics else "",
                f"{lr:.2e}",
            ])
            csv_file.flush()

            if global_ep % self.args.checkpoint_every == 0:
                torch.save({
                    "epoch": epoch,
                    "net": self.model.state_dict(),
                    "opt": self.trainer.optimizer.state_dict(),
                }, f"checkpoints/hinet_epoch_{global_ep}.pth")

        csv_file.close()

        torch.save({
            "epoch": epoch,
            "net": self.model.state_dict(),
            "opt": self.trainer.optimizer.state_dict(),
        }, "checkpoints/hinet_final.pth")
        print(f"\n  Training complete (epochs {start + 1}-{end}). "
              f"Best PSNR(secret)={best_psnr_secret:.2f}dB")
        print(f"  Logs: {csv_path}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="HiNet Training")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3.16e-5,
                        help="Learning rate (default: 10^-4.5 = 3.16e-5)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint .pth to resume from")
    parser.add_argument("--start_epoch", type=int, default=0,
                        help="Global epoch offset when resuming (e.g. 500)")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--val_batch_size", type=int, default=2)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--val_crop_size", type=int, default=1024)
    parser.add_argument("--val_freq", type=int, default=10)
    parser.add_argument("--checkpoint_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_dir", type=str,
                        default="datasets/DIV2K_train_HR")
    parser.add_argument("--val_dir", type=str,
                        default="datasets/DIV2K_valid_HR")
    parser.add_argument("--max_grad_norm", type=float, default=10.0,
                        help="Per-step gradient clipping threshold (default: 10.0)")
    parser.add_argument("--grad_watch_factor", type=float, default=5.0,
                        help="Epoch watchdog triggers at factor * EMA baseline (default: 5.0)")
    parser.add_argument("--no_grad_safety", action="store_true",
                        help="Disable gradient clipping and watchdog (exact paper reproduction)")
    parser.add_argument("--sanity", action="store_true",
                        help="Run sanity checks only")
    parser.add_argument("--overfit_one_batch", action="store_true",
                        help="Run overfit-one-batch capacity test only")
    args = parser.parse_args()

    app = HiNetApp(args)

    if args.sanity:
        app.run_sanity()
    elif args.overfit_one_batch:
        app.run_overfit_one_batch()
    else:
        app.run()


if __name__ == "__main__":
    main()
