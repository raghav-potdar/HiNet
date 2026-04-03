#!/usr/bin/env python
"""HiNet training entry point with Karpathy-style debugging recipe.

Usage:
    python main.py --sanity
    python main.py --overfit_one_batch
    python main.py --epochs 100 --batch_size 16
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
              f"seed={args.seed}")

        self.pipeline = DataPipeline(crop_size=224, val_crop_size=224)
        self.train_loader, self.val_loader = self.pipeline.get_loaders(
            args.train_dir, args.val_dir,
            batch_size=args.batch_size,
            num_workers=self.dm.get_optimal_workers(),
        )

        self.model = HiNetModel().to(self.device)
        init_model(self.model)
        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"[Model] Parameters: {param_count:,}")

        self.trainer = HiNetTrainer(
            self.model, self.device, total_epochs=args.epochs,
        )

        self.scaler = None
        if self.dm.is_cuda and not args.no_amp:
            self.scaler = self.dm.get_scaler()
            print("[AMP] Enabled")

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
            output_steg = output.narrow(1, 0, 4 * 3)
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

        metrics = {
            "g_loss": g_loss, "r_loss": r_loss,
            "l_loss": l_loss, "total_loss": total_loss,
        }

        print(f"\n  g_loss (guide):          {metrics['g_loss']:.4f}")
        print(f"  r_loss (reconstruction): {metrics['r_loss']:.4f}")
        print(f"  l_loss (low frequency):  {metrics['l_loss']:.4f}")
        print(f"  total_loss:              {metrics['total_loss']:.4f}")

        checks_passed = True

        if metrics["g_loss"] > 1000:
            print("  [WARN] g_loss very high — cover/stego mismatch at init")
            checks_passed = False

        if metrics["r_loss"] < 0.001:
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
            metrics = self.trainer.train_step(
                cover, secret, phase=1, scaler=self.scaler,
                lambda_reconstruction_override=15.0,
            )
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
        """Full training loop with phased schedule."""
        print("\n" + "=" * 60)
        print("TRAINING")
        print("=" * 60)

        csv_path = "results/training_log.csv"
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "epoch", "phase", "train_loss", "g_loss", "r_loss", "l_loss",
            "val_psnr_stego", "val_ssim_stego",
            "val_psnr_secret", "val_ssim_secret", "lr",
        ])

        best_psnr_secret = 0.0
        patience_counter = 0

        for epoch in range(self.args.epochs):
            if epoch < 50:
                phase = 1
                lam_rec = 5.0
            else:
                phase = 2
                lam_rec = 7.0

            self.model.train()
            epoch_losses = []

            desc = f"Epoch {epoch}/{self.args.epochs} [P{phase}]"
            pbar = tqdm(self.train_loader, desc=desc, leave=False)

            for cover, secret in pbar:
                metrics = self.trainer.train_step(
                    cover, secret, phase=phase, scaler=self.scaler,
                    lambda_reconstruction_override=lam_rec,
                )
                epoch_losses.append(metrics)
                pbar.set_postfix({
                    "loss": f"{metrics['total_loss']:.3f}",
                    "r": f"{metrics['r_loss']:.3f}",
                    "g": f"{metrics['g_loss']:.3f}",
                })

            self.trainer.scheduler.step()

            avg = {k: np.mean([m[k] for m in epoch_losses])
                   for k in epoch_losses[0]}

            val_metrics, sample = self.trainer.validate(self.val_loader)

            lr = self.trainer.optimizer.param_groups[0]["lr"]
            csv_writer.writerow([
                epoch, phase,
                f"{avg['total_loss']:.6f}",
                f"{avg['g_loss']:.6f}",
                f"{avg['r_loss']:.6f}",
                f"{avg['l_loss']:.6f}",
                f"{val_metrics['psnr_stego']:.2f}",
                f"{val_metrics['ssim_stego']:.4f}",
                f"{val_metrics['psnr_secret']:.2f}",
                f"{val_metrics['ssim_secret']:.4f}",
                f"{lr:.2e}",
            ])
            csv_file.flush()

            print(f"  Epoch {epoch:3d} | P{phase} | loss={avg['total_loss']:.4f} | "
                  f"PSNR(stego)={val_metrics['psnr_stego']:.1f}dB | "
                  f"PSNR(secret)={val_metrics['psnr_secret']:.1f}dB | "
                  f"SSIM(s)={val_metrics['ssim_secret']:.3f}")

            if sample is not None:
                cover_s, secret_s, steg_s, rev_s = sample
                save_image_grid(
                    [cover_s, secret_s, steg_s, rev_s],
                    f"results/epoch_{epoch}.png", nrow=4,
                )

            if val_metrics["psnr_secret"] > best_psnr_secret + 0.1:
                best_psnr_secret = val_metrics["psnr_secret"]
                patience_counter = 0
                torch.save({
                    "epoch": epoch,
                    "net": self.model.state_dict(),
                    "opt": self.trainer.optimizer.state_dict(),
                    "psnr_secret": best_psnr_secret,
                }, "checkpoints/hinet_best.pth")
                print(f"    -> Best PSNR(secret)={best_psnr_secret:.2f}dB saved")
            else:
                patience_counter += 1

            if (epoch + 1) % self.args.checkpoint_every == 0:
                torch.save({
                    "epoch": epoch,
                    "net": self.model.state_dict(),
                    "opt": self.trainer.optimizer.state_dict(),
                }, f"checkpoints/hinet_epoch_{epoch}.pth")

            if epoch >= self.args.min_epochs and patience_counter >= self.args.patience:
                print(f"\n  [Early Stop] No improvement for {self.args.patience} epochs")
                break

        csv_file.close()

        torch.save({
            "epoch": epoch,
            "net": self.model.state_dict(),
            "opt": self.trainer.optimizer.state_dict(),
        }, "checkpoints/hinet_final.pth")
        print(f"\n  Training complete. Best PSNR(secret)={best_psnr_secret:.2f}dB")
        print(f"  Logs: {csv_path}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="HiNet Training")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min_epochs", type=int, default=30,
                        help="Minimum epochs before early stopping kicks in")
    parser.add_argument("--checkpoint_every", type=int, default=10)
    parser.add_argument("--train_dir", type=str,
                        default="datasets/DIV2K_train_HR")
    parser.add_argument("--val_dir", type=str,
                        default="datasets/DIV2K_valid_HR")
    parser.add_argument("--no_amp", action="store_true",
                        help="Disable automatic mixed precision")
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
