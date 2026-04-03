import glob
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T


def _to_rgb(image):
    if image.mode != "RGB":
        rgb = Image.new("RGB", image.size)
        rgb.paste(image)
        return rgb
    return image


class DIV2KDataset(Dataset):
    EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp")

    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.files = []
        root = Path(root_dir)
        for ext in self.EXTS:
            self.files.extend(sorted(root.glob(ext)))
        self.files.sort()
        if len(self.files) == 0:
            raise FileNotFoundError(f"No images found in {root_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = Image.open(self.files[idx])
        image = _to_rgb(image)
        if self.transform:
            image = self.transform(image)
        return image


def split_batch_collate(batch):
    """Stack batch, then split first half = cover, second half = secret."""
    images = torch.stack(batch)
    if images.shape[0] % 2 != 0:
        images = images[:-1]
    mid = images.shape[0] // 2
    cover = images[mid:]
    secret = images[:mid]
    return cover, secret


class DataPipeline:
    def __init__(self, crop_size=224, val_crop_size=224):
        self.train_transform = T.Compose([
            T.RandomCrop(crop_size),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ToTensor(),
        ])
        self.val_transform = T.Compose([
            T.CenterCrop(val_crop_size),
            T.ToTensor(),
        ])

    def get_loaders(self, train_dir, val_dir, batch_size, num_workers=4):
        if batch_size % 2 != 0:
            batch_size += 1
            print(f"[Data] Adjusted batch_size to {batch_size} (must be even)")

        train_ds = DIV2KDataset(train_dir, transform=self.train_transform)
        val_ds = DIV2KDataset(val_dir, transform=self.val_transform)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=split_batch_collate,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=max(2, batch_size // 4),
            shuffle=False,
            num_workers=max(1, num_workers // 2),
            pin_memory=True,
            drop_last=True,
            collate_fn=split_batch_collate,
        )
        print(f"[Data] Train: {len(train_ds)} images, Val: {len(val_ds)} images")
        return train_loader, val_loader

    def get_val_loader(self, val_dir, batch_size=4, num_workers=2):
        if batch_size % 2 != 0:
            batch_size += 1
        val_ds = DIV2KDataset(val_dir, transform=self.val_transform)
        return DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=split_batch_collate,
        )
