"""
Dataset and data loading utilities for CRNN OCR training.

Expected directory layout:
    <root>/
        images/   — plate crop PNGs  (<id>.png)
        labels/   — matching text files  (<id>.txt, one plate string per file)

The prepare.py script in data/ builds this layout from DVLPD.
"""

import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from ocr.model import CHAR2IDX, CHARSET, IMG_H, IMG_W


def clean_label(text: str) -> str:
    """Uppercase and keep only chars present in CHARSET."""
    return ''.join(c for c in text.upper() if c in CHARSET)


# Training augmentation pipeline — simulates gate-camera degradations:
#   motion blur (moving vehicle), brightness shifts (lighting variation),
#   perspective distortion (camera angle), gaussian noise (low-light sensor).
_train_aug = A.Compose([
    A.MotionBlur(blur_limit=(3, 7), p=0.3),
    A.GaussNoise(var_limit=(5, 25), p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    A.Perspective(scale=(0.02, 0.05), p=0.3),
    A.Sharpen(alpha=(0.1, 0.3), p=0.2),
])


class PlateDataset(Dataset):
    """
    Loads plate-crop images and their text labels.

    Args:
        root:    path to directory with images/ and labels/ subdirs.
        augment: if True, apply training augmentations.
    """

    def __init__(self, root: str, augment: bool = False):
        self.augment = augment
        self.samples: list[tuple[str, str]] = []

        root = Path(root)
        images_dir = root / 'images'
        labels_dir = root / 'labels'

        for img_path in sorted(images_dir.glob('*.png')):
            lbl_path = labels_dir / (img_path.stem + '.txt')
            if not lbl_path.exists():
                continue
            label = clean_label(lbl_path.read_text().strip())
            if len(label) < 2:
                continue
            self.samples.append((str(img_path), label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, list[int], str]:
        img_path, label = self.samples[idx]

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f'Cannot read image: {img_path}')

        img = cv2.resize(img, (IMG_W, IMG_H))

        if self.augment:
            img = _train_aug(image=img)['image']

        # Normalise to [0, 1] and add channel dim  →  (1, H, W)
        img = img.astype(np.float32) / 255.0
        img = img[np.newaxis]

        encoded = [CHAR2IDX[c] for c in label]
        return img, encoded, label


def collate_fn(
    batch: list[tuple[np.ndarray, list[int], str]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[str, ...]]:
    """
    Custom collate for variable-length CTC targets.

    Returns:
        imgs           (B, 1, H, W)
        targets        (sum of label lengths,)   — concatenated, as required by CTCLoss
        target_lengths (B,)
        raw_labels     tuple of original strings
    """
    imgs, encoded_list, raw_labels = zip(*batch)
    imgs_t = torch.tensor(np.stack(imgs), dtype=torch.float32)
    targets = torch.tensor(
        [idx for enc in encoded_list for idx in enc], dtype=torch.long
    )
    target_lengths = torch.tensor([len(e) for e in encoded_list], dtype=torch.long)
    return imgs_t, targets, target_lengths, raw_labels
