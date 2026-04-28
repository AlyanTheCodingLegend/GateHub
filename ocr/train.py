"""
Train the CRNN OCR model on Pakistani plate crops.

Usage:
    python -m ocr.train --data data/ocr --epochs 30
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from ocr.model import CRNN, NUM_CLASSES, SEQ_LEN
from ocr.dataset import PlateDataset, collate_fn
from ocr.decode import greedy_decode


def _make_loaders(
    data_root: str,
    batch_size: int,
    val_frac: float = 0.15,
) -> tuple[DataLoader, DataLoader]:
    full = PlateDataset(data_root, augment=False)
    aug  = PlateDataset(data_root, augment=True)

    n_val   = max(1, int(len(full) * val_frac))
    n_train = len(full) - n_val

    indices = torch.randperm(len(full), generator=torch.Generator().manual_seed(42)).tolist()
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    from torch.utils.data import Subset
    train_set = Subset(aug,  train_idx)
    val_set   = Subset(full, val_idx)

    import platform
    workers = 0 if platform.system() == 'Windows' else 4
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )
    return train_loader, val_loader


def _word_accuracy(model: CRNN, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, _, _, raw_labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            preds  = greedy_decode(logits)
            for pred, label in zip(preds, raw_labels):
                correct += int(pred == label)
                total   += 1
    return correct / total if total > 0 else 0.0


def train(
    data_root: str,
    save_dir:  str  = 'checkpoints/crnn',
    epochs:    int  = 30,
    batch_size: int = 64,
    lr:        float = 1e-3,
    device:    str  = 'cuda',
) -> CRNN:
    device_ = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device_}')
    if device_.type == 'cuda':
        print(f'GPU    : {torch.cuda.get_device_name(device_)}')

    train_loader, val_loader = _make_loaders(data_root, batch_size)
    print(f'Train  : {len(train_loader.dataset)} samples')
    print(f'Val    : {len(val_loader.dataset)} samples')

    model     = CRNN().to(device_)
    criterion = nn.CTCLoss(blank=NUM_CLASSES - 1, reduction='mean', zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.1,
    )

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    history: dict[str, list] = {'train_loss': [], 'val_acc': [], 'lr': []}
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch:3d}/{epochs}', leave=False)
        for imgs, targets, target_lengths, _ in pbar:
            imgs           = imgs.to(device_)
            targets        = targets.to(device_)
            target_lengths = target_lengths.to(device_)

            logits = model(imgs)              # (T, B, C)
            T, B, _ = logits.shape
            input_lengths = torch.full((B,), T, dtype=torch.long, device=device_)

            loss = criterion(logits, targets, input_lengths, target_lengths)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=f'{loss.item():.4f}')

        avg_loss = running_loss / len(train_loader)
        val_acc  = _word_accuracy(model, val_loader, device_)
        cur_lr   = scheduler.get_last_lr()[0]

        history['train_loss'].append(avg_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(cur_lr)

        print(f'Epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}  lr={cur_lr:.2e}')

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path / 'best.pt')
            print(f'           -> saved best (acc={best_acc:.4f})')

    torch.save(model.state_dict(), save_path / 'last.pt')
    with open(save_path / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f'\nFinished. Best val word-accuracy: {best_acc:.4f}')
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',       default='data/ocr',      help='OCR dataset root')
    parser.add_argument('--save',       default='checkpoints/crnn')
    parser.add_argument('--epochs',     type=int,   default=30)
    parser.add_argument('--batch',      type=int,   default=64)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--device',     default='cuda')
    args = parser.parse_args()

    train(
        data_root  = args.data,
        save_dir   = args.save,
        epochs     = args.epochs,
        batch_size = args.batch,
        lr         = args.lr,
        device     = args.device,
    )
