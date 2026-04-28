"""
GateHub — dataset preparation.

Detection data  (bounding-box annotations, YOLO format):
  Primary  : Kaggle  ubaidp1049/pakistani-vehicle-number-plate-anpr-yolo  (~900 imgs)
  Secondary: Kaggle  zakirkhanaleemi/pakistani-car-number-plates-data
  Extra    : Roboflow  burhan-khan/pk-number-plates  (1 678 imgs)
             Roboflow  malik-kashif-saeed-aswwf/pakistani-number-plates  (336 imgs)

OCR data  (plate-crop images + text labels):
  Synthetic plate generator (data/synth_plates.py).
  No public Pakistani plate OCR text dataset exists — synthetic generation
  is the standard approach for this domain.

Prerequisites
─────────────
Kaggle (required for detection data):
  1. Create a free account at kaggle.com
  2. Account → Settings → Create API Token  →  downloads kaggle.json
  3. mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
  4. chmod 600 ~/.kaggle/kaggle.json

Roboflow (optional, adds ~2 000 more detection images):
  export ROBOFLOW_API_KEY=<your_key>   (free account at roboflow.com)

Usage:
  python -m data.prepare
"""

import os
import shutil
import subprocess
import zipfile
from pathlib import Path

from tqdm import tqdm

# ── directory layout ─────────────────────────────────────────────────
BASE     = Path('data')
DET_ROOT = BASE / 'detection'
OCR_ROOT = BASE / 'ocr'

for split in ('train', 'val', 'test'):
    (DET_ROOT / 'images' / split).mkdir(parents=True, exist_ok=True)
    (DET_ROOT / 'labels' / split).mkdir(parents=True, exist_ok=True)
(OCR_ROOT / 'images').mkdir(parents=True, exist_ok=True)
(OCR_ROOT / 'labels').mkdir(parents=True, exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────

def _check_kaggle() -> bool:
    kaggle_dir = Path.home() / '.kaggle'
    # Accept kaggle.json (classic) or access_token (newer Kaggle auth)
    has_creds = (
        (kaggle_dir / 'kaggle.json').exists()
        or (kaggle_dir / 'access_token').exists()
        or os.environ.get('KAGGLE_USERNAME')
    )
    if not has_creds:
        print(
            '\n  Kaggle credentials not found.\n'
            '  To download Pakistani plate detection data:\n'
            '    1. Create a free account at https://kaggle.com\n'
            '    2. Account → Settings → Create API Token\n'
            '    3. mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json\n'
            '    4. chmod 600 ~/.kaggle/kaggle.json\n'
            '    5. Re-run python -m data.prepare\n'
        )
        return False
    return True


def _kaggle_download(dataset_slug: str, dest: Path) -> bool:
    """Download and unzip a Kaggle dataset. Returns True on success."""
    if dest.exists() and any(dest.iterdir()):
        print(f'  {dataset_slug}: already downloaded, skipping.')
        return True
    dest.mkdir(parents=True, exist_ok=True)
    print(f'  Downloading {dataset_slug} ...')
    result = subprocess.run(
        ['kaggle', 'datasets', 'download', dataset_slug,
         '-p', str(dest), '--unzip'],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f'  ERROR: {result.stderr.strip()}')
        return False
    print(f'  Done → {dest}')
    return True


def _roboflow_download(workspace: str, project: str, version: int, dest: Path) -> bool:
    api_key = os.environ.get('ROBOFLOW_API_KEY')
    if not api_key:
        print(f'  Skipping {project} (no ROBOFLOW_API_KEY set).')
        return False
    if dest.exists() and any(dest.iterdir()):
        print(f'  {project}: already downloaded, skipping.')
        return True
    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key=api_key)
        rf.workspace(workspace).project(project).version(version).download(
            'yolov8', location=str(dest)
        )
        print(f'  Done → {dest}')
        return True
    except Exception as exc:
        print(f'  Roboflow download failed for {project}: {exc}')
        return False


# ── detection dataset merging ─────────────────────────────────────────

_IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}


def _collect_pairs(root: Path) -> list[tuple[Path, Path]]:
    """
    Recursively find all (image, label) pairs under root by matching
    filenames stems.  Works for any nesting depth or directory layout
    (flat, images/images/, train/images/, etc.).
    """
    # Build a stem → label path index from all .txt files found anywhere
    label_index: dict[str, Path] = {}
    for lbl in root.rglob('*.txt'):
        # Skip yaml/config-like txt files that have no matching image
        label_index[lbl.stem] = lbl

    pairs: list[tuple[Path, Path]] = []
    for img in root.rglob('*'):
        if img.suffix.lower() not in _IMG_EXTS:
            continue
        if img.stem in label_index:
            pairs.append((img, label_index[img.stem]))

    return pairs


def merge_yolo_dataset(src_root: Path, val_frac: float = 0.15,
                       test_frac: float = 0.05) -> int:
    """
    Merge a downloaded YOLO dataset into data/detection/.
    Handles any directory layout — flat, double-nested, pre-split, etc.
    Returns number of images merged.
    """
    import random as _random

    counter = [_count_existing_detection_images()]
    start   = counter[0]

    def _copy(img_path: Path, lbl_path: Path, split: str) -> None:
        new_stem = f'ds{counter[0]:06d}'
        shutil.copy2(img_path, DET_ROOT / 'images' / split / (new_stem + img_path.suffix))
        shutil.copy2(lbl_path, DET_ROOT / 'labels' / split / (new_stem + '.txt'))
        counter[0] += 1

    # Check for pre-existing train / valid / test split directories
    found_presplit = False
    for split_name, dir_names in [('train', ['train']),
                                   ('val',   ['valid', 'val']),
                                   ('test',  ['test'])]:
        for d in dir_names:
            split_dir = src_root / d
            if not split_dir.exists():
                continue
            for img_path, lbl_path in _collect_pairs(split_dir):
                _copy(img_path, lbl_path, split_name)
            found_presplit = True
            break

    if found_presplit and counter[0] > start:
        return counter[0] - start

    # No pre-split — collect everything and split ourselves
    all_pairs = _collect_pairs(src_root)
    if not all_pairs:
        print(f'  WARNING: no annotated image+label pairs found in {src_root}')
        return 0

    _random.seed(42)
    _random.shuffle(all_pairs)
    n       = len(all_pairs)
    n_val   = max(1, int(n * val_frac))
    n_test  = max(1, int(n * test_frac))
    n_train = n - n_val - n_test

    for img_path, lbl_path in all_pairs[:n_train]:
        _copy(img_path, lbl_path, 'train')
    for img_path, lbl_path in all_pairs[n_train:n_train + n_val]:
        _copy(img_path, lbl_path, 'val')
    for img_path, lbl_path in all_pairs[n_train + n_val:]:
        _copy(img_path, lbl_path, 'test')

    return counter[0] - start


def _count_existing_detection_images() -> int:
    return sum(
        1 for split in ('train', 'val', 'test')
        for _ in (DET_ROOT / 'images' / split).glob('*')
    )


def _clean_unannotated_detection_data() -> None:
    """
    Remove any images that were copied previously without a matching label file.
    This cleans up the incorrect DVLPD copy from the first prepare run.
    """
    removed = 0
    for split in ('train', 'val', 'test'):
        imgs_dir = DET_ROOT / 'images' / split
        lbls_dir = DET_ROOT / 'labels' / split
        for img_path in list(imgs_dir.glob('*')):
            lbl_path = lbls_dir / (img_path.stem + '.txt')
            if not lbl_path.exists():
                img_path.unlink()
                removed += 1
    if removed:
        print(f'  Cleaned {removed} unannotated images from previous run.')


# ── main ─────────────────────────────────────────────────────────────

def main() -> None:
    print('=' * 58)
    print('  GateHub — dataset preparation')
    print('=' * 58)

    # Clean up any incorrectly copied data from a previous run
    _clean_unannotated_detection_data()

    # ── Detection datasets ──────────────────────────────────────────
    print('\n[1/3] Detection datasets (Kaggle)')
    total_det = 0

    if _check_kaggle():
        raw_dir = BASE / 'raw'

        # Primary: Pakistani ANPR YOLO dataset (~900 annotated images)
        slug1 = 'ubaidp1049/pakistani-vehicle-number-plate-anpr-yolo'
        dest1 = raw_dir / 'ubaid_anpr'
        if _kaggle_download(slug1, dest1):
            n = merge_yolo_dataset(dest1)
            print(f'    ubaid_anpr  : +{n} images merged')
            total_det += n

        # Secondary: Zakir plates (images only — used for detection if annotated)
        slug2 = 'zakirkhanaleemi/pakistani-car-number-plates-data'
        dest2 = raw_dir / 'zakir_plates'
        if _kaggle_download(slug2, dest2):
            n = merge_yolo_dataset(dest2)
            print(f'    zakir_plates: +{n} images merged')
            total_det += n

    # ── Roboflow (optional extra data) ─────────────────────────────
    print('\n[2/3] Detection datasets (Roboflow — optional)')
    raw_dir = BASE / 'raw'

    dest_burhan = raw_dir / 'roboflow_burhan'
    if _roboflow_download('burhan-khan', 'pk-number-plates', 1, dest_burhan):
        n = merge_yolo_dataset(dest_burhan)
        print(f'    roboflow_burhan: +{n} images merged')
        total_det += n

    dest_malik = raw_dir / 'roboflow_malik'
    if _roboflow_download('malik-kashif-saeed-aswwf', 'pakistani-number-plates', 1, dest_malik):
        n = merge_yolo_dataset(dest_malik)
        print(f'    roboflow_malik : +{n} images merged')
        total_det += n

    # ── OCR synthetic data ─────────────────────────────────────────
    print('\n[3/3] OCR dataset (synthetic Pakistani plates)')
    from data.synth_plates import generate_dataset
    n_ocr = generate_dataset(OCR_ROOT, n=6000)

    # ── Summary ────────────────────────────────────────────────────
    det_counts = {
        split: len(list((DET_ROOT / 'images' / split).glob('*')))
        for split in ('train', 'val', 'test')
    }
    print('\n' + '=' * 58)
    print('  Done.')
    print(f'\n  Detection  : {det_counts["train"]} train / '
          f'{det_counts["val"]} val / {det_counts["test"]} test')
    print(f'  OCR samples: {n_ocr} synthetic plate crops')
    print()
    if total_det == 0:
        print('  WARNING: No detection images were downloaded.')
        print('  Set up Kaggle credentials (see instructions above)')
        print('  and re-run python -m data.prepare.')
    print()
    print('  Next steps:')
    print('    python -m detection.train --ablation --epochs 30')
    print('    python -m ocr.train --data data/ocr')
    print('=' * 58)


if __name__ == '__main__':
    main()
