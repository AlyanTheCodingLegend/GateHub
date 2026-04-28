"""
Download and prepare all GateHub training datasets.

Steps performed:
  1. Clone DVLPD from github.com/usama-x930/VT-LPR
  2. Extract plate crops + reconstruct plate strings from character annotations
     (this is the OCR training set for the CRNN)
  3. Copy DVLPD plate-level boxes to detection/  (YOLOv8 training set)
  4. Download Roboflow PK-plates datasets (requires ROBOFLOW_API_KEY env var)
  5. Print a summary and next-step instructions

IMPORTANT — DVLPD class ID mapping
------------------------------------
The class order in DVLPD is defined in its classes.txt file.
Run this script once, then check data/dvlpd_raw/ for classes.txt and
update DVLPD_CLASSES below if needed.  The defaults match the typical
VT-LPR annotation format described in the paper.

Usage:
    python -m data.prepare
    # or with Roboflow key:
    ROBOFLOW_API_KEY=<key> python -m data.prepare
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

# ── paths ────────────────────────────────────────────────────────────
BASE         = Path('data')
DVLPD_RAW    = BASE / 'dvlpd_raw'
DET_ROOT     = BASE / 'detection'
OCR_ROOT     = BASE / 'ocr'

for split in ('train', 'val', 'test'):
    (DET_ROOT / 'images' / split).mkdir(parents=True, exist_ok=True)
    (DET_ROOT / 'labels' / split).mkdir(parents=True, exist_ok=True)
(OCR_ROOT / 'images').mkdir(parents=True, exist_ok=True)
(OCR_ROOT / 'labels').mkdir(parents=True, exist_ok=True)

# ── DVLPD class mapping ───────────────────────────────────────────────
# class 0 = vehicle, 1 = plate, 2–11 = digits 0–9, 12–37 = letters A–Z
# Verify against data/dvlpd_raw/classes.txt after cloning.
PLATE_CLASS = 1
DIGIT_START = 2    # class 2 → digit '0', class 3 → '1', ...
ALPHA_START = 12   # class 12 → 'A', class 13 → 'B', ...


def _cls_to_char(cls_id: int) -> Optional[str]:
    if DIGIT_START <= cls_id <= DIGIT_START + 9:
        return str(cls_id - DIGIT_START)
    if ALPHA_START <= cls_id <= ALPHA_START + 25:
        return chr(ord('A') + cls_id - ALPHA_START)
    return None


# ── step 1: clone DVLPD ──────────────────────────────────────────────

def clone_dvlpd() -> Path:
    if DVLPD_RAW.exists() and any(DVLPD_RAW.iterdir()):
        print(f'DVLPD already present at {DVLPD_RAW}, skipping clone.')
        return DVLPD_RAW
    print('Cloning DVLPD (github.com/usama-x930/VT-LPR)...')
    subprocess.run(
        ['git', 'clone', '--depth', '1',
         'https://github.com/usama-x930/VT-LPR.git', str(DVLPD_RAW)],
        check=True,
    )
    print('Clone complete.')
    return DVLPD_RAW


# ── step 2: extract plate crops → OCR dataset ────────────────────────

def build_ocr_dataset(dvlpd_root: Path) -> int:
    """
    For each DVLPD image:
      - Find the plate bounding box (class PLATE_CLASS).
      - Crop the plate region from the image.
      - Collect all character annotations whose centre falls inside the plate.
      - Sort characters left-to-right to reconstruct the plate string.
      - Save crop as PNG + label as TXT.

    Returns the number of successfully extracted plate samples.
    """
    # Try to locate images/ and labels/ directories in the cloned repo.
    # The VT-LPR repo typically stores them under dataset/ or directly.
    candidates = [
        dvlpd_root / 'dataset',
        dvlpd_root / 'Dataset',
        dvlpd_root,
    ]
    images_dir = labels_dir = None
    for cand in candidates:
        if (cand / 'images').exists() and (cand / 'labels').exists():
            images_dir = cand / 'images'
            labels_dir = cand / 'labels'
            break

    if images_dir is None:
        print(
            'WARNING: Could not locate images/ and labels/ inside the DVLPD repo.\n'
            f'  Searched: {[str(c) for c in candidates]}\n'
            '  Please check the repo structure and update build_ocr_dataset() accordingly.'
        )
        return 0

    # Print classes.txt so the user can verify the mapping
    classes_file = next(
        (f for f in [labels_dir.parent / 'classes.txt',
                     dvlpd_root / 'classes.txt'] if f.exists()),
        None,
    )
    if classes_file:
        print(f'\nDVLPD classes ({classes_file}):')
        for i, line in enumerate(classes_file.read_text().splitlines()):
            print(f'  {i:2d}: {line}')
        print()

    ocr_imgs = OCR_ROOT / 'images'
    ocr_lbls = OCR_ROOT / 'labels'
    count = 0

    img_paths = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
    for img_path in tqdm(img_paths, desc='Building OCR dataset from DVLPD'):
        lbl_path = labels_dir / (img_path.stem + '.txt')
        if not lbl_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]

        annotations: list[tuple[int, float, float, float, float]] = []
        for line in lbl_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cid, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:5])
            annotations.append((cid, cx, cy, bw, bh))

        plate_anns = [a for a in annotations if a[0] == PLATE_CLASS]
        char_anns  = [a for a in annotations if a[0] != PLATE_CLASS and a[0] >= DIGIT_START]

        for p_cid, p_cx, p_cy, p_bw, p_bh in plate_anns:
            px1 = int((p_cx - p_bw / 2) * W)
            py1 = int((p_cy - p_bh / 2) * H)
            px2 = int((p_cx + p_bw / 2) * W)
            py2 = int((p_cy + p_bh / 2) * H)
            px1, py1 = max(0, px1), max(0, py1)

            crop = img[py1:py2, px1:px2]
            if crop.size == 0:
                continue

            # Characters whose centre falls within the plate bbox
            chars_in_plate: list[tuple[float, str]] = []
            for c_cid, c_cx, c_cy, *_ in char_anns:
                cx_abs = c_cx * W
                cy_abs = c_cy * H
                if px1 <= cx_abs <= px2 and py1 <= cy_abs <= py2:
                    ch = _cls_to_char(c_cid)
                    if ch:
                        chars_in_plate.append((cx_abs, ch))

            # Sort left-to-right
            chars_in_plate.sort(key=lambda x: x[0])
            plate_str = ''.join(c for _, c in chars_in_plate)

            if len(plate_str) < 2:
                continue

            name = f'dvlpd_{count:06d}'
            cv2.imwrite(str(ocr_imgs / f'{name}.png'), crop)
            (ocr_lbls / f'{name}.txt').write_text(plate_str)
            count += 1

    print(f'OCR dataset: {count} plate crops saved to {OCR_ROOT}')
    return count


# ── step 3: copy DVLPD detection labels for YOLOv8 ──────────────────

def build_detection_dataset(dvlpd_root: Path, val_frac: float = 0.15, test_frac: float = 0.05) -> None:
    """
    Copy DVLPD images + plate bounding-box labels to data/detection/
    in the YOLOv8 train/val/test split expected by dataset.yaml.

    We re-label everything as class 0 ('plate') to match the single-class setup.
    """
    candidates = [dvlpd_root / 'dataset', dvlpd_root / 'Dataset', dvlpd_root]
    images_dir = labels_dir = None
    for cand in candidates:
        if (cand / 'images').exists():
            images_dir = cand / 'images'
            labels_dir = cand / 'labels'
            break

    if images_dir is None:
        print('WARNING: DVLPD images not found; skipping detection dataset build.')
        return

    img_paths = sorted(list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png')))
    n         = len(img_paths)
    n_val     = max(1, int(n * val_frac))
    n_test    = max(1, int(n * test_frac))
    n_train   = n - n_val - n_test

    splits = (
        ('train', img_paths[:n_train]),
        ('val',   img_paths[n_train:n_train + n_val]),
        ('test',  img_paths[n_train + n_val:]),
    )

    for split_name, paths in splits:
        for img_path in tqdm(paths, desc=f'Detection {split_name}'):
            lbl_path = labels_dir / (img_path.stem + '.txt') if labels_dir else None

            dst_img = DET_ROOT / 'images' / split_name / img_path.name
            shutil.copy2(img_path, dst_img)

            if lbl_path and lbl_path.exists():
                dst_lbl = DET_ROOT / 'labels' / split_name / (img_path.stem + '.txt')
                plate_lines = []
                for line in lbl_path.read_text().splitlines():
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    if int(parts[0]) == PLATE_CLASS:
                        # Re-label as class 0 for single-class detection
                        plate_lines.append('0 ' + ' '.join(parts[1:]))
                if plate_lines:
                    dst_lbl.write_text('\n'.join(plate_lines))

    print(f'Detection dataset: {n_train} train / {n_val} val / {n_test} test')


# ── step 4: download Roboflow datasets (detection only) ──────────────

def download_roboflow(workspace: str, project: str, version: int, save_dir: Path) -> None:
    api_key = os.environ.get('ROBOFLOW_API_KEY')
    if not api_key:
        print(f'  Skipping {project} — set ROBOFLOW_API_KEY to enable Roboflow downloads.')
        return
    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key=api_key)
        rf.workspace(workspace).project(project).version(version).download(
            'yolov8', location=str(save_dir)
        )
        print(f'  Downloaded {project} to {save_dir}')
    except Exception as e:
        print(f'  Roboflow download failed for {project}: {e}')


# ── main ─────────────────────────────────────────────────────────────

def main() -> None:
    print('=' * 55)
    print('  GateHub — dataset preparation')
    print('=' * 55)

    dvlpd_root = clone_dvlpd()
    build_detection_dataset(dvlpd_root)
    n_ocr = build_ocr_dataset(dvlpd_root)

    print('\nDownloading Roboflow detection datasets...')
    download_roboflow(
        'burhan-khan', 'pk-number-plates', 1,
        DET_ROOT / 'roboflow_burhan',
    )
    download_roboflow(
        'malik-kashif-saeed-aswwf', 'pakistani-number-plates', 1,
        DET_ROOT / 'roboflow_malik',
    )

    print('\n' + '=' * 55)
    print('  Done.')
    print(f'  OCR samples   : {n_ocr}  (data/ocr/)')
    print(f'  Detection data: data/detection/')
    print()
    print('  IMPORTANT: open data/dvlpd_raw/classes.txt and verify')
    print('  that PLATE_CLASS, DIGIT_START, ALPHA_START in this file')
    print('  match the actual class IDs in the dataset.')
    print()
    print('  Next steps:')
    print('    1.  python -m detection.train')
    print('    2.  python -m ocr.train')
    print('    3.  streamlit run dashboard/app.py')
    print('=' * 55)


if __name__ == '__main__':
    main()
