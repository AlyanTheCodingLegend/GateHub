"""
Synthetic Pakistani license plate generator for CRNN OCR training.

Since no public Pakistani plate OCR dataset with text ground-truth exists,
we generate training data synthetically.  The plates follow real Pakistani
format conventions and are augmented to mimic gate-camera degradations.

Pakistani plate formats used (weighted by real-world frequency):
  60%  LLL-NNNN   e.g. "ABC-1234"   (3 letters + 4 digits)
  30%  LL-NNNN    e.g. "AB-1234"    (2 letters + 4 digits)
  10%  LLLL-NNN   e.g. "ABCD-123"   (4 letters + 3 digits)

Usage:
  python -m data.synth_plates          # generates 6 000 samples to data/ocr/
  python -m data.synth_plates --n 10000 --out data/ocr
"""

import argparse
import random
import string
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# ── plate string generation ───────────────────────────────────────────

_LETTERS = string.ascii_uppercase
_DIGITS  = string.digits

# City-code prefixes used on Pakistani plates (adds realism)
_CITY_CODES = [
    'KHI', 'LHE', 'ISB', 'RWP', 'FSB', 'MUL', 'PEW', 'QTA',
    'AB', 'AC', 'AE', 'AF', 'AG', 'AH', 'AJ', 'AK', 'AL',
    'LEB', 'LEA', 'LEC', 'LED', 'LEE', 'LEF', 'LEG',
]

def _random_plate_str() -> str:
    """Generate a random Pakistani plate string."""
    fmt = random.choices(
        ['LLL-NNNN', 'LL-NNNN', 'LLLL-NNN'],
        weights=[0.60, 0.30, 0.10],
    )[0]

    # Occasionally use a real city-code prefix for the letter part
    use_city = random.random() < 0.25
    if use_city:
        prefix = random.choice(_CITY_CODES)
        n_digits = 4 if len(prefix) <= 3 else 3
        digits   = ''.join(random.choices(_DIGITS, k=n_digits))
        return f'{prefix}-{digits}'

    letters = ''.join(random.choices(_LETTERS, k=fmt.count('L')))
    digits  = ''.join(random.choices(_DIGITS,  k=fmt.count('N')))
    result  = fmt
    for ch in letters:
        result = result.replace('L', ch, 1)
    for ch in digits:
        result = result.replace('N', ch, 1)
    return result


# ── font loading ──────────────────────────────────────────────────────

_FONT_CANDIDATES = [
    # Bold monospace — closest to plate fonts
    '/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf',
    '/usr/share/fonts/truetype/ubuntu/UbuntuMono-B.ttf',
    # Bold sans-serif fallbacks
    '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
    '/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf',
    '/usr/share/fonts/truetype/freefont/FreeSansBold.ttf',
    # macOS / Windows (for local dev)
    '/System/Library/Fonts/Helvetica.ttc',
    'C:/Windows/Fonts/arialbd.ttf',
    'C:/Windows/Fonts/courbd.ttf',
]

def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


# ── plate rendering ───────────────────────────────────────────────────

# Render at 2× CRNN input size so downscaling sharpens the result
_RENDER_W, _RENDER_H = 256, 64

# Pakistani plate background colours (BGR for OpenCV, RGB for PIL)
_BACKGROUNDS = [
    (255, 255, 255),   # white  — standard
    (255, 255, 180),   # pale yellow — commercial
    (230, 230, 230),   # light grey
    (245, 245, 220),   # beige
]

def _render_plate(text: str) -> np.ndarray:
    """
    Render a plate string as a clean RGB image.
    Returns a numpy array of shape (H, W, 3) in BGR.
    """
    bg = random.choice(_BACKGROUNDS)
    img  = Image.new('RGB', (_RENDER_W, _RENDER_H), bg)
    draw = ImageDraw.Draw(img)

    # Border
    border_w = random.randint(2, 4)
    draw.rectangle(
        [border_w, border_w, _RENDER_W - border_w - 1, _RENDER_H - border_w - 1],
        outline=(0, 0, 0),
        width=border_w,
    )

    # Font — vary size slightly for diversity
    font_size = random.randint(int(_RENDER_H * 0.55), int(_RENDER_H * 0.72))
    font = _load_font(font_size)

    # Centre text
    bbox    = draw.textbbox((0, 0), text, font=font)
    text_w  = bbox[2] - bbox[0]
    text_h  = bbox[3] - bbox[1]
    x = (_RENDER_W - text_w) // 2
    y = (_RENDER_H - text_h) // 2 - bbox[1]   # correct for ascent offset

    # Slight random colour variation on text (mostly black)
    text_colour = tuple(random.randint(0, 30) for _ in range(3))
    draw.text((x, y), text, fill=text_colour, font=font)

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ── augmentation pipeline ─────────────────────────────────────────────

def _augment(img: np.ndarray) -> np.ndarray:
    """
    Apply gate-camera degradations:
      - Motion blur      (moving vehicle)
      - Gaussian noise   (low-light sensor)
      - Brightness shift (lighting variation)
      - Perspective warp (camera angle)
      - JPEG compression (compressed stream)
    Each transform is applied with independent probability.
    """
    # Motion blur
    if random.random() < 0.4:
        k = random.choice([3, 5, 7])
        angle = random.choice([0, 90])   # horizontal or vertical
        kernel = np.zeros((k, k))
        if angle == 0:
            kernel[k // 2, :] = 1.0 / k
        else:
            kernel[:, k // 2] = 1.0 / k
        img = cv2.filter2D(img, -1, kernel)

    # Gaussian noise
    if random.random() < 0.4:
        noise = np.random.normal(0, random.uniform(3, 15), img.shape).astype(np.float32)
        img   = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Brightness / contrast
    if random.random() < 0.5:
        alpha = random.uniform(0.7, 1.3)   # contrast
        beta  = random.randint(-30, 30)    # brightness
        img   = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    # Perspective warp (mild — plates are fairly frontal at gates)
    if random.random() < 0.3:
        h, w = img.shape[:2]
        shift = int(w * random.uniform(0.02, 0.06))
        src   = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst   = np.float32([
            [random.randint(0, shift), random.randint(0, shift)],
            [w - random.randint(0, shift), random.randint(0, shift)],
            [w - random.randint(0, shift), h - random.randint(0, shift)],
            [random.randint(0, shift), h - random.randint(0, shift)],
        ])
        M   = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # JPEG compression artefacts
    if random.random() < 0.3:
        quality = random.randint(40, 85)
        _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        img    = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    return img


# ── dataset writer ────────────────────────────────────────────────────

def generate_dataset(out_root: Path, n: int = 6000, seed: int = 42) -> int:
    """
    Generate n synthetic Pakistani plate crop images + text labels.

    Layout written:
        <out_root>/images/<id>.png
        <out_root>/labels/<id>.txt   (plate string, e.g. "ABC-1234")

    Returns the number of samples successfully written.
    """
    random.seed(seed)
    np.random.seed(seed)

    imgs_dir = Path(out_root) / 'images'
    lbls_dir = Path(out_root) / 'labels'
    imgs_dir.mkdir(parents=True, exist_ok=True)
    lbls_dir.mkdir(parents=True, exist_ok=True)

    # Count existing synthetic samples so we don't regenerate needlessly
    existing = sorted(imgs_dir.glob('synth_*.png'))
    if len(existing) >= n:
        print(f'  {len(existing)} synthetic samples already present, skipping generation.')
        return len(existing)

    start_idx = len(existing)
    written   = 0

    for i in tqdm(range(start_idx, n), desc='Generating synthetic plates'):
        plate_str = _random_plate_str()
        img       = _render_plate(plate_str)
        img       = _augment(img)

        name = f'synth_{i:06d}'
        cv2.imwrite(str(imgs_dir / f'{name}.png'), img)
        (lbls_dir / f'{name}.txt').write_text(plate_str)
        written += 1

    total = start_idx + written
    print(f'  Synthetic dataset: {total} samples in {out_root}')
    return total


# ── CLI ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n',   type=int, default=6000, help='Number of samples to generate')
    parser.add_argument('--out', default='data/ocr',    help='Output directory')
    args = parser.parse_args()
    generate_dataset(Path(args.out), n=args.n)
