"""
CNIC OCR module — reads the 13-digit CNIC number from a scanned card image.

Preprocessing pipeline:
  1. CLAHE equalisation  — normalises uneven lighting from hand-held scanning
  2. Otsu binarisation   — produces a clean black-on-white image for OCR
PaddleOCR runs on CPU (CNIC scanning is a one-shot action per vehicle entry,
so throughput is not a concern).
"""

import re
import cv2
import numpy as np
from paddleocr import PaddleOCR

_ocr: PaddleOCR | None = None


def _get_ocr() -> PaddleOCR:
    global _ocr
    if _ocr is None:
        _ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)
    return _ocr


def _preprocess(image: np.ndarray) -> np.ndarray:
    """Enhance contrast and binarise for clean digit extraction."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced  = clahe.apply(gray)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def _parse_cnic(text: str) -> str | None:
    """
    Extract a 13-digit CNIC from raw OCR text.

    Tries three strategies:
      1. Formatted match: XXXXX-XXXXXXX-X
      2. Exactly 13 consecutive digits after stripping non-digits
      3. First 13 digits when >= 13 digits are found
    """
    # Strategy 1 — formatted
    m = re.search(r'(\d{5})[- ](\d{7})[- ](\d)', text)
    if m:
        return m.group(1) + m.group(2) + m.group(3)

    digits = re.sub(r'\D', '', text)

    # Strategy 2 — exact 13 digits
    if len(digits) == 13:
        return digits

    # Strategy 3 — take first 13 if enough
    if len(digits) >= 13:
        return digits[:13]

    return None


def read_cnic(image: np.ndarray) -> dict:
    """
    Extract the 13-digit CNIC number from a scanned CNIC card image.

    Args:
        image: BGR numpy array (camera frame or scanned image).

    Returns:
        {
          'cnic':       '1234567890123'  or None if not found,
          'formatted':  '12345-6789012-3' or None,
          'raw_text':   full OCR output text,
          'confidence': mean OCR confidence score,
          'valid':      bool,
        }
    """
    preprocessed = _preprocess(image)
    ocr    = _get_ocr()
    result = ocr.ocr(preprocessed, cls=True)

    texts: list[str]  = []
    confs: list[float] = []

    if result and result[0]:
        for line in result[0]:
            text, conf = line[1]
            texts.append(text)
            confs.append(float(conf))

    raw_text = ' '.join(texts)
    cnic     = _parse_cnic(raw_text)
    avg_conf = float(np.mean(confs)) if confs else 0.0
    formatted = f'{cnic[:5]}-{cnic[5:12]}-{cnic[12]}' if cnic else None

    return {
        'cnic':       cnic,
        'formatted':  formatted,
        'raw_text':   raw_text,
        'confidence': avg_conf,
        'valid':      cnic is not None,
    }
