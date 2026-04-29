"""
End-to-end GateHub inference pipeline.

Stages:
  1. YOLOv8 detects plate bounding boxes in the frame.
  2. Each detected region is perspective-corrected using contour fitting.
  3. CRNN reads the plate string from the rectified crop.

Usage:
    from pipeline.inference import GateHubPipeline

    pipe = GateHubPipeline(
        detector_weights='runs/detect/gatehub/weights/best.pt',
        crnn_weights='checkpoints/crnn/best.pt',
    )
    results = pipe.process_frame(frame)   # frame is a BGR numpy array
    for r in results:
        print(r['plate'], r['confidence'])
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from ocr.model import CRNN, IMG_H, IMG_W
from ocr.decode import greedy_decode, beam_search_decode
from ocr.dataset import clean_label

_EASYOCR_ALLOWLIST = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-'


_CLAHE = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))


def _crop_to_input(crop: np.ndarray) -> np.ndarray:
    """
    Convert a BGR plate crop to a grayscale (IMG_H, IMG_W) array ready for CRNN.

    Uses INTER_CUBIC when upsampling (small crop → bigger canvas) and
    INTER_AREA when downsampling to avoid aliasing.  No perspective
    correction is applied; YOLO's bbox is already a tight, well-aligned
    crop and any contour-based warp risks distorting the characters.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    h, w = gray.shape[:2]
    interp = cv2.INTER_CUBIC if w < IMG_W or h < IMG_H else cv2.INTER_AREA
    return cv2.resize(gray, (IMG_W, IMG_H), interpolation=interp)


class GateHubPipeline:
    """
    Wraps detector + CRNN into a single callable that takes a BGR frame
    and returns a list of detected plate results.
    """

    def __init__(
        self,
        detector_weights: str,
        crnn_weights:     str,
        device:           str   = 'cuda',
        conf_threshold:   float = 0.4,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.conf   = conf_threshold

        self.detector = YOLO(detector_weights)

        self.crnn = CRNN().to(self.device)
        self.crnn.load_state_dict(
            torch.load(crnn_weights, map_location=self.device, weights_only=True)
        )
        self.crnn.eval()

        # EasyOCR for still-image path — trained on real photographs, far better
        # than the CRNN (which was trained on synthetic renders) for real plates.
        try:
            import easyocr
            self._easyocr = easyocr.Reader(
                ['en'],
                gpu=torch.cuda.is_available(),
                verbose=False,
            )
        except Exception:
            self._easyocr = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> list[dict]:
        """
        Standard pipeline for video frames (speed-optimised).
        Uses greedy decode and default conf threshold.
        """
        detections = self._detect(frame)
        results    = []

        for x1, y1, x2, y2, det_conf in detections:
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            plate = self._read_plate(crop)
            results.append({
                'plate':    plate,
                'bbox':     [x1, y1, x2, y2],
                'det_conf': round(float(det_conf), 3),
            })

        return results

    def process_image(self, frame: np.ndarray) -> list[dict]:
        """
        Enhanced pipeline for still images (accuracy-optimised).

        Differences from process_frame:
          - Lower conf threshold (0.25) to catch all plate candidates
          - Larger imgsz (1280) for high-resolution photos
          - Bbox padding to avoid clipping edge characters
          - Multi-variant preprocessing + beam-search decode for best OCR accuracy
        """
        h, w   = frame.shape[:2]
        imgsz  = 1280 if max(h, w) > 800 else 640
        detections = self._detect(frame, conf=0.25, pad=0.05, imgsz=imgsz)
        results    = []

        for x1, y1, x2, y2, det_conf in detections:
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            plate = self._read_plate_easyocr(crop)
            results.append({
                'plate':    plate,
                'bbox':     [x1, y1, x2, y2],
                'det_conf': round(float(det_conf), 3),
                'crop_rgb': cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
            })

        return results

    def read_plate_crop(self, crop: np.ndarray) -> str:
        """Run OCR on a pre-cropped plate image."""
        return self._read_plate(crop)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect(
        self,
        frame: np.ndarray,
        conf:  float = None,
        pad:   float = 0.0,
        imgsz: int   = 640,
    ) -> list[tuple]:
        c     = conf if conf is not None else self.conf
        preds = self.detector.predict(frame, conf=c, verbose=False, imgsz=imgsz)
        fh, fw = frame.shape[:2]
        boxes  = []
        for box in preds[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            if pad > 0:
                dx = int((x2 - x1) * pad)
                dy = int((y2 - y1) * pad)
                x1 = max(0, x1 - dx)
                y1 = max(0, y1 - dy)
                x2 = min(fw, x2 + dx)
                y2 = min(fh, y2 + dy)
            x1, y1 = max(0, x1), max(0, y1)
            boxes.append((x1, y1, x2, y2, box.conf.item()))
        return boxes

    def _to_tensor(self, gray: np.ndarray) -> torch.Tensor:
        return (
            torch.tensor(gray, dtype=torch.float32)
            .unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            .div(255.0)
            .to(self.device)
        )

    def _read_plate(self, crop: np.ndarray) -> str:
        """Standard OCR: resize → CLAHE → greedy decode."""
        gray = _crop_to_input(crop)
        gray = _CLAHE.apply(gray)
        with torch.no_grad():
            logits = self.crnn(self._to_tensor(gray))
        return greedy_decode(logits)[0]

    def _read_plate_easyocr(self, crop: np.ndarray) -> str:
        """
        EasyOCR-based OCR for still images.

        EasyOCR is trained on real-world photographs and handles real plate
        fonts, lighting, and textures far better than the CRNN, which was
        trained only on synthetically rendered plates.

        Falls back to CRNN if EasyOCR is unavailable or returns nothing.
        """
        if self._easyocr is not None:
            # Upscale tiny crops so EasyOCR has enough resolution
            h, w = crop.shape[:2]
            if w < 160 or h < 40:
                scale = max(160 / w, 40 / h)
                crop = cv2.resize(
                    crop,
                    (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_CUBIC,
                )
            results = self._easyocr.readtext(
                crop,
                detail=0,
                allowlist=_EASYOCR_ALLOWLIST,
            )
            text = clean_label(''.join(results))
            if text:
                return text

        # Fallback: CRNN with CLAHE preprocessing
        return self._read_plate(crop)
