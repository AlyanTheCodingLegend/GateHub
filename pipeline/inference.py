"""
End-to-end GateHub inference pipeline.

Stages:
  1. YOLOv8 detects plate bounding boxes in the frame.
  2. CRNN reads the plate string from the crop.

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> list[dict]:
        """Standard pipeline for video frames — greedy decode, default conf."""
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
        Enhanced pipeline for still images — lower conf, larger imgsz,
        bbox padding, multi-variant preprocessing + beam-search decode.
        """
        h, w  = frame.shape[:2]
        imgsz = 1280 if max(h, w) > 800 else 640
        detections = self._detect(frame, conf=0.25, pad=0.05, imgsz=imgsz)
        results    = []

        for x1, y1, x2, y2, det_conf in detections:
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            plate = self._read_plate_best(crop)
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
        c      = conf if conf is not None else self.conf
        preds  = self.detector.predict(frame, conf=c, verbose=False, imgsz=imgsz)
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
            .unsqueeze(0).unsqueeze(0)
            .div(255.0)
            .to(self.device)
        )

    def _read_plate(self, crop: np.ndarray) -> str:
        """Resize → CLAHE → greedy decode."""
        gray = _crop_to_input(crop)
        gray = _CLAHE.apply(gray)
        with torch.no_grad():
            logits = self.crnn(self._to_tensor(gray))
        return greedy_decode(logits)[0]

    def _read_plate_best(self, crop: np.ndarray) -> str:
        """
        Multi-variant CRNN OCR for still images.

        Runs three preprocessing variants (plain, CLAHE, contrast boost)
        in a single batched forward pass, picks the variant with the
        highest mean per-step log-probability, then beam-search decodes it.
        """
        gray = _crop_to_input(crop)

        v_plain = gray
        v_clahe = _CLAHE.apply(gray)
        v_boost = cv2.convertScaleAbs(gray, alpha=1.3, beta=10)

        tensors = torch.stack([
            torch.tensor(v, dtype=torch.float32).unsqueeze(0).div(255.0)
            for v in (v_plain, v_clahe, v_boost)
        ]).to(self.device)  # (3, 1, H, W)

        with torch.no_grad():
            logits = self.crnn(tensors)  # (T, 3, C)

        scores   = logits.max(dim=2).values.mean(dim=0)  # (3,)
        best_idx = int(scores.argmax().item())

        best_logits = logits[:, best_idx:best_idx + 1, :]
        return beam_search_decode(best_logits)[0]
