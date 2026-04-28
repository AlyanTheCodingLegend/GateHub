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
from ocr.decode import greedy_decode


def _order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order four points as: top-left, top-right, bottom-right, bottom-left.
    Required by cv2.getPerspectiveTransform.
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s       = pts.sum(axis=1)
    diff    = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]      # smallest sum  → top-left
    rect[2] = pts[np.argmax(s)]      # largest sum   → bottom-right
    rect[1] = pts[np.argmin(diff)]   # smallest diff → top-right
    rect[3] = pts[np.argmax(diff)]   # largest diff  → bottom-left
    return rect


def _rectify(crop: np.ndarray) -> np.ndarray:
    """
    Perspective correction on a plate crop.

    Attempts to find a quadrilateral contour; if found, warps it to a
    canonical IMG_W × IMG_H rectangle.  Falls back to a plain resize if
    no suitable contour is detected (e.g. for already-frontal crops).

    This implements the detect-then-rectify step from Yousaf et al. (2021),
    which they showed improves OCR accuracy on fixed-angle gate cameras.
    """
    gray    = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        hull    = cv2.convexHull(largest)
        peri    = cv2.arcLength(hull, True)
        approx  = cv2.approxPolyDP(hull, 0.02 * peri, True)

        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype(np.float32)
            pts = _order_points(pts)
            dst = np.array(
                [[0, 0], [IMG_W - 1, 0], [IMG_W - 1, IMG_H - 1], [0, IMG_H - 1]],
                dtype=np.float32,
            )
            M = cv2.getPerspectiveTransform(pts, dst)
            return cv2.warpPerspective(crop, M, (IMG_W, IMG_H))

    # Fallback — no quad found, just resize
    return cv2.resize(crop, (IMG_W, IMG_H))


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
        conf_threshold:   float = 0.5,
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
        """
        Run full pipeline on one BGR frame.

        Returns a list of dicts:
            {'plate': 'ABC1234', 'bbox': [x1,y1,x2,y2], 'det_conf': 0.93}
        """
        detections = self._detect(frame)
        results    = []

        for x1, y1, x2, y2, det_conf in detections:
            crop    = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            plate   = self._read_plate(crop)
            results.append({
                'plate':    plate,
                'bbox':     [x1, y1, x2, y2],
                'det_conf': round(float(det_conf), 3),
            })

        return results

    def read_plate_crop(self, crop: np.ndarray) -> str:
        """Run OCR on a pre-cropped plate image."""
        return self._read_plate(crop)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect(self, frame: np.ndarray) -> list[tuple]:
        preds = self.detector.predict(frame, conf=self.conf, verbose=False)
        boxes = []
        for box in preds[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            boxes.append((x1, y1, x2, y2, box.conf.item()))
        return boxes

    def _read_plate(self, crop: np.ndarray) -> str:
        rectified = _rectify(crop)
        gray      = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY) if rectified.ndim == 3 else rectified
        gray      = cv2.resize(gray, (IMG_W, IMG_H))

        tensor = (
            torch.tensor(gray, dtype=torch.float32)
            .unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            .div(255.0)
            .to(self.device)
        )
        with torch.no_grad():
            logits = self.crnn(tensor)  # (T, 1, C)
        return greedy_decode(logits)[0]
