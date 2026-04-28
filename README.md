# GateHub — Intelligent Vehicle Access Management System

**Deep Learning End-Semester Project**
National University of Sciences and Technology (NUST) — SEECS

| Name | CMS ID |
|---|---|
| Abdullah Farooq | 462905 |
| Sohaib Amir Bukhari | 482061 |
| Huzaifa Ali Satti | 468629 |
| Alyan Ahmed Memon | 469355 |

---

## What is GateHub?

NUST currently manages gate access by collecting a visitor's CNIC at entry and returning it only at the same gate on exit. A driver who enters Gate 1 and finishes near Gate 2 must drive back across campus just to retrieve their ID.

GateHub replaces this with a camera-based pipeline:

1. Vehicle arrives at any gate — camera reads the plate.
2. Guard scans the CNIC — system logs the session and hands the card back immediately.
3. Vehicle exits from **any** gate — plate is scanned, session closes, driver is free to go.

No card held. No mandatory return trip.

---

## Deep Learning Components

| Component | What we built |
|---|---|
| **Plate Detection** | YOLOv8 fine-tuned on Pakistani plate datasets (DVLPD + Roboflow). Ablation over n/s/m model sizes. |
| **Plate OCR** | Custom CRNN (CNN backbone + 2× BiLSTM + CTC loss) trained from scratch on Pakistani plate crops. Compared against EasyOCR as a baseline. |
| **Perspective Correction** | Contour-based quadrilateral detection + `cv2.warpPerspective` applied before OCR, following the detect-then-rectify approach from Yousaf et al. (2021). |
| **CNIC OCR** | PaddleOCR with CLAHE preprocessing for 13-digit CNIC extraction. |
| **Dashboard** | Streamlit UI — live sessions, overstay alerts, blacklist management. |
| **Session DB** | SQLite — entry/exit logged with gate, plate, CNIC, and timestamp. |

### CRNN Architecture

```
Input (B, 1, 32, 128)
  │
  ├── CNN Backbone (7 blocks)
  │     Block 1: Conv(64)  + MaxPool(2,2)   → (B, 64,  16, 64)
  │     Block 2: Conv(128) + MaxPool(2,2)   → (B, 128,  8, 32)
  │     Block 3: Conv(256) + BN             → (B, 256,  8, 32)
  │     Block 4: Conv(256) + MaxPool(2,1)   → (B, 256,  4, 32)
  │     Block 5: Conv(512) + BN             → (B, 512,  4, 32)
  │     Block 6: Conv(512) + MaxPool(2,1)   → (B, 512,  2, 32)
  │     Block 7: Conv(512) + BN             → (B, 512,  1, 31)
  │
  ├── Reshape → (31, B, 512)   [sequence-first for LSTM]
  │
  ├── BiLSTM(512 → 256)
  ├── BiLSTM(256 → 256)
  │
  └── Linear + log_softmax → (31, B, 38)   [38 = 37 chars + CTC blank]

Loss: CTCLoss   Decoder: greedy (inference) / beam search (evaluation)
```

---

## Project Structure

```
dl-project/
├── data/
│   └── prepare.py          # download DVLPD + Roboflow, build OCR dataset
├── detection/
│   ├── dataset.yaml        # YOLOv8 dataset config
│   ├── train.py            # fine-tuning + n/s/m ablation
│   └── evaluate.py         # mAP reporting
├── ocr/
│   ├── model.py            # CRNN architecture
│   ├── dataset.py          # plate crop loader + augmentation
│   ├── train.py            # CTC training loop
│   ├── decode.py           # greedy + beam search decoder
│   └── evaluate.py         # accuracy metrics + EasyOCR comparison
├── cnic/
│   └── reader.py           # PaddleOCR + CLAHE preprocessing
├── pipeline/
│   └── inference.py        # end-to-end: detect → rectify → OCR
├── database/
│   ├── schema.sql          # SQLite schema
│   └── session.py          # entry/exit session logic
├── dashboard/
│   └── app.py              # Streamlit guard dashboard
└── requirements.txt
```

---

## Setup

### 1. Install PyTorch (RTX 5060 / Blackwell requires CUDA 12.8+)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

> If you are on an older GPU (RTX 30/40 series), use `cu121` instead of `cu128`.

### 2. Install remaining dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

Run all commands from the **project root** (`dl-project/`).

---

### Step 1 — Prepare datasets

Downloads DVLPD from GitHub, extracts plate crops with reconstructed plate strings (OCR training set), and copies plate bounding-box annotations (detection training set). Also downloads Roboflow PK-plates datasets if `ROBOFLOW_API_KEY` is set.

```bash
python -m data.prepare
```

**After this completes**, open `data/dvlpd_raw/classes.txt` and verify the class ID order. The defaults in `data/prepare.py` expect:

```
class 0 = vehicle
class 1 = plate
class 2–11 = digits 0–9
class 12–37 = letters A–Z
```

If the order differs, update `PLATE_CLASS`, `DIGIT_START`, and `ALPHA_START` at the top of `data/prepare.py` and re-run.

**Estimated time:** 20–40 min (mostly download + disk I/O)

---

### Step 2 — Train the plate detector (ablation)

Runs YOLOv8n, YOLOv8s, and YOLOv8m back-to-back on the same dataset and saves a comparison table to `runs/detect/ablation_summary.json`. Use the results to justify your final model choice in the report.

```bash
python -m detection.train --ablation --epochs 30
```

If you just want the final model without the ablation:

```bash
python -m detection.train --size n --epochs 50
```

**Estimated time:** 90–120 min (ablation) / 30–45 min (single run)

---

### Step 3 — Evaluate the detector

```bash
python -m detection.evaluate --weights runs/detect/gatehub/weights/best.pt
```

To evaluate against the test split specifically:

```bash
python -m detection.evaluate --weights runs/detect/gatehub/weights/best.pt --split test
```

---

### Step 4 — Train the CRNN OCR model

```bash
python -m ocr.train --data data/ocr --epochs 30
```

Checkpoints are saved to `checkpoints/crnn/`. Training history (loss + val accuracy per epoch) is saved as `checkpoints/crnn/history.json`.

Optional flags:

```bash
python -m ocr.train --data data/ocr --epochs 30 --batch 64 --lr 1e-3
```

**Estimated time:** 20–35 min

---

### Step 5 — Evaluate CRNN vs EasyOCR

Runs the CRNN on the held-out test set and compares word accuracy and character accuracy against EasyOCR.

```bash
python -m ocr.evaluate \
    --data data/ocr \
    --ckpt checkpoints/crnn/best.pt \
    --beam \
    --compare-easyocr \
    --out results/ocr_comparison.json
```

Without the EasyOCR comparison (faster):

```bash
python -m ocr.evaluate --data data/ocr --ckpt checkpoints/crnn/best.pt --beam
```

**Estimated time:** 25–40 min (EasyOCR is the slow part)

---

### Step 6 — Launch the dashboard

```bash
streamlit run dashboard/app.py
```

Opens at `http://localhost:8501`. Three tabs:

- **Live Sessions** — active vehicles, elapsed time, overstay and blacklist alerts
- **Entry / Exit** — manual logging with gate selection
- **Blacklist** — add/remove plates, with reason

---

## Evaluation Targets

| Metric | Target | How measured |
|---|---|---|
| Plate detection mAP@0.5 | >= 85% | `detection/evaluate.py` on test split |
| End-to-end plate accuracy | >= 75% | `ocr/evaluate.py` word accuracy |
| CNIC extraction accuracy | >= 90% | Manual test with sample cards |
| Inference latency | <= 500 ms/frame | Timed in `pipeline/inference.py` |

---

## References

1. Salma et al. (2021). *Development of ANPR Framework for Pakistani Vehicle Number Plates.* Complexity. https://doi.org/10.1155/2021/5597337
2. Yousaf et al. (2021). *A Deep Learning Based Approach for Localization and Recognition of Pakistani Vehicle License Plates.* Sensors, 21(22). https://doi.org/10.3390/s21227696
3. Usama et al. (2022). *Vehicle and License Plate Recognition with Novel Dataset for Toll Collection.* arXiv:2202.05631. https://arxiv.org/abs/2202.05631
4. Shi et al. (2015). *An End-to-End Trainable Neural Network for Image-based Sequence Recognition.* arXiv:1507.05717.
