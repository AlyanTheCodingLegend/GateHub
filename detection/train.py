"""
Fine-tune YOLOv8 for Pakistani license plate detection.

Usage:
    # Single run with YOLOv8n
    python -m detection.train

    # Ablation over n / s / m model sizes
    python -m detection.train --ablation

    # Custom size and epochs
    python -m detection.train --size s --epochs 50
"""

import argparse
import json
from pathlib import Path
from ultralytics import YOLO


# Augmentation parameters tuned for gate-camera conditions.
# Kept conservative: plates are small and perspective is roughly fixed.
_AUG = dict(
    degrees=10.0,        # rotation
    translate=0.1,
    scale=0.4,
    shear=3.0,
    perspective=0.0005,
    flipud=0.0,
    fliplr=0.3,
    mosaic=1.0,
    mixup=0.05,
    copy_paste=0.0,
    hsv_h=0.015,
    hsv_s=0.5,
    hsv_v=0.4,
)


def train_single(
    size: str       = 'n',
    data_yaml: str  = 'detection/dataset.yaml',
    epochs: int     = 50,
    imgsz: int      = 640,
    batch: int      = 16,
    device: str     = '0',
    name: str       = 'gatehub',
    project: str    = 'runs/detect',
) -> dict:
    """Fine-tune YOLOv8{size} and return mAP metrics."""
    model   = YOLO(f'yolov8{size}.pt')
    results = model.train(
        data    = data_yaml,
        epochs  = epochs,
        imgsz   = imgsz,
        batch   = batch,
        device  = device,
        project = project,
        name    = name,
        patience = 15,
        **_AUG,
    )
    metrics = results.results_dict
    return {
        'size':      f'yolov8{size}',
        'mAP50':     round(metrics.get('metrics/mAP50(B)',    0.0), 4),
        'mAP50-95':  round(metrics.get('metrics/mAP50-95(B)', 0.0), 4),
        'precision': round(metrics.get('metrics/precision(B)', 0.0), 4),
        'recall':    round(metrics.get('metrics/recall(B)',    0.0), 4),
    }


def run_ablation(data_yaml: str, epochs: int = 30) -> None:
    """
    Compare YOLOv8 n / s / m on the same dataset.
    Produces a JSON summary at runs/detect/ablation_summary.json.
    """
    summary = {}
    for size in ['n', 's', 'm']:
        print(f'\n{"="*50}')
        print(f'  YOLOv8{size}')
        print(f'{"="*50}')
        metrics = train_single(
            size      = size,
            data_yaml = data_yaml,
            epochs    = epochs,
            name      = f'ablation_{size}',
        )
        summary[f'yolov8{size}'] = metrics
        print(metrics)

    out = Path('runs/detect/ablation_summary.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f'\nAblation summary saved to {out}')

    # Print table
    print('\n  Model      mAP@0.5  mAP@0.5:0.95')
    for m, v in summary.items():
        print(f'  {m:<10} {v["mAP50"]:.4f}   {v["mAP50-95"]:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size',     default='n', choices=['n', 's', 'm'])
    parser.add_argument('--data',     default='detection/dataset.yaml')
    parser.add_argument('--epochs',   type=int, default=50)
    parser.add_argument('--batch',    type=int, default=16)
    parser.add_argument('--device',   default='0')
    parser.add_argument('--ablation', action='store_true',
                        help='Run n/s/m ablation (overrides --size)')
    args = parser.parse_args()

    if args.ablation:
        run_ablation(args.data, epochs=args.epochs)
    else:
        m = train_single(
            size      = args.size,
            data_yaml = args.data,
            epochs    = args.epochs,
            batch     = args.batch,
            device    = args.device,
        )
        print('\nResult:', m)
