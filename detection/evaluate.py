"""
Evaluate a trained YOLOv8 plate detector.

Usage:
    python -m detection.evaluate --weights runs/detect/gatehub/weights/best.pt
"""

import argparse
from ultralytics import YOLO


def evaluate(
    weights:   str,
    data_yaml: str = 'detection/dataset.yaml',
    imgsz:     int = 640,
    device:    str = '0',
    split:     str = 'test',
) -> dict:
    model   = YOLO(weights)
    metrics = model.val(data=data_yaml, imgsz=imgsz, device=device, split=split)

    report = {
        'mAP50':     round(metrics.box.map50, 4),
        'mAP50-95':  round(metrics.box.map,   4),
        'precision': round(metrics.box.mp,     4),
        'recall':    round(metrics.box.mr,     4),
    }

    print(f'\n{"="*40}')
    print('  Detection results')
    print(f'{"="*40}')
    for k, v in report.items():
        print(f'  {k:<14}: {v:.4f}')

    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--data',    default='detection/dataset.yaml')
    parser.add_argument('--split',   default='test', choices=['val', 'test'])
    parser.add_argument('--device',  default='0')
    args = parser.parse_args()
    evaluate(args.weights, args.data, split=args.split, device=args.device)
