"""
Evaluate CRNN accuracy on a held-out test set and compare against EasyOCR.

Usage:
    # CRNN only (greedy)
    python -m ocr.evaluate --data data/ocr/test --ckpt checkpoints/crnn/best.pt

    # CRNN + EasyOCR baseline comparison
    python -m ocr.evaluate --data data/ocr/test --ckpt checkpoints/crnn/best.pt --compare-easyocr

    # Use beam search instead of greedy
    python -m ocr.evaluate --data data/ocr/test --ckpt checkpoints/crnn/best.pt --beam
"""

import argparse
import json
from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ocr.model import CRNN
from ocr.dataset import PlateDataset, collate_fn, clean_label
from ocr.decode import greedy_decode, beam_search_decode


def word_accuracy(preds: list[str], labels: list[str]) -> float:
    return sum(p == l for p, l in zip(preds, labels)) / len(labels) if labels else 0.0


def char_accuracy(preds: list[str], labels: list[str]) -> float:
    """Per-character accuracy (edit-distance denominator = max length)."""
    correct = total = 0
    for p, l in zip(preds, labels):
        for pc, lc in zip(p, l):
            correct += int(pc == lc)
        total += max(len(p), len(l))
    return correct / total if total > 0 else 0.0


def load_model(ckpt_path: str, device: torch.device) -> CRNN:
    model = CRNN()
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device).eval()
    return model


def run_crnn(
    model: CRNN,
    loader: DataLoader,
    device: torch.device,
    use_beam: bool = False,
) -> tuple[list[str], list[str]]:
    all_preds: list[str] = []
    all_labels: list[str] = []

    with torch.no_grad():
        for imgs, _, _, raw_labels in tqdm(loader, desc='CRNN'):
            logits = model(imgs.to(device))
            preds  = beam_search_decode(logits) if use_beam else greedy_decode(logits)
            all_preds.extend(preds)
            all_labels.extend(raw_labels)

    return all_preds, all_labels


def run_easyocr(data_root: str) -> tuple[list[str], list[str]]:
    import easyocr
    use_gpu = torch.cuda.is_available()
    reader  = easyocr.Reader(['en'], gpu=use_gpu)

    root   = Path(data_root)
    preds: list[str]  = []
    labels: list[str] = []

    img_paths = sorted((root / 'images').glob('*.png'))
    for img_path in tqdm(img_paths, desc='EasyOCR'):
        lbl_path = root / 'labels' / (img_path.stem + '.txt')
        if not lbl_path.exists():
            continue
        label = clean_label(lbl_path.read_text().strip())
        if len(label) < 2:
            continue

        img    = cv2.imread(str(img_path))
        result = reader.readtext(img, detail=0)
        pred   = clean_label(' '.join(result))

        preds.append(pred)
        labels.append(label)

    return preds, labels


def print_report(name: str, preds: list[str], labels: list[str]) -> dict:
    wa = word_accuracy(preds, labels)
    ca = char_accuracy(preds, labels)
    print(f'\n{"="*40}')
    print(f'  {name}')
    print(f'{"="*40}')
    print(f'  Samples        : {len(labels)}')
    print(f'  Word accuracy  : {wa:.4f}  ({wa*100:.1f}%)')
    print(f'  Char accuracy  : {ca:.4f}  ({ca*100:.1f}%)')

    # Show a handful of mistakes
    mistakes = [(p, l) for p, l in zip(preds, labels) if p != l][:10]
    if mistakes:
        print('\n  Sample errors (pred → truth):')
        for p, l in mistakes:
            print(f'    {p!r:20s} → {l!r}')

    return {'word_acc': wa, 'char_acc': ca, 'n': len(labels)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',            required=True, help='Test data root')
    parser.add_argument('--ckpt',            required=True, help='CRNN checkpoint (.pt)')
    parser.add_argument('--beam',            action='store_true')
    parser.add_argument('--compare-easyocr', action='store_true')
    parser.add_argument('--out',             default=None,  help='Save JSON report to path')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = load_model(args.ckpt, device)

    dataset = PlateDataset(args.data, augment=False)
    loader  = DataLoader(dataset, batch_size=32, collate_fn=collate_fn, num_workers=2)

    decode_name = 'CRNN — beam search' if args.beam else 'CRNN — greedy'
    crnn_preds, labels = run_crnn(model, loader, device, use_beam=args.beam)
    crnn_report = print_report(decode_name, crnn_preds, labels)

    results = {'crnn': crnn_report}

    if args.compare_easyocr:
        easy_preds, easy_labels = run_easyocr(args.data)
        n = min(len(easy_preds), len(labels))
        easy_report = print_report('EasyOCR baseline', easy_preds[:n], easy_labels[:n])
        results['easyocr'] = easy_report

        print('\n  CRNN improvement over EasyOCR:')
        delta_w = crnn_report['word_acc'] - easy_report['word_acc']
        delta_c = crnn_report['char_acc'] - easy_report['char_acc']
        print(f'    Word acc delta : {delta_w:+.4f}')
        print(f'    Char acc delta : {delta_c:+.4f}')
        results['delta'] = {'word_acc': delta_w, 'char_acc': delta_c}

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f'\nReport saved to {args.out}')


if __name__ == '__main__':
    main()
