"""
CRNN for Pakistani license plate OCR.

Architecture (Shi et al., 2015 — adapted):
  CNN backbone  →  sequence of feature vectors  →  BiLSTM × 2  →  CTC output

Input : (B, 1, 32, 128)  — grayscale plate crop, height=32, width=128
Output: (T, B, NUM_CLASSES)  — log-softmax over character vocab, T=31
"""

import torch
import torch.nn as nn

# Pakistani plates use uppercase Latin chars + digits + dash
CHARSET = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-'
BLANK_IDX = len(CHARSET)       # index 37 is the CTC blank token
NUM_CLASSES = len(CHARSET) + 1  # 38

CHAR2IDX: dict[str, int] = {c: i for i, c in enumerate(CHARSET)}
IDX2CHAR: dict[int, str] = {i: c for i, c in enumerate(CHARSET)}

IMG_H, IMG_W = 32, 128
SEQ_LEN = 31   # CNN output width for a 128-wide input (derived below)


class _BidirectionalLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=False)
        self.proj = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)   # (T, B, 2*H)
        return self.proj(out)   # (T, B, output_size)


class CRNN(nn.Module):
    """
    CNN + BiLSTM + CTC decoder for license plate text recognition.

    CNN design goal: reduce the spatial height from 32 → 1 while keeping
    the width axis as a sequence dimension that the RNN can read left-to-right.

    Dimension trace (B=batch, H=height, W=width):
      Input          (B,   1, 32, 128)
      after block 1  (B,  64, 16,  64)   MaxPool(2,2)
      after block 2  (B, 128,  8,  32)   MaxPool(2,2)
      after block 3  (B, 256,  8,  32)   no pool
      after block 4  (B, 256,  4,  32)   MaxPool(2,1)  — halve H only
      after block 5  (B, 512,  4,  32)   no pool
      after block 6  (B, 512,  2,  32)   MaxPool(2,1)  — halve H only
      after block 7  (B, 512,  1,  31)   Conv(2,1,0)   — H→1, W→31
    Sequence length T = 31 >> 2*9-1 = 17, so CTC works for plates up to 9 chars.
    """

    def __init__(self, hidden_size: int = 256):
        super().__init__()

        def _conv_bn_relu(cin, cout, k=3, s=1, p=1, bn=True):
            layers = [nn.Conv2d(cin, cout, k, s, p)]
            if bn:
                layers.append(nn.BatchNorm2d(cout))
            layers.append(nn.ReLU(inplace=True))
            return layers

        self.cnn = nn.Sequential(
            # Block 1 — (B, 1, 32, 128) → (B, 64, 16, 64)
            *_conv_bn_relu(1, 64, bn=False),
            nn.MaxPool2d(2, 2),

            # Block 2 — (B, 64, 16, 64) → (B, 128, 8, 32)
            *_conv_bn_relu(64, 128, bn=False),
            nn.MaxPool2d(2, 2),

            # Block 3 — (B, 128, 8, 32) → (B, 256, 8, 32)
            *_conv_bn_relu(128, 256),

            # Block 4 — (B, 256, 8, 32) → (B, 256, 4, 32)
            *_conv_bn_relu(256, 256, bn=False),
            nn.MaxPool2d((2, 1), (2, 1)),

            # Block 5 — (B, 256, 4, 32) → (B, 512, 4, 32)
            *_conv_bn_relu(256, 512),

            # Block 6 — (B, 512, 4, 32) → (B, 512, 2, 32)
            *_conv_bn_relu(512, 512, bn=False),
            nn.MaxPool2d((2, 1), (2, 1)),

            # Block 7 — (B, 512, 2, 32) → (B, 512, 1, 31)
            # kernel (2,2), stride 1, no padding: H=(2-2)+1=1, W=(32-2)+1=31
            *_conv_bn_relu(512, 512, k=2, s=1, p=0),
        )

        self.rnn = nn.Sequential(
            _BidirectionalLSTM(512, hidden_size, hidden_size),
            _BidirectionalLSTM(hidden_size, hidden_size, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x)                 # (B, 512, 1, T)
        features = features.squeeze(2)         # (B, 512, T)
        features = features.permute(2, 0, 1)  # (T, B, 512)  — seq-first for LSTM
        logits = self.rnn(features)            # (T, B, NUM_CLASSES)
        return logits.log_softmax(2)
