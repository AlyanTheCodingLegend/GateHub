"""
CTC decoding — greedy and beam search.
"""

import torch
from ocr.model import IDX2CHAR, BLANK_IDX


def greedy_decode(logits: torch.Tensor) -> list[str]:
    """
    CTC greedy decode: argmax at each timestep, collapse consecutive
    repeats, then strip blank tokens.

    logits: (T, B, C)  — log-softmax outputs from CRNN.forward()
    Returns a list of decoded strings, one per batch element.
    """
    indices = logits.argmax(dim=2).permute(1, 0)  # (B, T)
    results = []

    for seq in indices.cpu().tolist():
        chars = []
        prev = BLANK_IDX
        for idx in seq:
            if idx != BLANK_IDX and idx != prev:
                chars.append(IDX2CHAR.get(idx, ''))
            prev = idx
        results.append(''.join(chars))

    return results


def beam_search_decode(logits: torch.Tensor, beam_width: int = 5) -> list[str]:
    """
    CTC beam search decode — higher accuracy than greedy, used for
    evaluation and ablation reporting.

    logits: (T, B, C)
    """
    log_probs = logits.detach().cpu()  # (T, B, C)
    return [_beam_single(log_probs[:, b, :], beam_width)
            for b in range(log_probs.shape[1])]


def _beam_single(log_probs: torch.Tensor, beam_width: int) -> str:
    """
    Beam search for one sequence.
    log_probs: (T, C)

    State: dict mapping (text, last_nonblank_idx) → cumulative log-prob.
    """
    T, C = log_probs.shape

    # beam key: (decoded_text, last_emitted_char_idx)
    # last_emitted_char_idx lets us correctly handle blank-separated repeats
    beams: dict[tuple[str, int], float] = {('', BLANK_IDX): 0.0}

    for t in range(T):
        step = log_probs[t]  # (C,)
        new_beams: dict[tuple[str, int], float] = {}

        for (text, last_idx), score in beams.items():
            for c in range(C):
                lp = step[c].item()
                if c == BLANK_IDX:
                    key = (text, BLANK_IDX)
                elif c == last_idx:
                    # Same char as last non-blank — only extend if separated by blank
                    key = (text, c)
                else:
                    key = (text + IDX2CHAR.get(c, ''), c)

                new_score = score + lp
                if key not in new_beams or new_beams[key] < new_score:
                    new_beams[key] = new_score

        beams = dict(
            sorted(new_beams.items(), key=lambda x: x[1], reverse=True)[:beam_width]
        )

    best_text = max(beams.items(), key=lambda x: x[1])[0][0]
    return best_text
