"""Data helpers for guppyemail ChatML training/evaluation files."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


ASSISTANT_MARKER = "<|im_start|>assistant\n"
USER_MARKER = "<|im_start|>user\n"
END_MARKER = "<|im_end|>"


def parse_chatml(text: str) -> tuple[str, str]:
    """Return email text and reference summary from one ChatML row."""
    if USER_MARKER not in text or ASSISTANT_MARKER not in text:
        raise ValueError("row is missing ChatML user/assistant markers")

    email = text.split(USER_MARKER, 1)[1].split(END_MARKER, 1)[0].strip()
    summary = text.split(ASSISTANT_MARKER, 1)[1].split(END_MARKER, 1)[0].strip()
    return email, summary


def load_chatml_rows(path: str | Path) -> list[dict]:
    """Load chatml rows."""
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            text = row.get("text", "")
            try:
                email, reference = parse_chatml(text)
            except ValueError:
                continue
            rows.append({**row, "email": email, "reference": reference})
    return rows


class EmailSummaryDataset(Dataset):
    """Dataset with labels masked so loss is computed only on assistant text."""

    def __init__(self, path: str | Path, tokenizer, max_len: int = 512):
        """Initialize the instance."""
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples: list[tuple[list[int], list[int]]] = []
        self.skipped = 0

        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                text = row.get("text", "")
                if ASSISTANT_MARKER not in text:
                    self.skipped += 1
                    continue

                ids = tokenizer.encode(text).ids[:max_len]
                prefix = text.split(ASSISTANT_MARKER, 1)[0] + ASSISTANT_MARKER
                prefix_len = len(tokenizer.encode(prefix).ids)

                labels = ids.copy()
                masked_len = min(prefix_len, len(labels))
                labels[:masked_len] = [-100] * masked_len
                if len(ids) < 2 or all(value == -100 for value in labels[1:]):
                    self.skipped += 1
                    continue

                self.samples.append((ids, labels))

    def __len__(self) -> int:
        """Return the number of available samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one indexed sample."""
        ids, labels = self.samples[idx]
        return (
            torch.tensor(ids[:-1], dtype=torch.long),
            torch.tensor(labels[1:], dtype=torch.long),
        )


def collate_batch(batch, pad_id: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Handle collate batch."""
    xs, ys = zip(*batch)
    max_len = max(x.numel() for x in xs)
    x_pad = torch.full((len(xs), max_len), pad_id, dtype=torch.long)
    y_pad = torch.full((len(ys), max_len), -100, dtype=torch.long)
    for i, (x, y) in enumerate(zip(xs, ys)):
        x_pad[i, : x.numel()] = x
        y_pad[i, : y.numel()] = y
    return x_pad, y_pad
