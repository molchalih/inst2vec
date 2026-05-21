"""Pure helpers for shaping descriptor outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def topk_csv(probs: np.ndarray, labels: list[str], k: int) -> tuple[str, str]:
    """Return ``(labels_csv, scores_csv)`` for the top-K entries in ``probs``.

    Labels are joined by ``", "``. Scores are formatted to 4 decimals.
    Discogs-style ``Parent---Child`` labels are flattened to ``Parent Child``.
    """
    if len(probs) != len(labels):
        raise ValueError(f"probs length {len(probs)} != labels length {len(labels)}")
    n = min(k, len(probs))
    order = np.argsort(-np.asarray(probs), kind="stable")[:n]
    label_parts = [labels[i].replace("---", " ") for i in order]
    score_parts = [f"{float(probs[i]):.4f}" for i in order]
    return ", ".join(label_parts), ", ".join(score_parts)


def load_labels(path: Path | str) -> list[str]:
    """Load a JSON array of strings.

    Raises ``ValueError`` if the payload is not a list of strings.
    """
    with open(path) as f:
        payload = json.load(f)
    if not isinstance(payload, list) or not all(isinstance(x, str) for x in payload):
        raise ValueError(f"{path}: expected JSON array of strings")
    return payload
