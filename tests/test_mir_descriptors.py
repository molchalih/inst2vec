"""Unit tests for the descriptor helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def test_topk_csv_returns_labels_and_scores():
    from modules.mir.descriptors import topk_csv

    probs = np.array([0.1, 0.5, 0.4, 0.05])
    labels = ["a", "b", "c", "d"]
    out_labels, out_scores = topk_csv(probs, labels, k=2)
    assert out_labels == "b, c"
    assert out_scores == "0.50, 0.40"


def test_topk_csv_caps_at_array_length():
    from modules.mir.descriptors import topk_csv

    probs = np.array([0.9, 0.1])
    out_labels, out_scores = topk_csv(probs, ["x", "y"], k=10)
    assert out_labels == "x, y"
    assert out_scores == "0.90, 0.10"


def test_topk_csv_rejects_label_mismatch():
    from modules.mir.descriptors import topk_csv

    probs = np.array([0.5, 0.5])
    with pytest.raises(ValueError):
        topk_csv(probs, ["only_one"], k=1)


def test_binary_decide_threshold_boundary():
    from modules.mir.descriptors import binary_decide

    assert binary_decide(0.6, threshold=0.5) is True
    assert binary_decide(0.5, threshold=0.5) is True  # >= threshold
    assert binary_decide(0.49, threshold=0.5) is False


def test_load_labels_reads_committed_json(tmp_path):
    from modules.mir.descriptors import load_labels

    path = tmp_path / "labels.json"
    path.write_text(json.dumps(["a", "b", "c"]))
    assert load_labels(path) == ["a", "b", "c"]


def test_load_labels_rejects_malformed_json(tmp_path):
    from modules.mir.descriptors import load_labels

    bad = tmp_path / "bad.json"
    bad.write_text('{"not": "a list"}')
    with pytest.raises(ValueError):
        load_labels(bad)


def test_committed_label_files_load():
    from modules.mir.descriptors import load_labels

    root = Path(__file__).resolve().parent.parent / "modules" / "mir" / "labels"
    assert len(load_labels(root / "genre_discogs519.json")) == 519
    assert len(load_labels(root / "mtg_jamendo_moodtheme.json")) == 56
    assert len(load_labels(root / "mtg_jamendo_instrument.json")) == 40


def test_topk_csv_ties_break_by_input_order():
    import numpy as np

    from modules.mir.descriptors import topk_csv

    probs = np.array([0.5, 0.5, 0.5])
    out_labels, out_scores = topk_csv(probs, ["first", "second", "third"], k=2)
    assert out_labels == "first, second"
    assert out_scores == "0.50, 0.50"
