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
    assert out_scores == "0.5000, 0.4000"


def test_topk_csv_caps_at_array_length():
    from modules.mir.descriptors import topk_csv

    probs = np.array([0.9, 0.1])
    out_labels, out_scores = topk_csv(probs, ["x", "y"], k=10)
    assert out_labels == "x, y"
    assert out_scores == "0.9000, 0.1000"


def test_topk_csv_rejects_label_mismatch():
    from modules.mir.descriptors import topk_csv

    probs = np.array([0.5, 0.5])
    with pytest.raises(ValueError):
        topk_csv(probs, ["only_one"], k=1)


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
    assert out_scores == "0.5000, 0.5000"


def test_topk_csv_flattens_discogs_parent_child_separator():
    """Discogs labels use ``Parent---Child``; we store them as ``Parent Child``."""
    from modules.mir.descriptors import topk_csv

    probs = np.array([0.6, 0.3, 0.1])
    labels = ["Electronic---Ambient", "Electronic---Downtempo", "Rock"]
    out_labels, _ = topk_csv(probs, labels, k=3)
    assert out_labels == "Electronic Ambient, Electronic Downtempo, Rock"


def test_topk_csv_emits_four_decimal_places():
    """Sigmoid heads need 4dp to keep human-visible ranks unambiguous."""
    import numpy as np

    from modules.mir.descriptors import topk_csv

    probs = np.array([0.12345, 0.23456, 0.34567])
    _labels, scores = topk_csv(probs, ["a", "b", "c"], k=3)
    # Expect 4-decimal formatting, e.g. "0.3457, 0.2346, 0.1235"
    for part in scores.split(", "):
        _whole, frac = part.split(".")
        assert len(frac) == 4, f"expected 4-decimal score, got {part!r}"
