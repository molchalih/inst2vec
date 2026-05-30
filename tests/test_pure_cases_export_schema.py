"""The export manifest case Literal must match the reworked case set."""

from __future__ import annotations

from typing import get_args

from modules.export.schemas import EmbeddingCase


def test_embedding_case_literal_args():
    args = set(get_args(EmbeddingCase))
    assert {"video", "sandwich", "auditory", "spoken", "textual"} <= args
    assert "audio" not in args
    assert "maest" not in args
