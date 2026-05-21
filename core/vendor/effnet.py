"""TF EffNet-Discogs wrapper: 1280-d embedding + per-head TensorflowPredict2D.

Loads one EffNet graph + one TensorflowPredict2D per head. ``heads``
maps a head name to ``(graph_path, output_op)`` — the output op varies
by head (``model/Softmax`` for binary classifiers, ``model/Sigmoid``
for multi-tag classifiers, ``model/Identity`` for regression heads),
sourced from each model's verified ``metadata.json``.

Essentia is imported lazily so this module is importable in tests that
stub the class.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class EffNet:
    def __init__(
        self,
        embed_pb: Path,
        heads: dict[str, tuple[Path, str]],
        *,
        embed_output: str = "PartitionedCall:1",
    ):
        from essentia.standard import (
            TensorflowPredict2D,
            TensorflowPredictEffnetDiscogs,
        )

        self._embed = TensorflowPredictEffnetDiscogs(
            graphFilename=str(embed_pb), output=embed_output,
        )
        self._heads = {
            name: TensorflowPredict2D(graphFilename=str(pb), output=out)
            for name, (pb, out) in heads.items()
        }

    def embed(self, audio: np.ndarray) -> np.ndarray:
        return self._embed(audio)

    def predict_all(self, embedding: np.ndarray) -> dict[str, np.ndarray]:
        """Return ``{head_name: per-window-mean prediction vector}``."""
        return {n: h(embedding).mean(axis=0) for n, h in self._heads.items()}

    def __enter__(self) -> "EffNet":
        return self

    def __exit__(self, *exc) -> None:
        return None
