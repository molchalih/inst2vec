"""TF MAEST wrapper: 30 s window → direct 519-class predictions.

Loads one TensorflowPredictMAEST graph at construction (essentia is
imported lazily inside ``__init__`` so the module can be imported in
tests that stub this class). The chosen checkpoint is the
``discogs-maest-30s-pw-519l-1`` variant which outputs sigmoid
predictions for 519 Discogs genres directly at
``StatefulPartitionedCall:0`` — no separate classification head is
required.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class MAEST:
    def __init__(self, pb: Path, *, output: str):
        from essentia.standard import TensorflowPredictMAEST

        self._predict = TensorflowPredictMAEST(
            graphFilename=str(pb), output=output,
        )

    def predict(self, audio: np.ndarray) -> np.ndarray:
        """Average per-window genre519 predictions over time → (519,)."""
        return self._predict(audio).mean(axis=0)

    def __enter__(self) -> "MAEST":
        return self

    def __exit__(self, *exc) -> None:
        return None
