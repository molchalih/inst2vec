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

import os
from pathlib import Path

import numpy as np

# Silence TensorFlow's C++ stderr (CUDA dso lookups, GPU init, oneDNN notice)
# before essentia.standard transitively imports TF. setdefault preserves any
# user override; level "3" suppresses INFO + WARNING + ERROR.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def _tile_to_length(audio: np.ndarray, n: int) -> np.ndarray:
    """Loop ``audio`` to fill ``n`` samples (zeros for empty input)."""
    if audio.size == 0:
        return np.zeros(n, dtype=audio.dtype)
    reps = -(-n // audio.size)
    return np.tile(audio, reps)[:n].astype(audio.dtype, copy=False)


class MAEST:
    def __init__(self, pb: Path, *, input: str, output: str, min_samples: int):
        # Flip Essentia's log flags BEFORE importing from essentia.standard:
        # algorithm registration during that import emits the one-shot
        # "[ INFO ] MusicExtractorSVM: no classifier models were configured by
        # default" line, which the post-import suppression in older revisions
        # could not catch.
        import essentia

        essentia.log.warningActive = False  # ty: ignore[unresolved-attribute]
        essentia.log.infoActive = False  # ty: ignore[unresolved-attribute]
        from essentia.standard import (
            TensorflowPredictMAEST,  # ty: ignore[unresolved-import]
        )

        if min_samples <= 0:
            raise ValueError("min_samples must be > 0")
        self._min_samples = int(min_samples)
        self._predict = TensorflowPredictMAEST(
            graphFilename=str(pb), input=input, output=output,
        )

    def predict(self, audio: np.ndarray) -> np.ndarray:
        """Average per-window genre519 predictions over time → (519,).

        Audio shorter than one patch (``min_samples``) is tiled to the patch
        length so MAEST always sees at least one full window. SavedModel-style
        exports return ``(N, 1, 1, K)``; original frozen graphs returned
        ``(N, K)``. Flatten leading dims first so either layout collapses to
        ``(K,)``.
        """
        if audio.size < self._min_samples:
            audio = _tile_to_length(audio, self._min_samples)
        out = self._predict(audio)
        return out.reshape(-1, out.shape[-1]).mean(axis=0)

    def __enter__(self) -> MAEST:
        return self

    def __exit__(self, *exc) -> None:
        return None
