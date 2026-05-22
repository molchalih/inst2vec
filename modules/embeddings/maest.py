"""MAEST raw-embedding provider.

Loads the same ``.pb`` checkpoint the MIR stage owns but requests the
``PartitionedCall/Identity_7`` tensor (CLS + DIST + per-patch signal
tokens, shape ``[N_patches, 1, T_tokens, 768]``). Aggregates each patch
as ``concat(CLS, DIST, mean(signal_tokens))`` → ``(2304,)`` and mean-pools
across patches.

Essentia is imported lazily so the module is importable in tests that
stub these symbols.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# Silence TF stderr before essentia transitively imports it.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def _tile_to_length(audio: np.ndarray, n: int) -> np.ndarray:
    """Loop ``audio`` to fill ``n`` samples (zeros for empty input)."""
    if audio.size == 0:
        return np.zeros(n, dtype=audio.dtype)
    reps = -(-n // audio.size)
    return np.tile(audio, reps)[:n].astype(audio.dtype, copy=False)


def _aggregate_patches(out: np.ndarray) -> np.ndarray:
    """Reduce ``[N_patches, 1, T_tokens, 768]`` → ``(2304,)``.

    Per patch: concat(CLS, DIST, mean(signal tokens 2:)). Then mean across
    patches.
    """
    cls_ = out[:, 0, 0, :]
    dist = out[:, 0, 1, :]
    sig = out[:, 0, 2:, :].mean(axis=1)
    per_patch = np.concatenate([cls_, dist, sig], axis=1)
    return per_patch.mean(axis=0).astype(np.float32)


class MaestProvider:
    """``Provider`` for the ``maest`` embedding case."""

    def __init__(
        self,
        *,
        checkpoint_path: Path | str,
        input_op: str,
        sample_rate: int,
        min_samples: int,
    ) -> None:
        import essentia
        from essentia.standard import (
            TensorflowPredictMAEST,  # ty: ignore[unresolved-import]
        )

        essentia.log.warningActive = False
        essentia.log.infoActive = False

        if min_samples <= 0:
            raise ValueError("min_samples must be > 0")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be > 0")
        self._sample_rate = int(sample_rate)
        self._min_samples = int(min_samples)
        self._predict = TensorflowPredictMAEST(
            graphFilename=str(checkpoint_path),
            input=input_op,
            output="PartitionedCall/Identity_7",
        )

    def embed(self, payload: dict):
        from essentia.standard import MonoLoader  # ty: ignore[unresolved-import]

        audio_path = payload["audio_path"]
        audio = MonoLoader(
            filename=str(audio_path),
            sampleRate=self._sample_rate,
            resampleQuality=4,
        )()
        if audio.size < self._min_samples:
            audio = _tile_to_length(audio, self._min_samples)
        out = np.asarray(self._predict(audio))
        return [_aggregate_patches(out)]
