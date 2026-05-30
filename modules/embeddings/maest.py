"""MAEST raw-embedding provider (ONNX GPU path).

Runs the MAEST transformer via onnxruntime, requesting the
``StatefulPartitionedCall:7`` token output (the 7th block's
``[n_patches, n_tokens, 768]`` tokens: token 0 = CLS, token 1 = DIST,
2: = per-patch signal). Aggregates each patch as
``concat(CLS, DIST, mean(signal_tokens))`` -> ``(2304,)`` and mean-pools
across patches.

Audio is decoded with Essentia's ``MonoLoader`` at 16 kHz (the only remaining
Essentia use here, for resample parity with the MIR audio path). The mel +
transformer run through ``core.vendor.maest.MaestTokens`` on the GPU. This
replaces the previous Essentia ``TensorflowPredictMAEST`` (.pb) backend; the
2304-d output vector is numerically equivalent.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# Silence TF stderr in case essentia transitively imports it during decode.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def _aggregate_patches(out: np.ndarray) -> np.ndarray:
    """Reduce ``[n_patches, n_tokens, 768]`` -> ``(2304,)``.

    Per patch: ``concat(CLS=token0, DIST=token1, mean(signal tokens 2:))``;
    then mean across patches. The ONNX :7 output is 3-D ``[N, T, 768]``; the
    old Essentia wrapper returned 4-D ``[N, 1, T, 768]`` — the squeezed middle
    axis is the only difference, the reduction is otherwise identical.
    """
    cls_ = out[:, 0, :]
    dist = out[:, 1, :]
    sig = out[:, 2:, :].mean(axis=1)
    per_patch = np.concatenate([cls_, dist, sig], axis=1)
    return per_patch.mean(axis=0).astype(np.float32)


class MaestProvider:
    """``Provider`` for the ``maest`` embedding case (ONNX backend)."""

    def __init__(
        self,
        *,
        onnx_path: Path | str,
        sample_rate: int,
        patch_frames: int = 1876,
        patch_hop: int = 1875,
    ) -> None:
        from core.vendor.maest import MaestTokens

        if sample_rate <= 0:
            raise ValueError("sample_rate must be > 0")
        self._sample_rate = int(sample_rate)
        self._model = MaestTokens(
            Path(onnx_path), patch_frames=patch_frames, patch_hop=patch_hop
        )

    def embed(self, payload: dict):
        import essentia
        from essentia.standard import MonoLoader  # ty: ignore[unresolved-import]

        essentia.log.warningActive = False
        essentia.log.infoActive = False
        audio = MonoLoader(
            filename=str(payload["audio_path"]),
            sampleRate=self._sample_rate,
            resampleQuality=4,
        )()
        # MaestTokens tiles short clips up to the 30 s window internally.
        out = np.asarray(self._model.tokens(audio))
        return [_aggregate_patches(out)]
