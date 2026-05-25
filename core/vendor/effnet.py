"""EffNet-Discogs ONNX wrapper: 1280-d embedding (GPU) + Essentia heads (CPU).

The EfficientNet backbone runs through onnxruntime; the 15 tiny classification
heads stay as Essentia TensorflowPredict2D graphs on CPU because they operate
on a single 1280-d embedding and are microsecond-cheap. Public API matches the
previous Essentia wrapper.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from core.vendor.mel import frame_patches, musicnn_mel
from core.vendor.onnx_session import make_session, pick_output_by_lastdim

_EMBED_DIM = 1280


def _build_essentia_heads(heads: dict[str, tuple[Path, str]]):
    """Construct one Essentia TensorflowPredict2D callable per head."""
    import essentia

    essentia.log.warningActive = False  # ty: ignore[unresolved-attribute]
    essentia.log.infoActive = False  # ty: ignore[unresolved-attribute]
    from essentia.standard import TensorflowPredict2D  # ty: ignore[unresolved-import]

    return {
        name: TensorflowPredict2D(graphFilename=str(pb), output=out)
        for name, (pb, out) in heads.items()
    }


class EffNet:
    def __init__(
        self,
        embed_pb: Path,
        heads: dict[str, tuple[Path, str]],
        *,
        patch_frames: int = 128,
        patch_hop: int = 62,
        embed_output: str | None = None,  # legacy kwarg, ignored
        **_ignored,
    ):
        self._patch_frames = int(patch_frames)
        # Essentia's TensorflowPredictEffnetDiscogs default patchHopSize is 62
        # (overlapping patches, ~one prediction per second), NOT patchSize. The
        # mean embedding only matches the .pb when this overlap is reproduced.
        self._patch_hop = int(patch_hop)
        self._session = make_session(Path(embed_pb))
        self._input = self._session.get_inputs()[0].name
        self._output = pick_output_by_lastdim(self._session.get_outputs(), _EMBED_DIM)
        self._heads = _build_essentia_heads(heads)

    def embed(self, audio: np.ndarray) -> np.ndarray:
        mel = musicnn_mel(audio)
        patches = frame_patches(
            mel, patch_size=self._patch_frames, hop_size=self._patch_hop
        )
        return self._session.run([self._output], {self._input: patches})[0]

    def predict_all(self, embedding: np.ndarray) -> dict[str, np.ndarray]:
        """Return ``{head_name: per-window-mean prediction vector}``."""
        return {n: np.asarray(h(embedding)).mean(axis=0) for n, h in self._heads.items()}

    def __enter__(self) -> "EffNet":
        return self

    def __exit__(self, *exc) -> None:
        return None
