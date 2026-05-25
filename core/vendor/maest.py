"""MAEST ONNX wrapper: 30 s mel patch -> 519-class genre predictions.

Mel extraction stays in Essentia (CPU, cheap); the heavy transformer graph
runs through onnxruntime (GPU when available). Output is the per-patch mean
of the sigmoid "predictions" head, shape (519,).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from core.vendor.mel import frame_patches, musicnn_mel, tile_to_length
from core.vendor.onnx_session import make_session, pick_output_by_lastdim

_GENRE_CLASSES = 519

# The maest embedding case reads the transformer-block token output the
# Essentia .pb exposed as StatefulPartitionedCall:7. In the ONNX export that
# tensor is named "layer_4_tokens": the runtime graph names the twelve block
# outputs layer_0..11_tokens, and the .pb output-slot numbering is NOT linear,
# so this mapping was confirmed empirically by parity (cos=1.0, max-abs=5e-6 vs
# the .pb tensor), not by index — see scripts/maest_embedding_parity.py. Bind by
# exact name: the ONNX declares these outputs with rank-1 {-1} shape metadata,
# so last-dim matching cannot disambiguate them.
_TOKEN_OUTPUT = "layer_4_tokens"

# Essentia's TensorflowPredictMAEST standardizes the MusiCNN mel before the
# transformer as ``(mel - mean) / (std * 2)``. These constants are copied
# verbatim from Essentia's src tensorflowpredictmaest.h so the ONNX path
# reproduces the .pb predictions bit-for-bit (without them the transformer
# sees unnormalized input and returns entirely different genres).
_MEL_MEAN = 2.06755686098554
_MEL_STD = 1.268292820667291

# MAEST consumes one 30 s window at 16 kHz (1876 mel frames); clips shorter
# than the window are tiled up to it, matching the old Essentia wrapper.
_MIN_SAMPLES = 30 * 16000


def maest_patches(audio: np.ndarray, *, patch_frames: int, patch_hop: int) -> np.ndarray:
    """Tile → MusiCNN mel → MAEST standardize → frame into model patches.

    Shared by both MAEST heads (the 519 predictions and the :7 token
    embedding): identical preprocessing, only the requested output tensor
    differs. Keeping it here means the load-bearing ``(mel - MEAN) / (2*STD)``
    normalization and the 30 s tiling rule live in exactly one place.
    """
    audio = tile_to_length(audio, _MIN_SAMPLES)
    mel = (musicnn_mel(audio) - _MEL_MEAN) / (2.0 * _MEL_STD)
    return frame_patches(mel, patch_size=patch_frames, hop_size=patch_hop)


class MAEST:
    def __init__(
        self, pb: Path, *, patch_frames: int = 1876, patch_hop: int = 1875, **_ignored
    ):
        # ``pb`` is the .onnx path (kept named ``pb`` for caller compatibility).
        # ``_ignored`` swallows legacy kwargs (input/output/min_samples).
        self._patch_frames = int(patch_frames)
        self._patch_hop = int(patch_hop)
        self._session = make_session(Path(pb))
        self._input = self._session.get_inputs()[0].name
        # Two width-519 outputs exist: the sigmoid "predictions" head and the
        # linear "logits". The ONNX export names them ("activations", "logits");
        # the SavedModel/.pb uses ("StatefulPartitionedCall:0", ":13"). Select
        # the predictions head by excluding the logits output (matches the .pb
        # reference, which reads the sigmoid predictions).
        self._output = pick_output_by_lastdim(
            self._session.get_outputs(),
            _GENRE_CLASSES,
            prefer=lambda o: "logit" not in o.name.lower(),
        )

    def predict(self, audio: np.ndarray) -> np.ndarray:
        """Average per-patch genre519 predictions over time -> (519,)."""
        # patchSize 1876 / patchHopSize 1875 mirror Essentia's MAEST 30 s
        # config; the trailing partial patch is discarded by frame_patches.
        patches = maest_patches(
            audio, patch_frames=self._patch_frames, patch_hop=self._patch_hop
        )
        out = self._session.run([self._output], {self._input: patches})[0]
        return out.reshape(-1, out.shape[-1]).mean(axis=0)

    def __enter__(self) -> "MAEST":
        return self

    def __exit__(self, *exc) -> None:
        return None


class MaestTokens:
    """MAEST :7 transformer-block tokens for the raw audio embedding case.

    Returns the per-patch token tensor ``[n_patches, n_tokens, 768]`` from the
    ``StatefulPartitionedCall:7`` output (token 0 = CLS, token 1 = DIST, 2: =
    signal). The ONNX tensor is 3-D; the old Essentia .pb wrapper returned the
    same data as 4-D ``[N, 1, T, 768]``. Aggregation into the 2304-d vector is
    owned by the embeddings stage, not here.
    """

    def __init__(
        self,
        onnx_path: Path,
        *,
        token_output: str = _TOKEN_OUTPUT,
        patch_frames: int = 1876,
        patch_hop: int = 1875,
    ):
        self._patch_frames = int(patch_frames)
        self._patch_hop = int(patch_hop)
        self._session = make_session(Path(onnx_path))
        self._input = self._session.get_inputs()[0].name
        names = [o.name for o in self._session.get_outputs()]
        if token_output not in names:
            raise ValueError(
                f"ONNX MAEST has no output {token_output!r}; available: {names}"
            )
        self._output = token_output

    def tokens(self, audio: np.ndarray) -> np.ndarray:
        """Per-patch token tensor, shape ``(n_patches, n_tokens, 768)``."""
        patches = maest_patches(
            audio, patch_frames=self._patch_frames, patch_hop=self._patch_hop
        )
        return self._session.run([self._output], {self._input: patches})[0]

    def __enter__(self) -> "MaestTokens":
        return self

    def __exit__(self, *exc) -> None:
        return None
