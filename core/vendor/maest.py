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
        audio = tile_to_length(audio, _MIN_SAMPLES)
        mel = (musicnn_mel(audio) - _MEL_MEAN) / (2.0 * _MEL_STD)
        # patchSize 1876 / patchHopSize 1875 mirror Essentia's MAEST 30 s
        # config; the trailing partial patch is discarded by frame_patches.
        patches = frame_patches(
            mel, patch_size=self._patch_frames, hop_size=self._patch_hop
        )
        out = self._session.run([self._output], {self._input: patches})[0]
        return out.reshape(-1, out.shape[-1]).mean(axis=0)

    def __enter__(self) -> "MAEST":
        return self

    def __exit__(self, *exc) -> None:
        return None
