"""Parity: Essentia .pb MAEST :7 aggregation vs the ONNX MaestTokens path.

Run on the GPU box with both model formats in models/mir and a few wavs in
data/audio_mir. Reports per-clip cosine similarity and worst per-element
max-abs diff of the 2304-d concat(CLS, DIST, mean(signal)) vector.
Exit 0 = within tolerance (cosine >= 0.999 AND max-abs <= 1e-2), 1 = drift.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np

MODEL_DIR = Path("models/mir")
MAEST_ONNX = MODEL_DIR / "discogs-maest-30s-pw-519l-1.onnx"
MAEST_PB = MODEL_DIR / "discogs-maest-30s-pw-519l-1.pb"

COSINE_MIN = 0.999
MAXABS_TOL = 1e-2
MAEST_MIN_SAMPLES = 30 * 16000  # one 30 s window at 16 kHz


def _ref_vec(audio: np.ndarray) -> np.ndarray:
    """Essentia .pb :7 tokens -> squeeze middle axis -> 2304-d aggregate."""
    from essentia.standard import (
        TensorflowPredictMAEST,  # ty: ignore[unresolved-import]
    )

    from modules.embeddings.maest import _aggregate_patches

    out = TensorflowPredictMAEST(
        graphFilename=str(MAEST_PB),
        input="serving_default_melspectrogram",
        output="StatefulPartitionedCall:7",
    )(audio)
    # .pb returns [N, 1, T, 768]; drop the singleton axis to match ONNX 3-D.
    return _aggregate_patches(np.asarray(out)[:, 0, :, :])


def main() -> int:
    from essentia.standard import MonoLoader  # ty: ignore[unresolved-import]

    from core.vendor.maest import MaestTokens
    from core.vendor.mel import tile_to_length
    from modules.embeddings.maest import _aggregate_patches

    wavs = sorted(glob.glob("data/audio_mir/*.wav"))[:8]
    if not wavs:
        print("PARITY: no wavs in data/audio_mir")
        return 1

    model = MaestTokens(MAEST_ONNX)

    worst_cos, worst_abs = 1.0, 0.0
    for w in wavs:
        audio = MonoLoader(filename=w, sampleRate=16000, resampleQuality=4)()
        audio = tile_to_length(audio, MAEST_MIN_SAMPLES)
        ref = _ref_vec(audio)
        onnx = _aggregate_patches(np.asarray(model.tokens(audio)))
        cos = float(ref @ onnx / (np.linalg.norm(ref) * np.linalg.norm(onnx) + 1e-9))
        worst_cos = min(worst_cos, cos)
        worst_abs = max(worst_abs, float(np.max(np.abs(ref - onnx))))

    print(f"PARITY: maest_embed cosine={worst_cos:.6f} max_abs={worst_abs:.6f}")
    ok = worst_cos >= COSINE_MIN and worst_abs <= MAXABS_TOL
    print("PARITY:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
