"""Compare the legacy Essentia .pb MIR path against the new ONNX path.

Run on the GPU box with both model formats present in models/mir and at least
a few extracted wavs in data/audio_mir. Reports max abs diff on MAEST genre
probs, EffNet embedding cosine similarity, and top-k genre label agreement.
Exit code 0 = within tolerance, 1 = drift.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np

MODEL_DIR = Path("models/mir")
MAEST_ONNX = MODEL_DIR / "discogs-maest-30s-pw-519l-1.onnx"
MAEST_PB = MODEL_DIR / "discogs-maest-30s-pw-519l-1.pb"
EFFNET_ONNX = MODEL_DIR / "discogs-effnet-bsdynamic-1.onnx"
EFFNET_PB = MODEL_DIR / "discogs-effnet-bs64-1.pb"

PROB_TOL = 1e-2  # MAEST sigmoid prob max-abs diff
COSINE_MIN = 0.999  # EffNet embedding cosine similarity
TOPK = 10
MAEST_MIN_SAMPLES = 30 * 16000  # MAEST consumes one 30 s window at 16 kHz


def _ref_maest(audio):
    from essentia.standard import (
        TensorflowPredictMAEST,  # ty: ignore[unresolved-import]
    )

    out = TensorflowPredictMAEST(
        graphFilename=str(MAEST_PB),
        input="serving_default_melspectrogram",
        output="StatefulPartitionedCall:0",
    )(audio)
    return out.reshape(-1, out.shape[-1]).mean(axis=0)


def _ref_effnet(audio):
    from essentia.standard import (
        TensorflowPredictEffnetDiscogs,  # ty: ignore[unresolved-import]
    )

    return TensorflowPredictEffnetDiscogs(
        graphFilename=str(EFFNET_PB), output="PartitionedCall:1"
    )(audio)


def main() -> int:
    from essentia.standard import MonoLoader  # ty: ignore[unresolved-import]

    from core.vendor.effnet import EffNet
    from core.vendor.maest import MAEST
    from core.vendor.mel import tile_to_length

    wavs = sorted(glob.glob("data/audio_mir/*.wav"))[:8]
    if not wavs:
        print("PARITY: no wavs in data/audio_mir")
        return 1

    maest = MAEST(pb=MAEST_ONNX, patch_frames=1876)
    effnet = EffNet(embed_pb=EFFNET_ONNX, heads={}, patch_frames=128)

    worst_prob, worst_cos, topk_ok = 0.0, 1.0, True
    for w in wavs:
        audio = MonoLoader(filename=w, sampleRate=16000, resampleQuality=4)()
        maest_audio = tile_to_length(audio, MAEST_MIN_SAMPLES)
        rp, op = _ref_maest(maest_audio), maest.predict(maest_audio)
        worst_prob = max(worst_prob, float(np.max(np.abs(rp - op))))
        topk_ok &= set(np.argsort(-rp)[:TOPK]) == set(np.argsort(-op)[:TOPK])

        re_, oe = _ref_effnet(audio).mean(axis=0), effnet.embed(audio).mean(axis=0)
        cos = float(re_ @ oe / (np.linalg.norm(re_) * np.linalg.norm(oe) + 1e-9))
        worst_cos = min(worst_cos, cos)

    print(
        f"PARITY: maest_max_abs={worst_prob:.5f} effnet_cos={worst_cos:.5f} "
        f"topk_match={topk_ok}"
    )
    ok = worst_prob <= PROB_TOL and worst_cos >= COSINE_MIN and topk_ok
    print("PARITY:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
