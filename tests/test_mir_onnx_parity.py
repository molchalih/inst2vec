import os
from pathlib import Path

import pytest

REQUIRED = [
    Path("models/mir/discogs-maest-30s-pw-519l-1.onnx"),
    Path("models/mir/discogs-maest-30s-pw-519l-1.pb"),
    Path("models/mir/discogs-effnet-bsdynamic-1.onnx"),
    Path("models/mir/discogs-effnet-bs64-1.pb"),
]


@pytest.mark.skipif(
    not all(p.exists() for p in REQUIRED) or os.environ.get("MIR_PARITY") != "1",
    reason="needs both model formats + real wavs; set MIR_PARITY=1 on the GPU box",
)
def test_onnx_matches_pb_within_tolerance():
    from scripts.mir_onnx_parity import main

    assert main() == 0
