"""Deterministic fakes for MAEST + EffNet used by MIR pipeline tests."""

from __future__ import annotations

import numpy as np

from modules.mir.models import EFFNET_HEAD_SPECS


class FakeMAEST:
    """Returns a deterministic 519-d genre probability vector."""

    def __init__(self, *args, **kwargs):
        pass

    def predict(self, audio):
        return np.linspace(0.0, 1.0, 519, dtype=np.float32)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


class FakeEffNet:
    """Returns a deterministic 1280-d embedding and per-head canned outputs."""

    def __init__(self, *args, **kwargs):
        pass

    def embed(self, audio):
        return np.ones((1, 1280), dtype=np.float32)

    def predict_all(self, embedding):
        out: dict[str, np.ndarray] = {}
        for name, (_filename, _output_op) in EFFNET_HEAD_SPECS.items():
            if name in {"approachability", "engagement"}:
                out[name] = np.array([0.42], dtype=np.float32)
            elif name == "moodtheme":
                out[name] = np.linspace(0.0, 1.0, 56, dtype=np.float32)
            elif name == "instrument":
                out[name] = np.linspace(0.0, 1.0, 40, dtype=np.float32)
            else:
                # binary head: pos at index 0 (POS=0 convention)
                out[name] = np.array([0.9, 0.1], dtype=np.float32)
        return out

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None
