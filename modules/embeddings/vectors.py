"""Tiny serde helpers for embedding blobs."""

from __future__ import annotations

import numpy as np


def to_bytes(tensor) -> bytes:
    """Serialize a torch-like tensor to a float32 byte blob."""
    return tensor.cpu().float().numpy().tobytes()


def bytes_to_array(blob: bytes) -> np.ndarray:
    """Deserialize a float32 byte blob to a writable numpy array."""
    return np.frombuffer(blob, dtype=np.float32).copy()
