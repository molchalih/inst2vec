"""Tiny serde helpers for embedding blobs."""

from __future__ import annotations

import numpy as np


def to_bytes(tensor) -> bytes:
    """Serialize a torch-like tensor, numpy array, or list to a float32 byte blob."""
    if hasattr(tensor, "cpu"):
        return tensor.cpu().float().numpy().tobytes()
    return np.asarray(tensor, dtype=np.float32).tobytes()


def bytes_to_array(blob: bytes) -> np.ndarray:
    """Deserialize a float32 byte blob to a writable numpy array."""
    return np.frombuffer(blob, dtype=np.float32).copy()
