"""Load sandwich user embeddings directly from the legacy SQLite database."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np


def load_sandwich_matrix(db_path: str | Path) -> tuple[np.ndarray, list[int]]:
    """Read user_embeddings for embedding_case='sandwich', ordered by user_id.

    The DB is opened read-only via SQLite URI; we never write to the legacy
    file. Embeddings are stored as float32 blobs.
    """
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"Legacy DB not found: {path}")
    uri = f"file:{path}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        rows = conn.execute(
            "SELECT user_id, embedding FROM user_embeddings "
            "WHERE embedding_case = 'sandwich' ORDER BY user_id"
        ).fetchall()
    if not rows:
        return np.empty((0, 0), dtype=np.float32), []
    user_ids = [int(r[0]) for r in rows]
    arrays = [np.frombuffer(r[1], dtype=np.float32).copy() for r in rows]
    return np.stack(arrays), user_ids


def l2_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalization."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return (matrix / norms).astype(matrix.dtype)
