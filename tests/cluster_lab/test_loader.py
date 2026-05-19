from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.cluster_lab.loader import l2_normalize, load_sandwich_matrix

LEGACY_DB = Path("data/old/inst2vec.db")


@pytest.mark.skipif(not LEGACY_DB.exists(), reason="legacy DB absent")
def test_load_sandwich_matrix_shape() -> None:
    mat, uids = load_sandwich_matrix(LEGACY_DB)
    assert mat.dtype == np.float32
    assert mat.shape[0] == len(uids) > 0
    assert mat.ndim == 2


def test_l2_normalize_unit_norm() -> None:
    rng = np.random.default_rng(0)
    mat = rng.standard_normal((10, 8)).astype(np.float32)
    out = l2_normalize(mat)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-6)


def test_l2_normalize_zero_row_safe() -> None:
    mat = np.zeros((3, 4), dtype=np.float32)
    out = l2_normalize(mat)
    assert np.isfinite(out).all()
