import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import csv
import io
import numpy as np
import pytest


# ── helpers imported after the script exists ──────────────────────────────────
# These are tested by importing directly once the script is written.
# For now, define the expected contracts so the tests drive the implementation.


def _import():
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location(
        "explore_audio_whitening",
        pathlib.Path(__file__).parent.parent / "scripts" / "explore_audio_whitening.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── _build_whitened_matrix ─────────────────────────────────────────────────────

def test_build_whitened_none_returns_same_shape():
    mod = _import()
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 64)).astype(np.float32)
    out = mod._build_whitened_matrix(X, n_components=None, use_scaler=False)
    assert out.shape == X.shape
    np.testing.assert_array_equal(out, X)


def test_build_whitened_pca_reduces_dims():
    mod = _import()
    rng = np.random.default_rng(1)
    X = rng.standard_normal((80, 128)).astype(np.float32)
    out = mod._build_whitened_matrix(X, n_components=32, use_scaler=False)
    assert out.shape == (80, 32)


def test_build_whitened_pca_unit_variance():
    """PCA whiten=True should produce approximately unit variance per component."""
    mod = _import()
    rng = np.random.default_rng(2)
    X = rng.standard_normal((200, 64)).astype(np.float32)
    out = mod._build_whitened_matrix(X, n_components=16, use_scaler=False)
    col_vars = out.var(axis=0)
    np.testing.assert_allclose(col_vars, np.ones(16), atol=0.15)


def test_build_whitened_scale_whiten_reduces_dims():
    mod = _import()
    rng = np.random.default_rng(3)
    X = rng.standard_normal((80, 128)).astype(np.float32)
    out = mod._build_whitened_matrix(X, n_components=32, use_scaler=True)
    assert out.shape == (80, 32)


# ── _load_done ─────────────────────────────────────────────────────────────────

FIELDNAMES = [
    "umap_n_components", "umap_n_neighbors", "umap_min_dist", "umap_metric",
    "hdbscan_min_cluster_size", "hdbscan_cluster_selection_method", "hdbscan_metric",
    "whitening", "n_clusters", "noise_ratio", "dbcv", "silhouette",
]

PARAM_COLS = [
    "umap_n_components", "umap_n_neighbors", "umap_min_dist", "umap_metric",
    "hdbscan_min_cluster_size", "hdbscan_cluster_selection_method", "hdbscan_metric",
]


def _make_csv(rows: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=FIELDNAMES)
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def test_load_done_empty_file(tmp_path):
    mod = _import()
    p = tmp_path / "results.csv"
    p.write_text(_make_csv([]))
    done = mod._load_done(str(p), PARAM_COLS)
    assert done == frozenset()


def test_load_done_parses_existing_rows(tmp_path):
    mod = _import()
    p = tmp_path / "results.csv"
    row = {
        "umap_n_components": "15", "umap_n_neighbors": "10", "umap_min_dist": "0.0",
        "umap_metric": "cosine", "hdbscan_min_cluster_size": "10",
        "hdbscan_cluster_selection_method": "eom", "hdbscan_metric": "euclidean",
        "whitening": "whiten_128", "n_clusters": "5", "noise_ratio": "0.2",
        "dbcv": "0.41", "silhouette": "0.3",
    }
    p.write_text(_make_csv([row]))
    done = mod._load_done(str(p), PARAM_COLS)
    key = ("whiten_128", "15", "10", "0.0", "cosine", "10", "eom", "euclidean")
    assert key in done


def test_load_done_missing_file(tmp_path):
    mod = _import()
    done = mod._load_done(str(tmp_path / "nonexistent.csv"), PARAM_COLS)
    assert done == frozenset()
