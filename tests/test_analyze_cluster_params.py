"""Unit tests for the cluster-param analysis script's pure helpers."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_cluster_params.py"
_SPEC = importlib.util.spec_from_file_location("analyze_cluster_params", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
acp = importlib.util.module_from_spec(_SPEC)
sys.modules["analyze_cluster_params"] = acp
_SPEC.loader.exec_module(acp)


def test_load_runs_reads_real_legacy_db():
    """Smoke test: the loader reads the real legacy DB and returns the expected columns."""
    db_path = Path(__file__).resolve().parents[1] / "data" / "old" / "inst2vec.db"
    if not db_path.exists():
        pytest.skip("legacy DB not present")
    df = acp.load_runs(str(db_path), "sandwich")
    assert len(df) > 0
    assert set(acp.ALL_PARAM_COLS).issubset(df.columns)
    assert {"dbcv", "silhouette", "n_clusters", "noise_ratio",
            "min_size", "median_size", "max_size"}.issubset(df.columns)


def test_load_runs_is_readonly(tmp_path):
    """The loader must not write to the source DB (read-only URI)."""
    src = Path(__file__).resolve().parents[1] / "data" / "old" / "inst2vec.db"
    if not src.exists():
        pytest.skip("legacy DB not present")
    copy = tmp_path / "copy.db"
    shutil.copyfile(src, copy)
    # Force the file to read-only: a writable connection would raise here.
    copy.chmod(0o444)
    mtime_before = copy.stat().st_mtime_ns
    acp.load_runs(str(copy), "sandwich")
    mtime_after = copy.stat().st_mtime_ns
    assert mtime_before == mtime_after, "loader mutated the DB file"
    # No journal/WAL sibling files should have been created.
    assert not (tmp_path / "copy.db-journal").exists()
    assert not (tmp_path / "copy.db-wal").exists()
    assert not (tmp_path / "copy.db-shm").exists()


@pytest.mark.parametrize("n_clusters,expected_range", [
    (1, (0.0, 0.2)),
    (3, (0.0, 0.5)),
    (10, (0.95, 1.0)),
    (25, (0.95, 1.0)),
    (50, (0.95, 1.0)),
    (200, (0.0, 0.5)),
    (1000, (0.0, 0.05)),
])
def test_count_penalty_shape(n_clusters, expected_range):
    lo, hi = expected_range
    val = acp.compute_count_penalty(n_clusters)
    assert lo <= val <= hi, f"count_penalty({n_clusters}) = {val} not in {expected_range}"


@pytest.mark.parametrize("noise,expected_range", [
    (0.0, (0.0, 0.5)),
    (0.01, (0.0, 0.7)),
    (0.05, (0.9, 1.0)),
    (0.3, (0.9, 1.0)),
    (0.5, (0.85, 1.0)),
    (0.9, (0.0, 0.3)),
    (1.0, (0.0, 0.1)),
])
def test_noise_penalty_shape(noise, expected_range):
    lo, hi = expected_range
    val = acp.compute_noise_penalty(noise)
    assert lo <= val <= hi


def test_shape_penalty_balanced_clusters_max():
    # min/median ≈ 1, max/median ≈ 1 → balanced → ~1.0
    assert acp.compute_shape_penalty(min_size=50, median_size=50, max_size=50) > 0.95


def test_shape_penalty_giant_cluster_low():
    # one giant cluster → low score
    assert acp.compute_shape_penalty(min_size=2, median_size=10, max_size=500) < 0.3


def test_shape_penalty_singleton_clusters_low():
    # many singletons against larger ones → low score
    assert acp.compute_shape_penalty(min_size=1, median_size=80, max_size=120) < 0.5


def test_penalties_handle_zero_median():
    # No degenerate divide-by-zero
    assert 0.0 <= acp.compute_shape_penalty(0, 0, 0) <= 1.0


def _make_row(**overrides):
    base = {
        "dbcv": 0.5,
        "silhouette": 0.5,
        "n_clusters": 20,
        "noise_ratio": 0.1,
        "min_size": 40,
        "median_size": 50,
        "max_size": 80,
    }
    base.update(overrides)
    return base


def test_quality_score_pre_filtered_rows_get_zero():
    df = pd.DataFrame([
        _make_row(),                       # scored
        _make_row(dbcv=None, silhouette=None),  # pre-filtered
    ])
    out = acp.compute_quality_score(df)
    assert out.loc[1, "quality_score"] == 0.0
    assert out.loc[0, "quality_score"] > 0.0


def test_quality_score_high_for_balanced_run():
    df = pd.DataFrame([_make_row(dbcv=0.6, silhouette=0.5)])
    out = acp.compute_quality_score(df)
    assert out.loc[0, "quality_score"] > 0.3


def test_quality_score_low_for_degenerate_run():
    # 3 clusters, ~1% noise — looks great on raw DBCV but is degenerate
    df = pd.DataFrame([_make_row(
        dbcv=0.83, silhouette=0.26, n_clusters=3, noise_ratio=0.01,
        min_size=10, median_size=20, max_size=30,
    )])
    out = acp.compute_quality_score(df)
    # Degenerate runs should rank well below balanced ones
    assert out.loc[0, "quality_score"] < 0.3


def test_quality_score_includes_component_columns():
    df = pd.DataFrame([_make_row()])
    out = acp.compute_quality_score(df)
    for col in [
        "quality_score",
        "dbcv_norm",
        "silhouette_norm",
        "count_penalty",
        "noise_penalty",
        "shape_penalty",
    ]:
        assert col in out.columns


def test_detect_varying_axes_ignores_constants():
    df = pd.DataFrame({
        "umap_n_components": [10, 15, 20],
        "umap_n_neighbors": [10, 10, 10],   # constant
        "umap_metric": ["cosine", "euclidean", "cosine"],
        "random_state": [42, 42, 42],         # constant
        "hdbscan_min_samples": [None, None, None],  # all null = constant
    })
    varying, fixed = acp.detect_varying_axes(df, list(df.columns))
    assert "umap_n_components" in varying
    assert "umap_metric" in varying
    assert "umap_n_neighbors" not in varying
    assert "random_state" not in varying
    assert "hdbscan_min_samples" not in varying
    assert fixed["umap_n_neighbors"] == 10
    assert fixed["random_state"] == 42


def test_per_param_stats_returns_one_row_per_value():
    df = pd.DataFrame({
        "umap_metric": ["cosine"] * 4 + ["euclidean"] * 6,
        "quality_score": [0.5, 0.4, 0.6, 0.7, 0.0, 0.0, 0.3, 0.2, 0.1, 0.0],
        "dbcv": [0.5, 0.4, 0.6, 0.7, None, None, 0.3, 0.2, 0.1, 0.0],
        "silhouette": [0.1] * 10,
    })
    stats = acp.per_param_stats(df, "umap_metric")
    assert len(stats) == 2
    cosine_row = stats[stats["value"] == "cosine"].iloc[0]
    assert cosine_row["n_rows"] == 4
    assert cosine_row["failure_rate"] == 0.0
    euclid_row = stats[stats["value"] == "euclidean"].iloc[0]
    assert euclid_row["n_rows"] == 6
    assert euclid_row["failure_rate"] == pytest.approx(2 / 6)


def test_kruskal_dunn_returns_pvalue_and_effect_size():
    rng = np.random.RandomState(0)
    df = pd.DataFrame({
        "axis": ["a"] * 30 + ["b"] * 30 + ["c"] * 30,
        "quality_score": (
            list(rng.normal(0.7, 0.05, 30))
            + list(np.random.RandomState(1).normal(0.3, 0.05, 30))
            + list(np.random.RandomState(2).normal(0.1, 0.05, 30))
        ),
    })
    result = acp.kruskal_dunn(df, "axis")
    assert result["kw_pvalue"] < 0.01
    assert 0.0 <= result["eta_squared"] <= 1.0
    assert isinstance(result["dunn_pvalues"], pd.DataFrame)
