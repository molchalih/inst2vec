"""Unit tests for the cluster-param analysis script's pure helpers."""

from __future__ import annotations

import importlib.util
import math
import shutil
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

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


def test_detect_uninformative_axes_finds_zero_effect_column():
    # `noise_axis` doubles every config but never changes quality_score —
    # exactly the umap2d_metric case.
    rows = []
    for nc in [10, 15, 20]:
        for noise_v in ["a", "b"]:
            rows.append({
                "umap_n_components": nc,
                "noise_axis": noise_v,
                "quality_score": 0.5 if nc == 20 else 0.2,
            })
    df = pd.DataFrame(rows)
    uninformative = acp.detect_uninformative_axes(
        df, ["umap_n_components", "noise_axis"]
    )
    assert uninformative == ["noise_axis"]


def test_detect_uninformative_axes_keeps_real_axis():
    rows = [
        {"a": 1, "b": "x", "quality_score": 0.1},
        {"a": 1, "b": "y", "quality_score": 0.3},
        {"a": 2, "b": "x", "quality_score": 0.4},
        {"a": 2, "b": "y", "quality_score": 0.6},
    ]
    df = pd.DataFrame(rows)
    assert acp.detect_uninformative_axes(df, ["a", "b"]) == []


def test_dedupe_on_axes_collapses_redundant_rows():
    rows = [
        {"a": 1, "b": "x", "quality_score": 0.5, "dbcv": 0.5},
        {"a": 1, "b": "x", "quality_score": 0.5, "dbcv": 0.5},  # duplicate
        {"a": 2, "b": "x", "quality_score": 0.6, "dbcv": 0.6},
    ]
    df = pd.DataFrame(rows)
    deduped = acp.dedupe_on_axes(df, ["a", "b"])
    assert len(deduped) == 2


def test_fit_classifier_learns_viability_signal():
    rng = np.random.RandomState(0)
    n = 300
    df = pd.DataFrame({
        "umap_metric": rng.choice(["cosine", "euclidean"], n),
        "hdbscan_min_cluster_size": rng.choice([10, 15, 25], n),
    })
    # leaf-like: euclidean almost always fails
    fail_prob = np.where(df["umap_metric"] == "euclidean", 0.95, 0.1)
    is_failed = rng.uniform(size=n) < fail_prob
    df["dbcv"] = np.where(is_failed, np.nan, rng.uniform(0.3, 0.7, n))

    _model, info = acp.fit_classifier(
        df, feature_cols=["umap_metric", "hdbscan_min_cluster_size"]
    )
    assert info["cv_auc_mean"] > 0.75


def test_fit_classifier_handles_single_class_target():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "dbcv": [0.5, 0.6, 0.7, 0.8]})
    model, info = acp.fit_classifier(df, feature_cols=["a"])
    assert model is None
    assert "single-class" in info.get("note", "")


def test_fit_classifier_skips_auc_when_minority_below_two():
    # 1 viable vs 19 failed — too few minority samples for stratified CV.
    df = pd.DataFrame({
        "a": list(range(20)),
        "dbcv": [0.5] + [float("nan")] * 19,
    })
    model, info = acp.fit_classifier(df, feature_cols=["a"])
    # Model is still fit so callers that want feature importance don't break,
    # but the AUC must be NaN rather than a misleading fold-averaged value.
    assert model is not None
    assert math.isnan(info["cv_auc_mean"])
    assert math.isnan(info["cv_auc_std"])
    assert "too few minority" in info.get("note", "")


def test_fit_classifier_reduces_folds_when_minority_small():
    # 3 viable vs 17 failed — must drop to 3 folds, not run 5.
    rng = np.random.RandomState(0)
    n = 20
    df = pd.DataFrame({"a": rng.uniform(size=n)})
    dbcv = [float("nan")] * n
    dbcv[2] = 0.4
    dbcv[7] = 0.5
    dbcv[13] = 0.6
    df["dbcv"] = dbcv
    _model, info = acp.fit_classifier(df, feature_cols=["a"])
    assert info["cv_n_splits"] == 3
    assert not math.isnan(info["cv_auc_mean"])


def test_interaction_strength_zero_for_additive_grid():
    # Cell mean = row_main + col_main + grand — no interaction.
    rows = []
    for a, ra in zip([1, 2, 3], [0.0, 0.1, 0.2], strict=True):
        for b, rb in zip(["x", "y"], [0.0, 0.05], strict=True):
            rows.append({"a": a, "b": b, "quality_score": 0.5 + ra + rb})
    df = pd.DataFrame(rows)
    rms, pivot = acp.interaction_strength(df, "a", "b")
    assert rms < 1e-9
    assert pivot.shape == (3, 2)


def test_interaction_strength_positive_when_nonadditive():
    # cell value flips on the (a=2, b=y) corner — strongly non-additive
    rows = [
        {"a": 1, "b": "x", "quality_score": 0.1},
        {"a": 1, "b": "y", "quality_score": 0.1},
        {"a": 2, "b": "x", "quality_score": 0.1},
        {"a": 2, "b": "y", "quality_score": 0.9},
    ]
    df = pd.DataFrame(rows)
    rms, _ = acp.interaction_strength(df, "a", "b")
    assert rms > 0.1


def test_interaction_ranked_orders_pairs_by_strength():
    rows = []
    for a in [1, 2]:
        for b in ["x", "y"]:
            for c in ["p", "q"]:
                # (a, b) is the interactive pair; c is purely additive
                q = 0.9 if (a == 2 and b == "y") else 0.1
                q += 0.05 if c == "q" else 0.0
                rows.append({"a": a, "b": b, "c": c, "quality_score": q})
    df = pd.DataFrame(rows)
    ranked = acp.interaction_ranked(df, ["a", "b", "c"])
    assert len(ranked) == 3
    top = ranked.iloc[0]
    pair = tuple(sorted([top["axis_a"], top["axis_b"]]))
    assert pair == ("a", "b")


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


def test_surrogate_model_fits_and_predicts():
    rng = np.random.RandomState(0)
    n = 200
    df = pd.DataFrame({
        "umap_n_components": rng.choice([10, 15, 20, 30], n),
        "umap_metric": rng.choice(["cosine", "euclidean"], n),
        "hdbscan_min_cluster_size": rng.choice([10, 15, 25], n),
    })
    df["quality_score"] = (
        (df["umap_metric"] == "cosine").astype(float) * 0.5
        + (df["umap_n_components"] / 60.0)
        + rng.normal(0, 0.05, n)
    ).clip(0, 1)

    _model, info = acp.fit_surrogate(
        df,
        feature_cols=["umap_n_components", "umap_metric", "hdbscan_min_cluster_size"],
    )
    # Should learn the signal — non-degenerate R²
    assert info["cv_r2"] > 0.3
    preds = info["predict"](df.head(5))
    assert all(0.0 <= p <= 1.5 for p in preds)


def test_detect_edge_optima_flags_top_edge():
    df = pd.DataFrame({
        "umap_n_components": [10, 15, 20, 30] * 5,
        "quality_score": [0.1, 0.2, 0.4, 0.8] * 5,
    })
    flagged = acp.detect_edge_optima(df, ["umap_n_components"])
    assert len(flagged) == 1
    assert flagged[0]["axis"] == "umap_n_components"
    assert flagged[0]["direction"] == "above"


def test_detect_edge_optima_flags_bottom_edge():
    df = pd.DataFrame({
        "umap_n_components": [10, 15, 20, 30] * 5,
        "quality_score": [0.8, 0.4, 0.2, 0.1] * 5,
    })
    flagged = acp.detect_edge_optima(df, ["umap_n_components"])
    assert flagged[0]["direction"] == "below"


def test_detect_edge_optima_skips_interior_optimum():
    df = pd.DataFrame({
        "umap_n_components": [10, 15, 20, 30] * 5,
        "quality_score": [0.2, 0.8, 0.7, 0.1] * 5,
    })
    flagged = acp.detect_edge_optima(df, ["umap_n_components"])
    assert flagged == []


def test_dominance_finds_clearly_dominated_value():
    # umap_metric=cosine always beats umap_metric=correlation, holding other axes fixed.
    rows = []
    for nc in [10, 15, 20]:
        for mcs in [10, 15]:
            rows.append({"umap_metric": "cosine", "umap_n_components": nc,
                         "hdbscan_min_cluster_size": mcs, "quality_score": 0.7})
            rows.append({"umap_metric": "correlation", "umap_n_components": nc,
                         "hdbscan_min_cluster_size": mcs, "quality_score": 0.2})
    df = pd.DataFrame(rows)
    dropped = acp.dominance_analysis(
        df,
        axes=["umap_metric", "umap_n_components", "hdbscan_min_cluster_size"],
    )
    assert "umap_metric" in dropped
    assert "correlation" in dropped["umap_metric"]


def test_dominance_keeps_value_that_wins_somewhere():
    rows = [
        {"umap_metric": "cosine", "umap_n_components": 10, "quality_score": 0.7},
        {"umap_metric": "cosine", "umap_n_components": 20, "quality_score": 0.3},
        {"umap_metric": "euclidean", "umap_n_components": 10, "quality_score": 0.3},
        {"umap_metric": "euclidean", "umap_n_components": 20, "quality_score": 0.7},
    ]
    df = pd.DataFrame(rows)
    dropped = acp.dominance_analysis(
        df, axes=["umap_metric", "umap_n_components"]
    )
    # Neither metric is dominated — each wins at some n_components
    assert dropped.get("umap_metric", []) == []


def test_build_suggested_grid_assembles_sections():
    df = pd.DataFrame({
        "umap_metric": ["cosine", "correlation"] * 6,
        "umap_n_components": [10, 10, 15, 15, 20, 20, 30, 30, 35, 35, 40, 40],
        "quality_score": [0.7, 0.2, 0.6, 0.1, 0.6, 0.1, 0.7, 0.2, 0.8, 0.1, 0.9, 0.1],
    })
    out = acp.build_suggested_grid(
        df,
        varying_axes=["umap_metric", "umap_n_components"],
        ordinal_axes=["umap_n_components"],
        top_pair=("umap_metric", "umap_n_components"),
    )
    assert "drop" in out and "keep" in out and "extend" in out and "focus_regions" in out
    # correlation should be flagged as dominated
    assert "correlation" in out["drop"].get("umap_metric", [])
    # n_components optimum is at 40 (top edge) → "above"
    assert out["extend"]["umap_n_components"]["direction"] == "above"
