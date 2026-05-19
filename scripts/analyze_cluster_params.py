"""Offline analyzer for the legacy cluster_runs table.

Mines data/old/inst2vec.db (read-only) to recommend a smarter clustering grid
for future runs: drop dominated parameter values, extend edge-bound axes,
and focus on promising regions.

Standalone — does not import from core/ or modules/. Run via:

    uv run --group analysis python scripts/analyze_cluster_params.py
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

# ── constants ─────────────────────────────────────────────────────────────────

DEFAULT_DB = "data/old/inst2vec.db"
DEFAULT_CASE = "sandwich"
DEFAULT_OUTPUT_DIR = "scripts/output/cluster_param_analysis"
DEFAULT_TOP_N = 20

# Grid columns we ever consider as analytical axes. Constant columns are
# auto-excluded at runtime (see detect_varying_axes).
ALL_PARAM_COLS = (
    "umap_n_components",
    "umap_n_neighbors",
    "umap_min_dist",
    "umap_metric",
    "umap2d_n_neighbors",
    "umap2d_min_dist",
    "umap2d_metric",
    "hdbscan_min_cluster_size",
    "hdbscan_min_samples",
    "hdbscan_cluster_selection_method",
    "hdbscan_metric",
    "random_state",
)


# ── data loading ──────────────────────────────────────────────────────────────


def load_runs(db_path: str, case: str) -> pd.DataFrame:
    """Load all cluster_runs rows for *case* from a read-only SQLite connection.

    Returns a DataFrame with grid columns + metric columns + cluster shape columns.
    The DB is opened via the SQLite URI ``mode=ro`` flag — the loader will fail
    rather than create or mutate the file.
    """
    cols = [
        *ALL_PARAM_COLS,
        "dbcv",
        "silhouette",
        "n_clusters",
        "noise_ratio",
        "min_size",
        "median_size",
        "max_size",
        "disqualified",
    ]
    col_sql = ", ".join(cols)
    uri = f"file:{db_path}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        df = pd.read_sql_query(
            f"SELECT {col_sql} FROM cluster_runs WHERE embedding_case = ?",
            conn,
            params=(case,),
        )
    return df


# ── penalty curves ────────────────────────────────────────────────────────────


def _logistic_window(
    x: float, low: float, high: float, edge_softness: float
) -> float:
    """Returns ~1.0 when x in [low, high], smoothly decays to 0 outside.

    edge_softness controls how steep the sigmoid is at each edge (larger = softer).
    """
    if edge_softness <= 0:
        return 1.0 if low <= x <= high else 0.0
    low_term = 1.0 / (1.0 + math.exp(-(x - low) / edge_softness))
    high_term = 1.0 / (1.0 + math.exp(-(high - x) / edge_softness))
    return low_term * high_term


def compute_count_penalty(n_clusters: int) -> float:
    """~1.0 when n_clusters ∈ [5, 60], smoothly decays outside."""
    return _logistic_window(float(n_clusters), low=5.0, high=60.0, edge_softness=1.5)


def compute_noise_penalty(noise_ratio: float) -> float:
    """1.0 when noise_ratio ∈ [0.02, 0.55], smoothly decays outside.

    Penalizes both 'no noise' (degenerate tight clustering) and 'mostly noise'.
    """
    return _logistic_window(
        float(noise_ratio), low=0.02, high=0.55, edge_softness=0.01
    )


def compute_shape_penalty(min_size: int, median_size: int, max_size: int) -> float:
    """Penalizes severe cluster-size imbalance.

    Uses min_size / median_size (penalizes singleton micro-clusters) and
    max_size / median_size (penalizes one giant cluster). Both ratios are
    fed through soft windows that cap at 1.0 for balanced output.
    """
    if median_size <= 0:
        return 0.0
    low_ratio = float(min_size) / float(median_size)
    high_ratio = float(max_size) / float(median_size)
    # min_ratio should be near 1.0; penalize when it falls below 0.2.
    low_score = _logistic_window(low_ratio, low=0.2, high=10.0, edge_softness=0.1)
    # max_ratio should be near 1; penalize as it grows beyond ~5x (one-sided decay).
    z = (high_ratio - 5.0) / 0.3
    if z >= 0:
        high_score = math.exp(-z) / (1.0 + math.exp(-z))
    else:
        high_score = 1.0 / (1.0 + math.exp(z))
    return low_score * high_score


# ── quality score ─────────────────────────────────────────────────────────────


def compute_quality_score(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with `quality_score` and component columns added.

    Pre-filtered rows (those without a DBCV — never scored by the pipeline)
    receive quality_score = 0. They carry the strongest possible signal that
    the parameter combination is unusable.
    """
    out = df.copy()
    out["dbcv_norm"] = (out["dbcv"].fillna(-1.0) + 1.0) / 2.0
    out["silhouette_norm"] = (out["silhouette"].fillna(-1.0) + 1.0) / 2.0
    out["count_penalty"] = out["n_clusters"].apply(
        lambda x: compute_count_penalty(int(x)) if pd.notna(x) else 0.0
    )
    out["noise_penalty"] = out["noise_ratio"].apply(
        lambda x: compute_noise_penalty(float(x)) if pd.notna(x) else 0.0
    )
    out["shape_penalty"] = out.apply(
        lambda r: compute_shape_penalty(
            int(r["min_size"]) if pd.notna(r["min_size"]) else 0,
            int(r["median_size"]) if pd.notna(r["median_size"]) else 0,
            int(r["max_size"]) if pd.notna(r["max_size"]) else 0,
        ),
        axis=1,
    )

    raw = (
        out["dbcv_norm"]
        * out["silhouette_norm"]
        * out["count_penalty"]
        * out["noise_penalty"]
        * out["shape_penalty"]
    )
    # Pre-filtered rows have NaN dbcv → dbcv_norm = 0 → raw = 0 already.
    # Be explicit about that contract.
    out["quality_score"] = raw.where(out["dbcv"].notna(), 0.0)
    return out


# ── axis detection ────────────────────────────────────────────────────────────


def detect_varying_axes(
    df: pd.DataFrame, candidate_cols: list[str]
) -> tuple[list[str], dict[str, object]]:
    """Split grid columns into varying axes and constant (fixed) ones.

    A column is "constant" if it has zero or one distinct non-null value
    across the dataset. Fixed columns are returned with their constant value
    so they can be reported in the 'fixed parameters' table.
    """
    varying: list[str] = []
    fixed: dict[str, object] = {}
    for col in candidate_cols:
        if col not in df.columns:
            continue
        unique = df[col].dropna().unique()
        if len(unique) <= 1:
            fixed[col] = unique[0] if len(unique) == 1 else None
        else:
            varying.append(col)
    return varying, fixed


def detect_uninformative_axes(
    df: pd.DataFrame, axes: list[str], target: str = "quality_score"
) -> list[str]:
    """Axes whose values produce identical *target* given the other axes.

    For each axis we group by all OTHER axes and check whether within-group
    variance of the target is ~0. Axes that never move the target are
    redundant — they mark duplicate measurements of the same configuration
    (e.g. ``umap2d_metric`` which only affects the post-cluster 2D viz).
    """
    out: list[str] = []
    if target not in df.columns:
        return out
    for axis in axes:
        other = [a for a in axes if a != axis]
        if not other:
            continue
        grouped = df.groupby(other, dropna=False)[target]
        # Require at least one group with >1 distinct value of the axis present,
        # otherwise the variance estimate is meaningless.
        size_ok = (df.groupby(other, dropna=False)[axis].nunique() > 1).any()
        if not size_ok:
            continue
        if grouped.var(ddof=0).fillna(0).max() < 1e-12:
            out.append(axis)
    return out


def dedupe_on_axes(df: pd.DataFrame, axes: list[str]) -> pd.DataFrame:
    """Drop rows that duplicate the same (axes + metrics) tuple.

    Used after demoting uninformative axes so each unique configuration
    appears once.
    """
    metric_cols = [
        "dbcv", "silhouette", "n_clusters", "noise_ratio",
        "min_size", "median_size", "max_size", "quality_score",
    ]
    subset = [c for c in (*axes, *metric_cols) if c in df.columns]
    return df.drop_duplicates(subset=subset).reset_index(drop=True)


# ── univariate analysis ───────────────────────────────────────────────────────


def per_param_stats(df: pd.DataFrame, param: str) -> pd.DataFrame:
    """Per-value summary stats for a single grid axis.

    `failure_rate` is the fraction of rows where the pipeline did NOT compute a
    DBCV (i.e. `dbcv` is null) — the pre-filter signal. It is a stronger and
    more interpretable signal than ``quality_score == 0`` because the latter
    conflates pre-filtered rows with scored-but-degenerate ones.
    """
    rows = []
    for value, group in df.groupby(param, dropna=False):
        n = len(group)
        n_fail = int(group["dbcv"].isna().sum())
        rows.append(
            {
                "value": value,
                "n_rows": n,
                "failure_rate": n_fail / n if n else 0.0,
                "quality_mean": float(group["quality_score"].mean()),
                "quality_median": float(group["quality_score"].median()),
                "quality_max": float(group["quality_score"].max()),
                "dbcv_mean": float(group["dbcv"].dropna().mean())
                if group["dbcv"].notna().any()
                else float("nan"),
                "silhouette_mean": float(group["silhouette"].dropna().mean())
                if group["silhouette"].notna().any()
                else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values("quality_mean", ascending=False)


def kruskal_dunn(df: pd.DataFrame, param: str) -> dict:
    """Kruskal-Wallis omnibus test + post-hoc Dunn pairwise comparisons.

    Returns dict with:
        kw_statistic, kw_pvalue, eta_squared, dunn_pvalues (DataFrame).
    """
    import scikit_posthocs as sp
    from scipy import stats as sp_stats

    groups = [g["quality_score"].values for _, g in df.groupby(param, dropna=False)]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        return {
            "kw_statistic": float("nan"),
            "kw_pvalue": float("nan"),
            "eta_squared": float("nan"),
            "dunn_pvalues": pd.DataFrame(),
        }
    h, p = sp_stats.kruskal(*groups)
    n_total = sum(len(g) for g in groups)
    k = len(groups)
    eta_sq = max(0.0, (h - k + 1) / (n_total - k)) if n_total > k else 0.0
    dunn = sp.posthoc_dunn(
        df, val_col="quality_score", group_col=param, p_adjust="bonferroni"
    )
    return {
        "kw_statistic": float(h),
        "kw_pvalue": float(p),
        "eta_squared": float(eta_sq),
        "dunn_pvalues": dunn,
    }


def plot_boxplot(df: pd.DataFrame, param: str, out_path: Path) -> None:
    """Boxplot of quality_score grouped by *param*. Saves PNG to *out_path*."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    sns.boxplot(data=df, x=param, y="quality_score", ax=ax, color="#6aa6ff")
    sns.stripplot(
        data=df,
        x=param,
        y="quality_score",
        ax=ax,
        color="black",
        size=2,
        alpha=0.3,
        jitter=0.2,
    )
    ax.set_title(f"Quality score by {param}")
    ax.set_xlabel(param)
    ax.set_ylabel("quality_score")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_failure_rate(df: pd.DataFrame, param: str, out_path: Path) -> None:
    """Bar chart of pipeline-failure rate per value of *param*."""
    import matplotlib.pyplot as plt

    stats = per_param_stats(df, param).sort_values("value")
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    ax.bar(stats["value"].astype(str), stats["failure_rate"], color="#d96666")
    ax.set_title(f"Pipeline-failure rate by {param}")
    ax.set_ylabel("fraction of runs the pipeline pre-filtered (no DBCV)")
    ax.set_xlabel(param)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# ── surrogate model ───────────────────────────────────────────────────────────


def _build_design_matrix(
    df: pd.DataFrame, feature_cols: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    """One-hot encode categorical (object) columns, leave numerics alone."""
    parts = []
    out_cols: list[str] = []
    for col in feature_cols:
        s = df[col]
        if s.dtype == object or pd.api.types.is_string_dtype(s):
            dummies = pd.get_dummies(s, prefix=col, dummy_na=False)
            parts.append(dummies.astype(float))
            out_cols.extend(dummies.columns.tolist())
        else:
            parts.append(s.astype(float).to_frame())
            out_cols.append(col)
    X = pd.concat(parts, axis=1)
    return X, out_cols


def fit_surrogate(
    df: pd.DataFrame, feature_cols: list[str]
) -> tuple[object, dict]:
    """Fit HistGradientBoostingRegressor on params → quality_score.

    Returns (model, info) where info has cv_r2, feature_names, source_cols,
    and a `predict` callable that accepts a raw param DataFrame (handles
    one-hot internally).
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import cross_val_score

    X, feature_names = _build_design_matrix(df, feature_cols)
    y = df["quality_score"].astype(float).values

    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=0,
    )
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    model.fit(X, y)

    def predict(raw_df: pd.DataFrame) -> np.ndarray:
        X_new, _ = _build_design_matrix(raw_df, feature_cols)
        # Align columns to training feature set (missing one-hots become 0).
        X_new = X_new.reindex(columns=feature_names, fill_value=0.0)
        return model.predict(X_new)

    info = {
        "cv_r2": float(cv_scores.mean()),
        "cv_r2_std": float(cv_scores.std()),
        "feature_names": feature_names,
        "source_cols": feature_cols,
        "predict": predict,
        "X": X,
        "y": y,
    }
    return model, info


def fit_classifier(
    df: pd.DataFrame, feature_cols: list[str]
) -> tuple[object, dict]:
    """Fit HistGradientBoostingClassifier predicting P(viable | params).

    "Viable" means the pipeline produced a DBCV score (not pre-filtered).
    Returns (model, info) with cv_auc_mean / cv_auc_std for 5-fold CV.
    Returns (None, info) if the target is single-class.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_score

    X, feature_names = _build_design_matrix(df, feature_cols)
    y = df["dbcv"].notna().astype(int).values

    info: dict = {
        "feature_names": feature_names,
        "source_cols": feature_cols,
        "n_viable": int(y.sum()),
        "n_total": len(y),
    }
    if y.sum() in (0, len(y)):
        info["cv_auc_mean"] = float("nan")
        info["cv_auc_std"] = float("nan")
        info["note"] = "single-class target — classifier skipped"
        return None, info

    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=6, random_state=0,
    )
    # Need at least one minority sample per fold for ROC-AUC to be defined;
    # otherwise sklearn returns NaN for that fold and the reported mean is
    # silently misleading. Cap folds at the minority count (min 2) and skip
    # CV entirely if even 2-fold can't be stratified.
    from sklearn.model_selection import StratifiedKFold
    minority = int(min(y.sum(), len(y) - y.sum()))
    if minority < 2:
        info["cv_auc_mean"] = float("nan")
        info["cv_auc_std"] = float("nan")
        info["note"] = (
            f"too few minority-class samples ({minority}) for stratified CV — "
            "AUC skipped"
        )
        model.fit(X, y)
        info["X"] = X
        info["y"] = y
        return model, info
    n_splits = min(5, minority)
    cv = cross_val_score(
        model, X, y,
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0),
        scoring="roc_auc",
    )
    model.fit(X, y)
    info["cv_auc_mean"] = float(cv.mean())
    info["cv_auc_std"] = float(cv.std())
    info["cv_n_splits"] = n_splits
    info["X"] = X
    info["y"] = y
    return model, info


def permutation_importance_report(model: object, info: dict) -> pd.DataFrame:
    """Permutation feature importance, aggregated back to the raw param level."""
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        model, info["X"], info["y"], n_repeats=15, random_state=0, n_jobs=-1
    )
    raw_imp = pd.DataFrame({
        "feature": info["feature_names"],
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    })

    def base_of(name: str) -> str:
        for col in info.get("source_cols", []):
            if name == col or name.startswith(col + "_"):
                return col
        return name

    raw_imp["base_param"] = raw_imp["feature"].apply(base_of)
    return raw_imp


def plot_feature_importance(importance_df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    grouped = importance_df.groupby("base_param")["importance_mean"].sum().sort_values()
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.barh(grouped.index, grouped.values, color="#6aa6ff")
    ax.set_title("Permutation feature importance (aggregated per param)")
    ax.set_xlabel("mean importance (R² drop)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_partial_dependence(
    info: dict, df: pd.DataFrame, param: str, out_path: Path
) -> None:
    """Manual PDP: for each unique value of *param*, average predictions
    holding everything else at observed values (we just substitute the value
    across the full dataset and average)."""
    import matplotlib.pyplot as plt

    values = sorted(df[param].dropna().unique(), key=lambda v: (isinstance(v, str), v))
    means = []
    for v in values:
        copy = df.copy()
        copy[param] = v
        preds = info["predict"](copy)
        means.append(float(preds.mean()))

    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    xs = [str(v) for v in values]
    ax.plot(xs, means, marker="o", color="#6aa6ff")
    ax.set_title(f"Partial dependence on {param}")
    ax.set_xlabel(param)
    ax.set_ylabel("mean predicted quality_score")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_ice(
    info: dict, df: pd.DataFrame, param: str, out_path: Path, n_samples: int = 40
) -> None:
    """ICE plot for *param*: per-row prediction curve as we sweep the param."""
    import matplotlib.pyplot as plt

    values = sorted(df[param].dropna().unique(), key=lambda v: (isinstance(v, str), v))
    sample = df.sample(min(n_samples, len(df)), random_state=0)

    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    for _, row in sample.iterrows():
        ys = []
        for v in values:
            tmp = pd.DataFrame([row])
            tmp[param] = v
            ys.append(float(info["predict"](tmp)[0]))
        ax.plot([str(v) for v in values], ys, color="#888", alpha=0.3)
    ax.set_title(f"ICE curves: predicted quality_score vs {param}")
    ax.set_xlabel(param)
    ax.set_ylabel("predicted quality_score")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# ── interactions ──────────────────────────────────────────────────────────────


def plot_interaction_heatmap(
    df: pd.DataFrame, param_a: str, param_b: str, out_path: Path
) -> None:
    """Heatmap of mean quality_score over (param_a × param_b)."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    pivot = df.pivot_table(
        index=param_a, columns=param_b, values="quality_score", aggfunc="mean"
    )
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    sns.heatmap(
        pivot, annot=True, fmt=".2f", cmap="viridis", ax=ax,
        cbar_kws={"label": "mean quality_score"},
    )
    ax.set_title(f"Interaction: {param_a} × {param_b}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def interaction_strength(
    df: pd.DataFrame, param_a: str, param_b: str, target: str = "quality_score"
) -> tuple[float, pd.DataFrame]:
    """Non-additive interaction strength between two axes.

    Computed as the RMS of (cell mean − additive prediction from main effects).
    A pair with zero interaction has cell means fully explained by row + column
    marginals. Returns (rms, pivot_of_cell_means).
    """
    sub = df.dropna(subset=[param_a, param_b, target])
    if len(sub) == 0:
        return 0.0, pd.DataFrame()
    pivot = sub.pivot_table(
        index=param_a, columns=param_b, values=target, aggfunc="mean"
    )
    if pivot.empty:
        return 0.0, pivot
    grand = float(np.nanmean(pivot.values))
    row_dev = pivot.mean(axis=1).to_numpy() - grand
    col_dev = pivot.mean(axis=0).to_numpy() - grand
    additive = grand + row_dev[:, None] + col_dev[None, :]
    residual = pivot.to_numpy() - additive
    rms = float(np.sqrt(np.nanmean(residual ** 2)))
    return rms, pivot


def interaction_ranked(
    df: pd.DataFrame, axes: list[str], target: str = "quality_score"
) -> pd.DataFrame:
    """Rank every C(n, 2) axis pair by interaction strength.

    Columns: axis_a, axis_b, interaction_rms, n_cells.
    """
    rows = []
    for i, a in enumerate(axes):
        for b in axes[i + 1:]:
            rms, pivot = interaction_strength(df, a, b, target=target)
            rows.append({
                "axis_a": a,
                "axis_b": b,
                "interaction_rms": rms,
                "n_cells": int(pivot.size) if not pivot.empty else 0,
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("interaction_rms", ascending=False).reset_index(drop=True)


# ── boundary detection ────────────────────────────────────────────────────────


def detect_edge_optima(df: pd.DataFrame, ordinal_axes: list[str]) -> list[dict]:
    """For each ordinal axis, flag if the best mean quality_score sits at
    the smallest or largest tested value.
    """
    out = []
    for axis in ordinal_axes:
        if axis not in df.columns:
            continue
        means = df.groupby(axis)["quality_score"].mean().sort_index()
        if len(means) < 2:
            continue
        sorted_values = means.index.tolist()
        best = means.idxmax()
        if best == sorted_values[0]:
            out.append({"axis": axis, "direction": "below", "best_value": best})
        elif best == sorted_values[-1]:
            out.append({"axis": axis, "direction": "above", "best_value": best})
    return out


# ── dominance pruning ─────────────────────────────────────────────────────────


def dominance_analysis(
    df: pd.DataFrame, axes: list[str]
) -> dict[str, list]:
    """Find Pareto-dominated values per axis.

    A value X of axis A is dominated if there exists another value Y of A
    such that for every combination of the other axes that appears with
    BOTH X and Y, score(Y, combo) > score(X, combo). If a value is dominated
    by ANY other value, it's flagged.
    """
    dominated: dict[str, list] = {}
    for axis in axes:
        other = [a for a in axes if a != axis]
        if not other:
            dominated[axis] = []
            continue
        # Aggregate score per (axis_value, other_combo). Take mean across
        # any remaining duplicates (e.g. random_state replicates).
        agg = (
            df.groupby([*other, axis])["quality_score"]
            .mean()
            .reset_index()
        )
        # Pivot: rows = other-combo, cols = axis value
        pivot = agg.pivot_table(
            index=other, columns=axis, values="quality_score"
        )
        values = list(pivot.columns)
        flagged = []
        for x in values:
            # Search for some y that strictly beats x on every shared row
            for y in values:
                if x == y:
                    continue
                shared = pivot[[x, y]].dropna()
                if shared.empty:
                    continue
                if (shared[y] > shared[x]).all():
                    flagged.append(x)
                    break
        dominated[axis] = flagged
    return dominated


# ── suggested grid ────────────────────────────────────────────────────────────


def _jsonify(v):
    """Convert numpy / pandas scalars to JSON-safe Python types."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:
            pass
    return v


def build_suggested_grid(
    df: pd.DataFrame,
    varying_axes: list[str],
    ordinal_axes: list[str],
    top_pair: tuple[str, str] | None,
) -> dict:
    """Compose the JSON recommendation: drop / keep / extend / focus_regions."""
    drop = dominance_analysis(df, varying_axes)
    keep = {}
    for axis in varying_axes:
        all_values = sorted(
            df[axis].dropna().unique(), key=lambda v: (isinstance(v, str), v)
        )
        keep[axis] = [v for v in all_values if v not in drop.get(axis, [])]

    extend = {}
    for flag in detect_edge_optima(df, ordinal_axes):
        axis = flag["axis"]
        values = sorted(df[axis].dropna().unique())
        if flag["direction"] == "below":
            step = values[1] - values[0] if len(values) >= 2 else 1
            candidate = max(0, values[0] - step)
            if candidate >= values[0]:
                # Lower bound is already at the axis floor (e.g. min_dist=0); skip.
                continue
            extend[axis] = {"direction": "below", "suggested_new_values": [candidate]}
        else:
            step = values[-1] - values[-2] if len(values) >= 2 else 1
            new_vals = [values[-1] + step, values[-1] + 2 * step]
            extend[axis] = {"direction": "above", "suggested_new_values": new_vals[:1]}

    focus_regions: list[dict] = []
    if top_pair is not None:
        a, b = top_pair
        pivot = df.pivot_table(
            index=a, columns=b, values="quality_score", aggfunc="mean"
        )
        flat = pivot.stack().sort_values(ascending=False)
        for (av, bv), score in flat.head(3).items():
            focus_regions.append({a: av, b: bv, "mean_quality": float(score)})

    return {
        "drop": {k: [_jsonify(v) for v in vs] for k, vs in drop.items() if vs},
        "keep": {k: [_jsonify(v) for v in vs] for k, vs in keep.items()},
        "extend": {
            k: {
                "direction": info["direction"],
                "suggested_new_values": [
                    _jsonify(v) for v in info["suggested_new_values"]
                ],
            }
            for k, info in extend.items()
        },
        "focus_regions": [
            {k: _jsonify(v) for k, v in region.items()} for region in focus_regions
        ],
    }


def write_suggested_grid_json(grid: dict, out_path: Path) -> None:
    out_path.write_text(json.dumps(grid, indent=2, sort_keys=True))


# ── markdown report ───────────────────────────────────────────────────────────


def _df_to_md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False, floatfmt=".4f")


def write_markdown_report(
    df: pd.DataFrame,
    varying_axes: list[str],
    fixed_axes: dict[str, object],
    uninformative_axes: list[str],
    univariate_stats: dict[str, pd.DataFrame],
    univariate_tests: dict[str, dict],
    importance_df: pd.DataFrame,
    surrogate_metrics: dict,
    interactions_ranked: pd.DataFrame,
    interaction_top_k: int,
    edge_flags: list[dict],
    dominated: dict[str, list],
    suggested_grid: dict,
    out_path: Path,
    top_n: int,
    case: str,
) -> None:
    """Human-readable report. Every table referenced here also lives as a CSV
    under ``data/`` and every PNG lives under ``plots/`` — this report is a
    pointer, not the source of truth.
    """
    lines: list[str] = []
    lines.append(f"# Cluster Parameter Analysis — {case.capitalize()} Embedding")
    lines.append("")
    lines.append(
        "Companion data lives in `data/` (CSV/JSON), companion plots in `plots/`."
    )
    lines.append("")
    lines.append("## 1. Overview")
    lines.append("")
    n_total = len(df)
    n_scored = int(df["dbcv"].notna().sum())
    n_failed = n_total - n_scored
    lines.append(f"- Total rows (after dedup): **{n_total}**")
    lines.append(f"- Scored by pipeline (viable): **{n_scored}**")
    lines.append(f"- Pre-filtered (treated as quality=0): **{n_failed}**")
    lines.append("")
    lines.append("**Quality score formula**:")
    lines.append("")
    lines.append("```")
    lines.append("quality_score = dbcv_norm * silhouette_norm")
    lines.append("              * count_penalty * noise_penalty * shape_penalty")
    lines.append("```")
    lines.append("")
    lines.append("Pre-filtered rows have `dbcv = NaN` and receive `quality_score = 0`.")
    lines.append("")
    if fixed_axes:
        lines.append("### Fixed parameters (constant across all rows)")
        lines.append("")
        fixed_df = pd.DataFrame(
            [{"param": k, "value": v} for k, v in fixed_axes.items()]
        )
        lines.append(_df_to_md(fixed_df))
        lines.append("")
    if uninformative_axes:
        lines.append("### Uninformative axes (no effect on quality_score)")
        lines.append("")
        lines.append(
            "These axes were detected as fully redundant — every value gives "
            "identical metrics given the other axes. They were dropped before "
            "analysis, and the dataset was deduplicated on the remaining axes."
        )
        lines.append("")
        for a in uninformative_axes:
            lines.append(f"- `{a}`")
        lines.append("")

    lines.append(f"## 2. Top-{top_n} runs by quality_score")
    lines.append("")
    top_cols = [
        *varying_axes,
        "n_clusters",
        "noise_ratio",
        "dbcv",
        "silhouette",
        "count_penalty",
        "noise_penalty",
        "shape_penalty",
        "quality_score",
    ]
    top_df = df.sort_values("quality_score", ascending=False).head(top_n)[top_cols]
    lines.append(_df_to_md(top_df))
    lines.append("")
    lines.append("Full table → `data/runs.csv`. Viable-only → `data/runs_viable.csv`.")
    lines.append("")

    lines.append("## 3. Per-parameter univariate analysis")
    lines.append("")
    for axis in varying_axes:
        lines.append(f"### {axis}")
        lines.append("")
        lines.append(_df_to_md(univariate_stats[axis]))
        lines.append("")
        test = univariate_tests[axis]
        lines.append(
            f"Kruskal-Wallis: H = {test['kw_statistic']:.3f}, "
            f"p = {test['kw_pvalue']:.3g}, η² = {test['eta_squared']:.3f}"
        )
        lines.append("")
        lines.append(
            f"Tables: `data/univariate_{axis}.csv`, `data/dunn_{axis}.csv`. "
            f"Plots: `plots/boxplot_{axis}.png`, `plots/failure_rate_{axis}.png`."
        )
        lines.append("")
        lines.append(f"![boxplot](plots/boxplot_{axis}.png)")
        lines.append("")
        lines.append(f"![failure rate](plots/failure_rate_{axis}.png)")
        lines.append("")

    lines.append("## 4. Surrogate-model results")
    lines.append("")
    auc_mean = surrogate_metrics.get("classifier_cv_auc_mean")
    auc_std = surrogate_metrics.get("classifier_cv_auc_std")
    r2_mean = surrogate_metrics.get("regressor_cv_r2_mean")
    r2_std = surrogate_metrics.get("regressor_cv_r2_std")
    clf_n_splits = surrogate_metrics.get("classifier_cv_n_splits")
    stage2_ran = bool(surrogate_metrics.get("regressor_ran", False))
    lines.append("Two-stage model: a classifier predicts whether the pipeline "
                 "will produce a DBCV at all; a regressor then explains "
                 "quality_score on viable rows only. This decoupling avoids "
                 "the bimodal target that kept the single-stage R² low.")
    lines.append("")
    if auc_mean is None:
        lines.append(
            "- **Stage 1 — viability classifier (all rows)**: skipped "
            f"({surrogate_metrics.get('classifier_note') or 'no AUC available'})"
        )
    else:
        fold_str = f"{clf_n_splits}-fold" if clf_n_splits else "5-fold"
        lines.append(
            f"- **Stage 1 — viability classifier (all rows)**: "
            f"{fold_str} CV ROC-AUC = **{auc_mean:.3f}** (± {auc_std:.3f})"
        )
    if stage2_ran:
        lines.append(
            f"- **Stage 2 — quality regressor (viable rows only)**: "
            f"5-fold CV R² = **{r2_mean:.3f}** (± {r2_std:.3f})"
        )
    else:
        lines.append(
            "- **Stage 2 — quality regressor (viable rows only)**: skipped "
            f"({surrogate_metrics.get('n_viable_rows', 0)} viable rows; "
            "needs ≥ 20)"
        )
    lines.append("")
    if stage2_ran:
        lines.append(
            "Raw metrics: `data/surrogate_metrics.json`. "
            "Feature importance: `data/feature_importance.csv`. "
            "Partial dependence values: `data/pdp_<axis>.csv`."
        )
        lines.append("")
        lines.append("![feature importance](plots/feature_importance.png)")
        lines.append("")
        for axis in varying_axes:
            lines.append(f"![PDP {axis}](plots/pdp_{axis}.png)")
        lines.append("")
    else:
        lines.append(
            "Raw metrics: `data/surrogate_metrics.json`. "
            "Feature importance and PDP artifacts are not produced when "
            "stage 2 is skipped."
        )
        lines.append("")

    lines.append("## 5. Interactions")
    lines.append("")
    lines.append(
        "Strength of non-additive interaction per pair, measured as the RMS "
        "of (cell mean − additive prediction from main effects). Zero means "
        "the pair's behavior is fully explained by its marginals."
    )
    lines.append("")
    if not interactions_ranked.empty:
        lines.append(_df_to_md(interactions_ranked))
        lines.append("")
        lines.append(
            "Full ranking: `data/interactions_ranked.csv`. "
            "Per-pair pivot tables: `data/interaction_<a>_x_<b>.csv`."
        )
        lines.append("")
        lines.append(f"Heatmaps for the top-{interaction_top_k} pairs:")
        lines.append("")
        for _, row in interactions_ranked.head(interaction_top_k).iterrows():
            a, b = row["axis_a"], row["axis_b"]
            lines.append(f"![{a} × {b}](plots/interaction_{a}_x_{b}.png)")
        lines.append("")

    lines.append("## 6. Boundary diagnostics")
    lines.append("")
    if edge_flags:
        edge_df = pd.DataFrame(edge_flags)
        lines.append(_df_to_md(edge_df))
    else:
        lines.append("(no axes flagged — optima are interior)")
    lines.append("")
    lines.append("Table → `data/boundary.csv`.")
    lines.append("")

    lines.append("## 7. Dominance pruning (viable rows only)")
    lines.append("")
    dom_rows = [
        {"axis": k, "dominated_values": ", ".join(map(str, vs))}
        for k, vs in dominated.items()
        if vs
    ]
    if dom_rows:
        lines.append(_df_to_md(pd.DataFrame(dom_rows)))
    else:
        lines.append("(no values strictly dominated)")
    lines.append("")
    lines.append("Table → `data/dominance.csv`.")
    lines.append("")

    lines.append("## 8. Recommendations")
    lines.append("")
    if suggested_grid["drop"]:
        for axis, vs in suggested_grid["drop"].items():
            lines.append(f"- Drop `{axis}` ∈ {vs}")
    if suggested_grid["extend"]:
        for axis, info in suggested_grid["extend"].items():
            lines.append(
                f"- Extend `{axis}` {info['direction']}: try "
                f"{info['suggested_new_values']}"
            )
    if suggested_grid["focus_regions"]:
        for region in suggested_grid["focus_regions"]:
            lines.append(f"- Focus region: {region}")
    if not (
        suggested_grid["drop"]
        or suggested_grid["extend"]
        or suggested_grid["focus_regions"]
    ):
        lines.append("(no actionable recommendations)")
    lines.append("")
    lines.append("Machine-readable: `suggested_grid.json`.")

    out_path.write_text("\n".join(lines))


# ── main ──────────────────────────────────────────────────────────────────────


ORDINAL_AXES = (
    "umap_n_components",
    "umap_n_neighbors",
    "umap_min_dist",
    "hdbscan_min_cluster_size",
)


INTERACTION_TOP_K = 6
STALE_ROOT_GLOBS = ("boxplot_*.png", "failure_rate_*.png", "pdp_*.png",
                    "ice_*.png", "interaction_*.png", "feature_importance.png",
                    "quality_distribution.png")


def _clean_stale_outputs(out_dir: Path) -> None:
    """Remove PNGs left at the output-dir root by the previous layout.

    The current layout keeps PNGs under ``plots/`` and structured data under
    ``data/``. Orphan root-level PNGs from older runs would otherwise confuse
    readers.
    """
    for pattern in STALE_ROOT_GLOBS:
        for p in out_dir.glob(pattern):
            if p.is_file():
                p.unlink()


def _save_axes_classification(
    path: Path, varying: list[str], fixed: dict, uninformative: list[str]
) -> None:
    path.write_text(json.dumps(
        {
            "varying": list(varying),
            "fixed": {k: _jsonify(v) for k, v in fixed.items()},
            "uninformative": list(uninformative),
        },
        indent=2, sort_keys=True,
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=DEFAULT_DB, help="Path to legacy SQLite DB")
    parser.add_argument("--case", default=DEFAULT_CASE, help="embedding_case to analyze")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    plots_dir = out_dir / "plots"
    data_dir = out_dir / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)
    _clean_stale_outputs(out_dir)

    print(f"Loading {args.case} runs from {args.db} …")
    df = load_runs(args.db, args.case)
    df = compute_quality_score(df)
    n_raw = len(df)
    print(f"  {n_raw} raw rows ({int(df['dbcv'].notna().sum())} scored)")

    varying, fixed = detect_varying_axes(df, list(ALL_PARAM_COLS))
    uninformative = detect_uninformative_axes(df, varying)
    if uninformative:
        print(f"  uninformative axes (zero effect on quality): {uninformative}")
        varying = [a for a in varying if a not in uninformative]
        df = dedupe_on_axes(df, varying)
        print(f"  {len(df)} unique rows after dedup")
    print(f"  varying axes: {varying}")
    print(f"  fixed: {list(fixed.keys())}")

    _save_axes_classification(data_dir / "axes.json", varying, fixed, uninformative)
    df.to_csv(data_dir / "runs.csv", index=False)
    viable = df[df["dbcv"].notna()].copy().reset_index(drop=True)
    viable.to_csv(data_dir / "runs_viable.csv", index=False)
    print(f"  viable rows for regressor / interactions / dominance: {len(viable)}")

    print("Computing univariate stats + plots …")
    uni_stats: dict[str, pd.DataFrame] = {}
    uni_tests: dict[str, dict] = {}
    kw_rows: list[dict] = []
    for axis in varying:
        stats = per_param_stats(df, axis)
        uni_stats[axis] = stats
        stats.to_csv(data_dir / f"univariate_{axis}.csv", index=False)

        test = kruskal_dunn(df, axis)
        uni_tests[axis] = test
        if isinstance(test["dunn_pvalues"], pd.DataFrame) and not test["dunn_pvalues"].empty:
            test["dunn_pvalues"].to_csv(data_dir / f"dunn_{axis}.csv")
        kw_rows.append({
            "axis": axis,
            "kw_statistic": test["kw_statistic"],
            "kw_pvalue": test["kw_pvalue"],
            "eta_squared": test["eta_squared"],
        })
        plot_boxplot(df, axis, plots_dir / f"boxplot_{axis}.png")
        plot_failure_rate(df, axis, plots_dir / f"failure_rate_{axis}.png")
    pd.DataFrame(kw_rows).to_csv(data_dir / "kruskal_summary.csv", index=False)

    print("Stage 1: viability classifier (all rows) …")
    _clf, clf_info = fit_classifier(df, varying)
    if not math.isnan(clf_info["cv_auc_mean"]):
        print(f"  CV ROC-AUC = {clf_info['cv_auc_mean']:.3f} "
              f"(± {clf_info['cv_auc_std']:.3f})")

    importance = pd.DataFrame()
    reg_info: dict = {"cv_r2": float("nan"), "cv_r2_std": float("nan")}
    top3: list[str] = []
    stage2_ran = False
    if len(viable) >= 20:
        print("Stage 2: quality regressor (viable rows only) …")
        reg, reg_info = fit_surrogate(viable, varying)
        stage2_ran = True
        print(f"  CV R² (viable) = {reg_info['cv_r2']:.3f} "
              f"(± {reg_info['cv_r2_std']:.3f})")

        importance = permutation_importance_report(reg, reg_info)
        importance.to_csv(data_dir / "feature_importance.csv", index=False)
        plot_feature_importance(importance, plots_dir / "feature_importance.png")

        print("Plotting PDP + ICE …")
        for axis in varying:
            values = sorted(
                viable[axis].dropna().unique(),
                key=lambda v: (isinstance(v, str), v),
            )
            means = []
            for v in values:
                copy = viable.copy()
                copy[axis] = v
                preds = reg_info["predict"](copy)
                means.append(float(preds.mean()))
            pd.DataFrame({
                "value": values,
                "mean_predicted_quality": means,
            }).to_csv(data_dir / f"pdp_{axis}.csv", index=False)
            plot_partial_dependence(reg_info, viable, axis, plots_dir / f"pdp_{axis}.png")

        top3 = (
            importance.groupby("base_param")["importance_mean"]
            .sum()
            .sort_values(ascending=False)
            .head(3)
            .index.tolist()
        )
        for axis in top3:
            plot_ice(reg_info, viable, axis, plots_dir / f"ice_{axis}.png")
    else:
        print("  too few viable rows for a stable regressor — skipping stage 2")
        # Remove stale stage-2 artifacts from a prior run on this output dir so
        # the on-disk view stays consistent with the report (which now omits
        # these links when stage 2 is skipped).
        for stale in [
            data_dir / "feature_importance.csv",
            plots_dir / "feature_importance.png",
            *data_dir.glob("pdp_*.csv"),
            *plots_dir.glob("pdp_*.png"),
            *plots_dir.glob("ice_*.png"),
        ]:
            stale.unlink(missing_ok=True)

    def _json_safe(value: float) -> float | None:
        return None if isinstance(value, float) and math.isnan(value) else value

    surrogate_metrics = {
        "classifier_cv_auc_mean": _json_safe(clf_info.get("cv_auc_mean", float("nan"))),
        "classifier_cv_auc_std": _json_safe(clf_info.get("cv_auc_std", float("nan"))),
        "classifier_cv_n_splits": clf_info.get("cv_n_splits"),
        "classifier_note": clf_info.get("note"),
        "regressor_ran": stage2_ran,
        "regressor_cv_r2_mean": _json_safe(reg_info["cv_r2"]) if stage2_ran else None,
        "regressor_cv_r2_std": _json_safe(reg_info["cv_r2_std"]) if stage2_ran else None,
        "n_total_rows": len(df),
        "n_viable_rows": len(viable),
        "n_raw_rows_before_dedup": int(n_raw),
    }
    (data_dir / "surrogate_metrics.json").write_text(
        json.dumps(surrogate_metrics, indent=2, sort_keys=True, allow_nan=False)
    )

    print("Ranking interactions across all axis pairs …")
    ranked = interaction_ranked(viable, varying)
    ranked.to_csv(data_dir / "interactions_ranked.csv", index=False)
    if not ranked.empty:
        for _, row in ranked.head(INTERACTION_TOP_K).iterrows():
            a, b = row["axis_a"], row["axis_b"]
            _rms, pivot = interaction_strength(viable, a, b)
            if not pivot.empty:
                pivot.to_csv(data_dir / f"interaction_{a}_x_{b}.csv")
                plot_interaction_heatmap(
                    viable, a, b, plots_dir / f"interaction_{a}_x_{b}.png"
                )

    print("Computing boundary + dominance …")
    edges = detect_edge_optima(df, [a for a in ORDINAL_AXES if a in varying])
    if edges:
        pd.DataFrame(edges).to_csv(data_dir / "boundary.csv", index=False)
    else:
        pd.DataFrame(columns=["axis", "direction", "best_value"]).to_csv(
            data_dir / "boundary.csv", index=False
        )

    # Dominance on viable rows only — pre-filtered ties at quality=0 were
    # masking real signal.
    dominated = dominance_analysis(viable, varying) if len(viable) else {}
    dom_rows = [
        {"axis": k, "dominated_value": v}
        for k, vs in dominated.items() for v in vs
    ]
    if dom_rows:
        pd.DataFrame(dom_rows).to_csv(data_dir / "dominance.csv", index=False)
    else:
        pd.DataFrame(columns=["axis", "dominated_value"]).to_csv(
            data_dir / "dominance.csv", index=False
        )

    top_pair = (top3[0], top3[1]) if len(top3) >= 2 else None
    grid = build_suggested_grid(
        viable if len(viable) else df,
        varying_axes=varying,
        ordinal_axes=[a for a in ORDINAL_AXES if a in varying],
        top_pair=top_pair,
    )
    write_suggested_grid_json(grid, out_dir / "suggested_grid.json")

    # Quality distribution plot + CSV
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    counts, bin_edges, _ = ax.hist(
        df["quality_score"], bins=40, color="#6aa6ff", edgecolor="white"
    )
    ax.set_title("quality_score distribution")
    ax.set_xlabel("quality_score")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(plots_dir / "quality_distribution.png", dpi=300)
    plt.close(fig)
    pd.DataFrame({
        "bin_left": bin_edges[:-1],
        "bin_right": bin_edges[1:],
        "count": counts.astype(int),
    }).to_csv(data_dir / "quality_distribution.csv", index=False)

    write_markdown_report(
        df=df,
        varying_axes=varying,
        fixed_axes=fixed,
        uninformative_axes=uninformative,
        univariate_stats=uni_stats,
        univariate_tests=uni_tests,
        importance_df=importance,
        surrogate_metrics=surrogate_metrics,
        interactions_ranked=ranked,
        interaction_top_k=INTERACTION_TOP_K,
        edge_flags=edges,
        dominated=dominated,
        suggested_grid=grid,
        out_path=out_dir / "report.md",
        top_n=args.top_n,
        case=args.case,
    )

    # Stdout summary
    print()
    print("=" * 60)
    print(f"Top {min(5, args.top_n)} runs:")
    top5 = df.sort_values("quality_score", ascending=False).head(5)
    print(top5[[*varying, "quality_score", "dbcv", "n_clusters", "noise_ratio"]].to_string(index=False))
    print()
    if grid["drop"]:
        print("Recommended drops:", grid["drop"])
    if grid["extend"]:
        print("Recommended extensions:", grid["extend"])
    print()
    print(f"Full report → {out_dir / 'report.md'}")
    print(f"Structured data → {data_dir}/")
    print(f"Plots → {plots_dir}/")


if __name__ == "__main__":
    main()
