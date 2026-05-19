"""Offline analyzer for the legacy cluster_runs table.

Mines data/old/inst2vec.db (read-only) to recommend a smarter clustering grid
for future runs: drop dominated parameter values, extend edge-bound axes,
and focus on promising regions.

Standalone — does not import from core/ or modules/. Run via:

    uv run --group analysis python scripts/analyze_cluster_params.py
"""

from __future__ import annotations

import argparse  # noqa: F401
import json  # noqa: F401
import math
import sqlite3
from pathlib import Path

import numpy as np  # noqa: F401
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
    high_score = 1.0 / (1.0 + math.exp(-(5.0 - high_ratio) / 0.3))
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


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    raise NotImplementedError("filled in later tasks")


if __name__ == "__main__":
    main()
