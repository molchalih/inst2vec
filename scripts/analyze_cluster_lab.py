"""Cluster lab analyzer.

Reads data/cluster_testing.db (populated by run_cluster_lab.py and
run_stability.py) and emits scripts/output/cluster_lab/report.md plus
companion CSV/PNG files in data/ and plots/.
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── quality score (legacy formula) ────────────────────────────────────────────


def _logistic_window(x: float, low: float, high: float, edge: float) -> float:
    if edge <= 0:
        return 1.0 if low <= x <= high else 0.0
    low_t = 1.0 / (1.0 + math.exp(-(x - low) / edge))
    high_t = 1.0 / (1.0 + math.exp(-(high - x) / edge))
    return low_t * high_t


def _count_penalty(n: int) -> float:
    return _logistic_window(float(n), 5.0, 60.0, 1.5)


def _noise_penalty(nr: float) -> float:
    return _logistic_window(float(nr), 0.02, 0.55, 0.01)


def _shape_penalty(mn: int, md: int, mx: int) -> float:
    if md <= 0:
        return 0.0
    lr = float(mn) / float(md)
    hr = float(mx) / float(md)
    low = _logistic_window(lr, 0.2, 10.0, 0.1)
    z = (hr - 5.0) / 0.3
    high = math.exp(-z) / (1.0 + math.exp(-z)) if z >= 0 else 1.0 / (1.0 + math.exp(z))
    return low * high


def add_quality_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dbcv_norm"] = (out["dbcv"].fillna(-1.0) + 1.0) / 2.0
    out["silhouette_norm"] = (out["silhouette"].fillna(-1.0) + 1.0) / 2.0
    out["count_penalty"] = out["n_clusters"].apply(
        lambda x: _count_penalty(int(x)) if pd.notna(x) else 0.0
    )
    out["noise_penalty"] = out["noise_ratio"].apply(
        lambda x: _noise_penalty(float(x)) if pd.notna(x) else 0.0
    )
    out["shape_penalty"] = out.apply(
        lambda r: _shape_penalty(
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
    out["quality_score"] = raw.where(out["dbcv"].notna(), 0.0)
    return out


# ── DB helpers ────────────────────────────────────────────────────────────────


def load_runs(db_path: str) -> pd.DataFrame:
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        return pd.read_sql_query("SELECT * FROM cluster_runs", conn)


def load_stability(db_path: str) -> pd.DataFrame:
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        try:
            return pd.read_sql_query("SELECT * FROM stability", conn)
        except Exception:
            return pd.DataFrame()


def load_cross_method(db_path: str) -> pd.DataFrame:
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        try:
            return pd.read_sql_query("SELECT * FROM cross_method", conn)
        except Exception:
            return pd.DataFrame()


# ── plotting ──────────────────────────────────────────────────────────────────


def plot_algo_violin(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sub = df.dropna(subset=["silhouette"]).copy()
    sub["bucket"] = sub["algorithm"] + "/" + sub["reducer"]
    order = (
        sub.groupby("bucket")["silhouette"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    sns.violinplot(
        data=sub,
        x="bucket",
        y="silhouette",
        order=order,
        cut=0,
        inner="quartile",
        ax=ax,
    )
    ax.set_title("Silhouette by (algorithm / reducer)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_min_samples_box(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sub = df[
        (df["algorithm"] == "hdbscan")
        & (df["reducer"] == "umap")
        & (df["hdbscan_min_samples"].notna())
        & (df["silhouette"].notna())
        & (df["n_clusters"] >= 5)
    ].copy()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    sns.boxplot(data=sub, x="hdbscan_min_samples", y="silhouette", ax=ax)
    ax.set_title("Silhouette by hdbscan_min_samples (viable runs, n_clusters≥5)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_stability_bar(stab: pd.DataFrame, out_path: Path) -> None:
    if stab.empty:
        return
    import matplotlib.pyplot as plt

    short = stab["config_key"].str[:10]
    fig, ax = plt.subplots(figsize=(9, 4), dpi=150)
    ax.bar(short, stab["mean_ari"], yerr=stab["std_ari"], color="#6aa6ff", label="ARI")
    ax.bar(
        short,
        stab["mean_nmi"],
        color="none",
        edgecolor="#d96666",
        linewidth=1.5,
        label="NMI",
    )
    ax.set_ylim(0, 1)
    ax.set_xlabel("config_key (first 10)")
    ax.set_ylabel("ARI / NMI across seeds")
    ax.set_title("Seed stability (top viable HDBSCAN+UMAP configs)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_cross_method_heatmap(out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    ari_path = out_dir / "data" / "cross_method_ari.csv"
    nmi_path = out_dir / "data" / "cross_method_nmi.csv"
    if not ari_path.exists():
        return
    ari = pd.read_csv(ari_path, index_col=0)
    nmi = pd.read_csv(nmi_path, index_col=0)
    for matrix, name, cmap in [
        (ari, "cross_method_ari", "viridis"),
        (nmi, "cross_method_nmi", "magma"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 7), dpi=150)
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap=cmap, ax=ax, vmin=0, vmax=1)
        ax.set_title(f"{name} — top configs per algorithm")
        fig.tight_layout()
        fig.savefig(out_dir / "plots" / f"{name}.png", dpi=300)
        plt.close(fig)


def plot_metric_correlation(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sub = df[df["silhouette"].notna()][
        ["silhouette", "calinski_harabasz", "davies_bouldin", "dbcv"]
    ].copy()
    sub = sub.dropna(how="any")
    if sub.shape[0] < 30:
        return
    corr = sub.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Internal metric correlation (Spearman)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# ── report writer ─────────────────────────────────────────────────────────────


def _df_to_md(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    return df.to_markdown(index=False, floatfmt=".4f")


PRESENTATION_COLS = (
    "algorithm",
    "reducer",
    "k",
    "umap_n_components",
    "umap_n_neighbors",
    "umap_min_dist",
    "umap_metric",
    "pca_n_components",
    "hdbscan_min_cluster_size",
    "hdbscan_min_samples",
    "hdbscan_cluster_selection_method",
    "hdbscan_metric",
    "covariance_type",
    "linkage",
    "distance_metric",
    "affinity",
    "n_neighbors",
    "n_clusters",
    "noise_ratio",
    "silhouette",
    "dbcv",
    "calinski_harabasz",
    "davies_bouldin",
)


def _slim(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in PRESENTATION_COLS if c in df.columns]
    return df[cols]


def build_report(
    df: pd.DataFrame, stab: pd.DataFrame, cross: pd.DataFrame, out_dir: Path
) -> str:
    lines: list[str] = []
    lines.append("# Cluster Lab — Sandwich Embedding")
    lines.append("")
    lines.append(
        "Stress-test on the sandwich case: extended HDBSCAN+UMAP grid, "
        "alternative algorithms (KMeans, GMM, Agglomerative, Spectral, "
        "PCA+HDBSCAN, direct HDBSCAN), seed stability, and cross-method ARI. "
        "All companion data lives in `data/`, plots in `plots/`."
    )
    lines.append("")

    # 1. Overview
    lines.append("## 1. Overview")
    lines.append("")
    summary = (
        df.groupby(["algorithm", "reducer"])
        .agg(
            n_rows=("config_hash", "count"),
            n_viable=("silhouette", lambda s: int(s.notna().sum())),
            best_silhouette=("silhouette", "max"),
        )
        .reset_index()
        .sort_values("n_rows", ascending=False)
    )
    lines.append(_df_to_md(summary))
    lines.append("")
    lines.append(f"- Total rows: **{len(df)}**")
    lines.append(
        f"- Viable (have silhouette): **{int(df['silhouette'].notna().sum())}**"
    )
    lines.append("")

    # 2. Top-20 across whole DB by silhouette + n_clusters≥5
    lines.append("## 2. Top-20 configs by silhouette (n_clusters ≥ 5)")
    lines.append("")
    top_overall = (
        df[(df["silhouette"].notna()) & (df["n_clusters"] >= 5)]
        .sort_values("silhouette", ascending=False)
        .head(20)
    )
    lines.append(_df_to_md(_slim(top_overall)))
    lines.append("")
    lines.append("Full table → `data/top20_silhouette.csv`.")
    top_overall.to_csv(out_dir / "data" / "top20_silhouette.csv", index=False)
    lines.append("")

    # 3. Top-20 by legacy quality_score (HDBSCAN-only — formula assumes noise/sizes)
    lines.append("## 3. Top-20 HDBSCAN configs by legacy quality_score")
    lines.append("")
    hdb = df[df["algorithm"] == "hdbscan"].copy()
    hdb = add_quality_score(hdb)
    top_quality = hdb.sort_values("quality_score", ascending=False).head(20)
    show_cols = [
        "reducer",
        "umap_n_components",
        "umap_n_neighbors",
        "umap_min_dist",
        "umap_metric",
        "pca_n_components",
        "hdbscan_min_cluster_size",
        "hdbscan_min_samples",
        "hdbscan_cluster_selection_method",
        "hdbscan_metric",
        "n_clusters",
        "noise_ratio",
        "dbcv",
        "silhouette",
        "quality_score",
    ]
    show_cols = [c for c in show_cols if c in top_quality.columns]
    lines.append(_df_to_md(top_quality[show_cols]))
    lines.append("")
    hdb.to_csv(out_dir / "data" / "hdbscan_with_quality.csv", index=False)
    legacy_best = 0.5911  # from prior report.md
    top_now = float(top_quality["quality_score"].max())
    lines.append(
        f"Best new quality_score = **{top_now:.4f}** (legacy report best: {legacy_best})."
    )
    lines.append("")

    # 4. Per-algorithm top-5
    lines.append("## 4. Per-algorithm / per-reducer top-5")
    lines.append("")
    for (algo, reducer), group in df.groupby(["algorithm", "reducer"]):
        sub = (
            group[group["silhouette"].notna()]
            .sort_values("silhouette", ascending=False)
            .head(5)
        )
        if sub.empty:
            continue
        lines.append(f"### {algo} / {reducer}")
        lines.append("")
        lines.append(_df_to_md(_slim(sub)))
        lines.append("")

    # 5. min_samples sweep
    lines.append("## 5. New axis: `hdbscan_min_samples`")
    lines.append("")
    ms_sub = df[
        (df["algorithm"] == "hdbscan")
        & (df["reducer"] == "umap")
        & (df["hdbscan_min_samples"].notna())
        & (df["silhouette"].notna())
        & (df["n_clusters"] >= 5)
    ]
    if not ms_sub.empty:
        stats = (
            ms_sub.groupby("hdbscan_min_samples")
            .agg(
                n_rows=("config_hash", "count"),
                silhouette_mean=("silhouette", "mean"),
                silhouette_median=("silhouette", "median"),
                silhouette_max=("silhouette", "max"),
                dbcv_mean=("dbcv", "mean"),
                n_clusters_median=("n_clusters", "median"),
                noise_ratio_mean=("noise_ratio", "mean"),
            )
            .reset_index()
            .sort_values("hdbscan_min_samples")
        )
        lines.append(_df_to_md(stats))
        lines.append("")
        lines.append("![min_samples boxplot](plots/min_samples_boxplot.png)")
        lines.append("")
    else:
        lines.append("(no min_samples sweep data found)")
        lines.append("")

    # 6. Seed stability
    lines.append("## 6. Seed stability")
    lines.append("")
    if not stab.empty:
        s = stab.copy()
        s["config_key"] = s["config_key"].str.slice(0, 12)
        lines.append(_df_to_md(s))
        lines.append("")
        lines.append("![stability bar](plots/stability_bar.png)")
        lines.append("")
        lines.append(
            f"Mean ARI across these {len(s)} configs: "
            f"**{s['mean_ari'].mean():.3f} ± {s['mean_ari'].std():.3f}**. "
            f"Mean NMI: **{s['mean_nmi'].mean():.3f}**. "
            f"NMI > ARI is normal: cluster boundaries shift across seeds, "
            f"but overall partition structure is preserved."
        )
        lines.append("")
    else:
        lines.append("(stability table empty)")
        lines.append("")

    # 7. Cross-method ARI
    lines.append("## 7. Cross-method agreement")
    lines.append("")
    if not cross.empty:
        lines.append(
            f"Pairwise ARI / NMI between top-K configs per algorithm ({len(cross)} pairs)."
        )
        top_pairs = cross.sort_values("ari", ascending=False).head(10)
        lines.append(_df_to_md(top_pairs))
        lines.append("")
        lines.append("![cross-method ARI](plots/cross_method_ari.png)")
        lines.append("")
        lines.append("![cross-method NMI](plots/cross_method_nmi.png)")
        lines.append("")
    else:
        lines.append("(cross_method table empty)")
        lines.append("")

    # 8. Metric correlations
    lines.append("## 8. Internal metric correlations")
    lines.append("")
    lines.append(
        "Spearman correlation between the four internal metrics on viable runs. "
        "Highly correlated metrics are giving the same signal; weak correlations "
        "suggest one of them is exposing real geometry the others miss."
    )
    lines.append("")
    lines.append("![metric correlation](plots/metric_correlation.png)")
    lines.append("")

    # 9. Algorithm comparison
    lines.append("## 9. Algorithm comparison")
    lines.append("")
    lines.append("![silhouette by algorithm](plots/algo_silhouette_violin.png)")
    lines.append("")

    # 10. Recommendations
    lines.append("## 10. Recommendations")
    lines.append("")
    best_quality_row = top_quality.iloc[0].to_dict()
    best_qual_cfg_str = "; ".join(
        f"{k}={v}"
        for k, v in best_quality_row.items()
        if k in show_cols
        and pd.notna(v)
        and k
        not in ("quality_score", "n_clusters", "noise_ratio", "dbcv", "silhouette")
    )
    lines.append(
        f"- **Best HDBSCAN by quality_score (n={top_quality.iloc[0]['n_clusters']:.0f}, "
        f"q={top_quality.iloc[0]['quality_score']:.4f})**: {best_qual_cfg_str}"
    )
    if not stab.empty:
        best_stab = stab.sort_values("mean_nmi", ascending=False).iloc[0]
        lines.append(
            f"- **Most stable seed config (NMI={best_stab['mean_nmi']:.3f}, "
            f"median n_clusters={best_stab['median_n_clusters']:.1f})**: "
            f"config_key={best_stab['config_key'][:12]}…"
        )
    # Compare alt-algorithm best vs HDBSCAN-UMAP best
    alt_best = (
        df[
            (df["algorithm"] != "hdbscan")
            & (df["silhouette"].notna())
            & (df["n_clusters"] >= 5)
        ]
        .sort_values("silhouette", ascending=False)
        .head(1)
    )
    if not alt_best.empty:
        r = alt_best.iloc[0]
        lines.append(
            f"- **Best non-HDBSCAN baseline**: {r['algorithm']}/{r['reducer']} "
            f"silhouette={r['silhouette']:.4f}, k={r['k'] if pd.notna(r.get('k')) else '?'}, "
            f"n_clusters={r['n_clusters']:.0f}"
        )
    lines.append("")
    lines.append(
        "**Conceptual note**: silhouette favours degenerate (1 big cluster + "
        "noise) solutions because it rewards compactness; we explicitly filter "
        "`n_clusters ≥ 5` to compare like-with-like. The legacy `quality_score` "
        "formula (multiplicative DBCV·silhouette·size·noise penalties) already "
        "guards against this and is the recommended ranker."
    )
    lines.append("")
    lines.append(
        "**Suggested next experiment** (not run here): re-cluster with the "
        "highest-NMI stability config across all 8 seeds, then aggregate via "
        "ensemble (co-association matrix → final clustering) to get a single "
        "robust assignment that is provably stable on this embedding."
    )

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="data/cluster_testing.db")
    ap.add_argument("--output-dir", default="scripts/output/cluster_lab")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.db} …")
    df = load_runs(args.db)
    stab = load_stability(args.db)
    cross = load_cross_method(args.db)
    print(
        f"  {len(df)} runs, {len(stab)} stability entries, {len(cross)} cross-method pairs"
    )

    print("Plotting …")
    plot_algo_violin(df, out_dir / "plots" / "algo_silhouette_violin.png")
    plot_min_samples_box(df, out_dir / "plots" / "min_samples_boxplot.png")
    plot_stability_bar(stab, out_dir / "plots" / "stability_bar.png")
    plot_cross_method_heatmap(out_dir)
    plot_metric_correlation(df, out_dir / "plots" / "metric_correlation.png")

    print("Writing report …")
    report = build_report(df, stab, cross, out_dir)
    (out_dir / "report.md").write_text(report)

    # Dump raw runs as CSV for convenience
    df.to_csv(out_dir / "data" / "all_runs.csv", index=False)
    if not stab.empty:
        stab.to_csv(out_dir / "data" / "stability_dump.csv", index=False)
    if not cross.empty:
        cross.to_csv(out_dir / "data" / "cross_method_dump.csv", index=False)

    print(f"Done. Report → {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
