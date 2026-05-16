"""Paper-facing summary of the validation-best cluster run per embedding case."""

from __future__ import annotations

import math

from sqlalchemy.orm import Session

from modules.clustering import (
    DEFAULT_CASES,
    select_best_cluster_run,
)
from modules.database import ClusterRun

__all__ = ("best_run_to_markdown", "best_runs_all_to_markdown")

BEST_TABLE_ROWS: tuple[tuple[str, str], ...] = (
    (r"$\mathrm{DBCV}^*$", "dbcv"),
    (r"$\mathrm{plateau}^*$", "plateau_score"),
    (r"$\Delta_{\mathrm{drop}}^*$", "drop"),
    (r"$\mathrm{sil}^*$", "silhouette"),
    (r"$k^*$", "n_clusters"),
    (r"$\mathrm{noise\_ratio}$", "noise_ratio"),
    (r"$n_{\mathrm{UMAP}}^*$", "umap_n_neighbors"),
    (r"$m_{\mathrm{HDBSCAN}}^*$", "hdbscan_min_cluster_size"),
)


def _fmt(val: float | None) -> str:
    if val is None or not math.isfinite(float(val)):
        return "-"
    return f"{float(val):.4f}"


def _best_cells(best: ClusterRun | None) -> dict[str, str]:
    keys = [k for _, k in BEST_TABLE_ROWS]
    if best is None:
        return dict.fromkeys(keys, "-")
    dbcv_val = best.dbcv if best.dbcv is not None else 0.0
    plateau_val = (
        best.param_plateau_score if best.param_plateau_score is not None else 0.0
    )
    drop = dbcv_val - plateau_val
    return {
        "dbcv": _fmt(best.dbcv),
        "plateau_score": _fmt(best.param_plateau_score),
        "drop": f"{drop:.4f}",
        "silhouette": _fmt(best.silhouette),
        "n_clusters": str(best.n_clusters),
        "noise_ratio": f"{best.noise_ratio:.1%}",
        "umap_n_neighbors": str(best.umap_n_neighbors),
        "hdbscan_min_cluster_size": str(best.hdbscan_min_cluster_size),
    }


def best_run_to_markdown(eng, case: str) -> str:
    if case not in DEFAULT_CASES:
        raise ValueError(f"unknown embedding case: {case}")

    with Session(eng) as session:
        best = select_best_cluster_run(session, case)
    if best is None:
        return f"No eligible cluster runs found for `{case}`."

    cells = _best_cells(best)
    lines = ["| Field | Value |", "|---|---:|"]
    for label, key in BEST_TABLE_ROWS:
        lines.append(f"| {label} | {cells[key]} |")
    return "\n".join(lines)


def best_runs_all_to_markdown(
    eng,
    *,
    cases: tuple[str, ...] = DEFAULT_CASES,
) -> str:
    if not cases:
        raise ValueError("cases must contain at least one embedding case")

    summaries: dict[str, dict[str, str]] = {}
    with Session(eng) as session:
        for case in cases:
            best = select_best_cluster_run(session, case)
            summaries[case] = _best_cells(best)

    col_header = " | ".join(cases)
    align_row = "|".join(["---:"] * len(cases))
    table_lines = [
        f"| Field | {col_header} |",
        f"|---|{align_row}|",
    ]
    for label, key in BEST_TABLE_ROWS:
        values = [summaries[case][key] for case in cases]
        table_lines.append(f"| {label} | {' | '.join(values)} |")
    table_lines.extend(
        [
            "",
            ": Best validated cluster run per embedding case.",
        ]
    )
    return "\n".join(table_lines)
