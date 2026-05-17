"""Read-only helpers for cluster-run reporting and best-run selection."""

from __future__ import annotations

import math
from statistics import median
from types import SimpleNamespace

from sqlalchemy.orm import Session

from core.database import ClusterRun

DEFAULT_CASES: tuple[str, ...] = ("audio", "video", "sandwich")


def get_plateau_drop_threshold(settings: SimpleNamespace | None = None) -> float:
    """Get plateau drop threshold from settings, or use hardcoded default."""
    if settings is None or not hasattr(settings, "plateau_drop_threshold"):
        return 0.05
    return float(settings.plateau_drop_threshold)


def list_case_rows(session: Session, case: str) -> list[ClusterRun]:
    return session.query(ClusterRun).filter(ClusterRun.embedding_case == case).all()


def list_best_candidate_rows(session: Session, case: str) -> list[ClusterRun]:
    return (
        session.query(ClusterRun)
        .filter(
            ClusterRun.embedding_case == case,
            ClusterRun.passes_validation.is_(True),
            ClusterRun.dbcv.isnot(None),
            ClusterRun.param_plateau_score.isnot(None),
        )
        .all()
    )


def pick_best_cluster_run(
    rows: list[ClusterRun],
    threshold: float | None = None,
    settings: SimpleNamespace | None = None,
) -> ClusterRun | None:
    if not rows:
        return None
    t = get_plateau_drop_threshold(settings) if threshold is None else threshold
    survivors = [r for r in rows if (r.dbcv - r.param_plateau_score) <= t]  # type: ignore[operator]
    pool = survivors if survivors else rows
    return max(pool, key=lambda r: r.dbcv if r.dbcv is not None else 0.0)  # type: ignore[arg-type]


def select_best_cluster_run(
    session: Session,
    case: str,
    threshold: float | None = None,
    settings: SimpleNamespace | None = None,
) -> ClusterRun | None:
    rows = list_best_candidate_rows(session, case)
    return pick_best_cluster_run(rows, threshold=threshold, settings=settings)


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _std_ddof1(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    var = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
    return math.sqrt(var)


def summarize_case_rows(rows: list[ClusterRun]) -> dict[str, str]:
    filtered_rows = [r for r in rows if r.passes_validation is True]
    dbcv_vals = [
        float(r.dbcv)
        for r in rows
        if r.dbcv is not None and math.isfinite(float(r.dbcv))
    ]
    sil_vals = [
        float(r.silhouette)
        for r in rows
        if r.silhouette is not None and math.isfinite(float(r.silhouette))
    ]
    k_vals = [float(r.n_clusters) for r in rows if r.n_clusters is not None]
    noise_pct_vals = [
        float(r.noise_ratio) * 100.0
        for r in rows
        if r.noise_ratio is not None and math.isfinite(float(r.noise_ratio))
    ]
    k_median = median(k_vals) if k_vals else 0.0
    return {
        "n_runs": str(len(rows)),
        "n_filtered": str(len(filtered_rows)),
        "dbcv_mean": f"{_mean(dbcv_vals):.3f}",
        "dbcv_std": f"{_std_ddof1(dbcv_vals):.3f}",
        "silhouette_mean": f"{_mean(sil_vals):.3f}",
        "k_median": f"{k_median:.3f}",
        "noise_pct_mean": f"{_mean(noise_pct_vals):.3f}",
    }
