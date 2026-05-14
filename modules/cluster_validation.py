"""Phase 6b — clustering validation: filter, score, plateau, select."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import cast

import hdbscan.validity
import numpy as np
from sklearn.metrics import silhouette_score
from sqlalchemy import or_
from sqlalchemy.orm import Session

from modules.cluster_results import (
    list_eligible_best_rows,
    pick_best_cluster_run,
)
from modules.clustering import (
    DEFAULT_HDBSCAN_METRIC,
    compute_clusters,
    load_user_matrix,
    resolve_hdbscan_metric,
)
from modules.console import log, progress
from modules.database import ClusterRun, get_session
from modules.eligibility import Eligibility, eligibility_db, is_eligible

_PARAM_COLS = [
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
]

_NUMERIC_PARAM_COLS = [
    "umap_n_components",
    "umap_n_neighbors",
    "umap_min_dist",
    "umap2d_n_neighbors",
    "umap2d_min_dist",
    "hdbscan_min_cluster_size",
    "hdbscan_min_samples",
    "random_state",
]


def _compute_validation_config_hash(settings) -> str:
    config = {
        "max_noise": str(settings.max_noise_ratio),
        "min_clusters": str(settings.min_clusters),
        "max_clusters": str(settings.max_clusters),
        "plateau_drop_threshold": str(settings.plateau_drop_threshold),
    }
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]


def _invalidate_stale_rows(session: Session, case: str, current_hash: str) -> None:
    stale = (
        session.query(ClusterRun)
        .filter(
            ClusterRun.embedding_case == case,
            ClusterRun.in_current_grid,
            or_(
                ClusterRun.validation_config_hash.is_(None),
                ClusterRun.validation_config_hash != current_hash,
            ),
        )
        .all()
    )
    for row in stale:
        row.param_plateau_score = None
        row.validation_config_hash = current_hash
    if stale:
        session.commit()
        log(
            f"validate:{case}",
            f"invalidated {len(stale)} stale rows (config hash changed)",
        )


def _row_to_params(row: ClusterRun) -> dict:
    return {col: getattr(row, col) for col in _PARAM_COLS}


def _compute_row_scores(matrix: np.ndarray, params: dict) -> tuple[float, float] | str:
    """Run clustering and metrics for one param set. Returns (dbcv, silhouette) or error token."""
    try:
        result = compute_clusters(matrix, return_nd_matrix=True, **params)
    except ValueError:
        return "value_error"

    if result.matrix_nd is None:
        return "value_error"
    X_nd = result.matrix_nd.astype(np.float64)
    labels = result.labels
    validation_metric = resolve_hdbscan_metric(
        params.get("hdbscan_metric", DEFAULT_HDBSCAN_METRIC)
    )

    try:
        _vi_result = hdbscan.validity.validity_index(
            X_nd, labels, metric=validation_metric
        )
        dbcv = float(cast(float, _vi_result))
    except Exception:
        return "dbcv_fail"

    non_noise = labels != -1
    unique_clusters = np.unique(labels[non_noise])
    if len(unique_clusters) >= 2:
        try:
            sil = float(
                silhouette_score(
                    X_nd[non_noise],
                    labels[non_noise],
                    metric=validation_metric,
                )
            )
        except Exception:
            sil = 0.0
    else:
        sil = 0.0

    return dbcv, sil


def _phase_filter(session: Session, case: str, settings) -> None:
    max_noise = float(settings.max_noise_ratio)
    min_clusters = int(settings.min_clusters)
    max_clusters = int(settings.max_clusters)

    rows = (
        session.query(ClusterRun)
        .filter(
            ClusterRun.embedding_case == case,
            ClusterRun.in_current_grid,
        )
        .all()
    )
    n_pass = 0
    if rows:
        with progress(len(rows), f"validate filter · {case}") as advance:
            for row in rows:
                passes = (
                    row.noise_ratio <= max_noise
                    and min_clusters <= row.n_clusters <= max_clusters
                )
                row.eligibility = (
                    eligibility_db(Eligibility.ELIGIBLE)
                    if passes
                    else eligibility_db(Eligibility.DISQUALIFIED)
                )
                n_pass += int(passes)
                advance(1)
    session.commit()
    log(
        f"validate:{case}",
        f"filter — {n_pass} passed, {len(rows) - n_pass} disqualified",
    )


def _phase_score(
    session: Session, case: str, matrix: np.ndarray, clustering_grid_workers: int = 1
) -> None:
    rows = (
        session.query(ClusterRun)
        .filter(
            ClusterRun.embedding_case == case,
            is_eligible(ClusterRun.eligibility),
            ClusterRun.dbcv.is_(None),
        )
        .all()
    )
    if not rows:
        return

    workers = max(1, clustering_grid_workers)

    def persist_row_result(row_id: int, outcome) -> None:
        sess = get_session()
        try:
            row = sess.get(ClusterRun, row_id)
            if row is None:
                return
            if outcome == "value_error":
                log(
                    f"validate:{case}",
                    f"score skip id={row_id} — ValueError",
                    level="warn",
                )
                row.eligibility = eligibility_db(Eligibility.DISQUALIFIED)
            elif outcome == "dbcv_fail":
                log(
                    f"validate:{case}",
                    f"dbcv failed id={row_id} — disqualifying",
                    level="err",
                )
                row.eligibility = eligibility_db(Eligibility.DISQUALIFIED)
            else:
                dbcv, sil = outcome
                row.dbcv = dbcv
                row.silhouette = sil
            sess.commit()
        finally:
            sess.close()

    with progress(len(rows), f"validate score · {case}") as advance:
        if workers == 1:
            for i, row in enumerate(rows):
                params = _row_to_params(row)
                advance(0, detail=f"id={row.id} ({i + 1}/{len(rows)})")
                outcome = _compute_row_scores(matrix, params)
                if outcome == "value_error":
                    log(
                        f"validate:{case}",
                        f"score skip id={row.id} — ValueError",
                        level="warn",
                    )
                    row.eligibility = eligibility_db(Eligibility.DISQUALIFIED)
                elif outcome == "dbcv_fail":
                    log(
                        f"validate:{case}",
                        f"dbcv failed id={row.id} — disqualifying",
                        level="err",
                    )
                    row.eligibility = eligibility_db(Eligibility.DISQUALIFIED)
                else:
                    dbcv, sil = outcome
                    row.dbcv = dbcv
                    row.silhouette = sil
                session.commit()
                advance(
                    1,
                    detail=(
                        f"id={row.id} dbcv={row.dbcv:.4f} sil={row.silhouette:.4f}"
                        if row.dbcv is not None
                        else f"id={row.id} skip"
                    ),
                )
        else:

            def work(item: tuple[int, dict]) -> tuple[int, object]:
                rid, params = item
                return rid, _compute_row_scores(matrix, params)

            payload = [(row.id, _row_to_params(row)) for row in rows]
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = {ex.submit(work, p): p[0] for p in payload}
                for fut in as_completed(futures):
                    row_id = futures[fut]
                    short = f"id={row_id}"
                    advance(0, detail=short)
                    try:
                        rid, outcome = fut.result()
                    except Exception as exc:
                        log(
                            f"validate:{case}",
                            f"score skip id={row_id} — {exc}",
                            level="warn",
                        )
                        persist_row_result(row_id, "value_error")
                        advance(1, detail=f"{short} skip")
                        continue
                    persist_row_result(rid, outcome)
                    advance(1, detail=f"{short} done")

    if workers > 1:
        session.expire_all()


def _find_param_neighbors(
    target: ClusterRun, candidates: list[ClusterRun]
) -> list[ClusterRun]:
    all_rows = [target, *candidates]
    distinct: dict[str, list] = {}
    for col in _NUMERIC_PARAM_COLS:
        vals = sorted(
            set(getattr(r, col) for r in all_rows if getattr(r, col) is not None)
        )
        distinct[col] = vals

    neighbors = []
    for cand in candidates:
        if cand.id == target.id:
            continue
        n_diffs = 0
        valid = True
        for col in _PARAM_COLS:
            tv, cv = getattr(target, col), getattr(cand, col)
            if tv == cv:
                continue
            n_diffs += 1
            if n_diffs > 1:
                valid = False
                break
            if col in _NUMERIC_PARAM_COLS:
                vals = distinct[col]
                if tv not in vals or cv not in vals:
                    valid = False
                    break
                if abs(vals.index(tv) - vals.index(cv)) != 1:
                    valid = False
                    break
            # categorical: any other value counts as adjacent
        if valid and n_diffs == 1:
            neighbors.append(cand)
    return neighbors


def _phase_plateau(session: Session, case: str, settings) -> None:
    """Compute local DBCV neighborhood mean for every qualifying run in the current grid."""
    all_scored = (
        session.query(ClusterRun)
        .filter(
            ClusterRun.embedding_case == case,
            ClusterRun.in_current_grid,
            is_eligible(ClusterRun.eligibility),
            ClusterRun.dbcv.isnot(None),
        )
        .all()
    )
    needs_plateau = [r for r in all_scored if r.param_plateau_score is None]
    if not needs_plateau:
        log(f"validate:{case}", "plateau — nothing to do")
        return

    with progress(len(needs_plateau), f"validate plateau · {case}") as advance:
        for row in needs_plateau:
            neighbors = _find_param_neighbors(row, all_scored)
            dbcv_vals = [n.dbcv for n in neighbors if n.dbcv is not None]
            # No neighbors → fallback to own dbcv so drop = 0 (neutral, not rejected)
            row.param_plateau_score = (
                float(np.mean(dbcv_vals)) if dbcv_vals else row.dbcv
            )
            advance(
                1,
                detail=f"id={row.id} plateau={row.param_plateau_score:.4f} ({len(dbcv_vals)} neighbors)",
            )
    session.commit()
    log(f"validate:{case}", f"plateau — scored {len(needs_plateau)} rows")


def _select_best(session: Session, case: str, settings) -> ClusterRun | None:
    threshold = float(settings.plateau_drop_threshold)
    rows = list_eligible_best_rows(session, case)
    if not rows:
        log(f"validate:{case}", "select — no eligible runs", level="warn")
        return None

    survivors = [r for r in rows if r.dbcv - r.param_plateau_score <= threshold]  # type: ignore[operator]
    if not survivors:
        log(
            f"validate:{case}",
            f"plateau filter rejected all {len(rows)} runs — falling back to DBCV rank",
            level="warn",
        )

    best = pick_best_cluster_run(rows, threshold=threshold)
    if best is None:
        return None
    log(
        f"validate:{case}",
        f"selected run id={best.id} dbcv={best.dbcv:.4f} plateau={best.param_plateau_score:.4f}",
        level="ok",
    )
    return best


def validate_clustering(
    settings, clustering_grid_workers: int = 1
) -> dict[str, dict | None]:
    """Phase 6b entry point: filter → score → plateau → select, per embedding case."""
    current_hash = _compute_validation_config_hash(settings)
    result: dict[str, dict | None] = {}
    for case in ["video", "sandwich", "audio"]:
        log(f"validate:{case}", "starting")
        matrix, _ = load_user_matrix(case)
        if matrix.shape[0] == 0:
            log(f"validate:{case}", "no embeddings — skipping", level="warn")
            result[case] = None
            continue
        session = get_session()
        try:
            _invalidate_stale_rows(session, case, current_hash)
            _phase_filter(session, case, settings)
            _phase_score(session, case, matrix, clustering_grid_workers)
            _phase_plateau(session, case, settings)
            best = _select_best(session, case, settings)
            result[case] = _row_to_params(best) if best is not None else None
        finally:
            session.close()
        log(f"validate:{case}", "done", level="ok")
    return result
