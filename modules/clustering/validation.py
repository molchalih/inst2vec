"""Phase 6b — clustering validation: filter, score, plateau, select."""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import SimpleNamespace
from typing import cast

import hdbscan.validity
import numpy as np
from sklearn.metrics import silhouette_score

from core import fingerprint as fp
from core.config import Settings
from core.console import log, progress
from core.database import ClusterRun, get_session
from core.pipeline import Stage
from modules.clustering.core import (
    CLUSTER_PARAM_COLS,
    DEFAULT_HDBSCAN_METRIC,
    compute_clusters,
    load_user_matrix,
    resolve_hdbscan_metric,
)
from modules.embeddings.cases import default_cases

STAGE = Stage.CLUSTER_VALIDATION

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


def _row_to_params(row: ClusterRun) -> dict:
    return {col: getattr(row, col) for col in CLUSTER_PARAM_COLS}


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
        for col in CLUSTER_PARAM_COLS:
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


# ── fingerprint helpers ───────────────────────────────────────────────────────


def _fingerprint(session, case: str, settings) -> fp.Fingerprint:
    rows = (
        session.query(
            ClusterRun.id,
            ClusterRun.n_clusters,
            ClusterRun.noise_ratio,
            *[getattr(ClusterRun, col) for col in CLUSTER_PARAM_COLS],
        )
        .filter(ClusterRun.embedding_case == case)
        .order_by(ClusterRun.id)
        .all()
    )
    data = fp.hash_rows(tuple(r) for r in rows)
    config = fp.hash_text(
        json.dumps(
            {
                "max_noise_ratio": float(settings.max_noise_ratio),
                "min_clusters": int(settings.min_clusters),
                "max_clusters": int(settings.max_clusters),
                "plateau_drop_threshold": float(settings.plateau_drop_threshold),
            },
            sort_keys=True,
        )
    )
    dependency = fp.stage_dependency_hash(session, Stage.CLUSTER_SEARCH, case)
    return fp.Fingerprint(data=data, config=config, dependency=dependency)


def _compute_updates(
    case: str,
    matrix: np.ndarray,
    settings,
    workers: int = 1,
) -> dict[int, dict]:
    """Compute filter+score+plateau for all ClusterRun rows for *case* in memory.

    Opens and closes its own read-only session. Returns a mapping of
    {run_id: {passes_validation, dbcv, silhouette, param_plateau_score}}
    without touching the database.
    """
    max_noise = float(settings.max_noise_ratio)
    min_clusters = int(settings.min_clusters)
    max_clusters = int(settings.max_clusters)

    # --- load rows (read-only) ------------------------------------------------
    session = get_session()
    try:
        all_rows = (
            session.query(ClusterRun)
            .filter(ClusterRun.embedding_case == case)
            .order_by(ClusterRun.id)
            .all()
        )
        # Snapshot into plain dicts so we can close the session immediately.
        snapshots = []
        for row in all_rows:
            snapshots.append(
                {
                    "id": row.id,
                    "noise_ratio": row.noise_ratio,
                    "n_clusters": row.n_clusters,
                    "params": _row_to_params(row),
                    # keep param cols on snapshot for neighbor detection
                    **{col: getattr(row, col) for col in CLUSTER_PARAM_COLS},
                }
            )
    finally:
        session.close()

    if not snapshots:
        return {}

    # --- phase filter ---------------------------------------------------------
    updates: dict[int, dict] = {}
    for snap in snapshots:
        passes = (
            snap["noise_ratio"] <= max_noise
            and min_clusters <= snap["n_clusters"] <= max_clusters
        )
        updates[snap["id"]] = {
            "passes_validation": passes,
            "dbcv": None,
            "silhouette": None,
            "param_plateau_score": None,
        }

    # --- phase score ----------------------------------------------------------
    to_score = [snap for snap in snapshots if updates[snap["id"]]["passes_validation"]]

    if to_score:
        _workers = max(1, workers)
        if _workers == 1:
            with progress(len(to_score), f"validate score · {case}") as advance:
                for i, snap in enumerate(to_score):
                    advance(0, detail=f"id={snap['id']} ({i + 1}/{len(to_score)})")
                    outcome = _compute_row_scores(matrix, snap["params"])
                    if outcome in ("value_error", "dbcv_fail"):
                        log(
                            f"validate:{case}",
                            f"score skip id={snap['id']} — {outcome}",
                            level="warn",
                        )
                        updates[snap["id"]]["passes_validation"] = False
                    else:
                        dbcv, sil = outcome
                        updates[snap["id"]]["dbcv"] = dbcv
                        updates[snap["id"]]["silhouette"] = sil
                    advance(
                        1,
                        detail=(
                            f"id={snap['id']} dbcv={updates[snap['id']]['dbcv']:.4f}"
                            if updates[snap["id"]]["dbcv"] is not None
                            else f"id={snap['id']} skip"
                        ),
                    )
        else:
            results_map: dict[int, object] = {}

            def work(item: tuple[int, dict]) -> tuple[int, object]:
                rid, params = item
                return rid, _compute_row_scores(matrix, params)

            payload = [(snap["id"], snap["params"]) for snap in to_score]
            with (
                progress(len(to_score), f"validate score · {case}") as advance,
                ThreadPoolExecutor(max_workers=_workers) as ex,
            ):
                futures = {ex.submit(work, p): p[0] for p in payload}
                for fut in as_completed(futures):
                    row_id = futures[fut]
                    advance(0, detail=f"id={row_id}")
                    try:
                        rid, outcome = fut.result()
                    except Exception as exc:
                        log(
                            f"validate:{case}",
                            f"score skip id={row_id} — {exc}",
                            level="warn",
                        )
                        results_map[row_id] = "value_error"
                        advance(1, detail=f"id={row_id} skip")
                        continue
                    results_map[rid] = outcome
                    advance(1, detail=f"id={row_id} done")

            for snap in to_score:
                rid = snap["id"]
                outcome = results_map.get(rid, "value_error")
                if outcome in ("value_error", "dbcv_fail"):
                    updates[rid]["passes_validation"] = False
                else:
                    dbcv_val, sil_val = cast(tuple[float, float], outcome)
                    updates[rid]["dbcv"] = dbcv_val
                    updates[rid]["silhouette"] = sil_val

    # --- phase plateau --------------------------------------------------------
    # Build lightweight proxy objects for _find_param_neighbors (it reads
    # ORM attribute names via getattr).  We use a SimpleNamespace per snapshot.
    scored_proxies = []
    for snap in snapshots:
        u = updates[snap["id"]]
        if u["passes_validation"] and u["dbcv"] is not None:
            proxy = SimpleNamespace(
                id=snap["id"],
                dbcv=u["dbcv"],
                **{col: snap[col] for col in CLUSTER_PARAM_COLS},
            )
            scored_proxies.append(proxy)

    if scored_proxies:
        with progress(len(scored_proxies), f"validate plateau · {case}") as advance:
            for proxy in scored_proxies:
                neighbors = _find_param_neighbors(proxy, scored_proxies)  # type: ignore[arg-type]
                dbcv_vals = [n.dbcv for n in neighbors if n.dbcv is not None]
                plateau = float(np.mean(dbcv_vals)) if dbcv_vals else proxy.dbcv
                updates[proxy.id]["param_plateau_score"] = plateau
                advance(
                    1,
                    detail=f"id={proxy.id} plateau={plateau:.4f} ({len(dbcv_vals)} neighbors)",
                )
        log(f"validate:{case}", f"plateau — scored {len(scored_proxies)} rows")

    n_pass = sum(1 for u in updates.values() if u["passes_validation"])
    n_fail = len(updates) - n_pass
    log(f"validate:{case}", f"filter — {n_pass} passed, {n_fail} disqualified")

    return updates


# ── orchestration entry point ─────────────────────────────────────────────────


def validate_clustering(settings: Settings, clustering_grid_workers: int = 1) -> None:
    """Filter -> score -> plateau -> select, fingerprint-gated per case.

    ``settings`` is the full runtime settings object; per-case validation
    knobs are read from ``settings.validation``, and the case set is
    gated via ``default_cases(settings)``.
    """
    validation_settings = settings.validation
    for case in default_cases(settings):
        # 1. fingerprint check
        session = get_session()
        try:
            current = _fingerprint(session, case, validation_settings)
            stale = fp.is_stale(session, STAGE, case, current)
            diff = fp.describe_diff(session, STAGE, case, current) if stale else ""
        finally:
            session.close()

        if not stale:
            log(f"validate:{case}", "fingerprint match — skipping")
            continue

        log(f"validate:{case}", f"stale ({diff}) — recomputing")

        # 2. load matrix (read-only)
        matrix, _ = load_user_matrix(case)

        if matrix.shape[0] == 0:
            log(f"validate:{case}", "no embeddings — sealing empty")
            session = get_session()
            try:
                session.query(ClusterRun).filter(
                    ClusterRun.embedding_case == case
                ).update(
                    {
                        "passes_validation": None,
                        "dbcv": None,
                        "silhouette": None,
                        "param_plateau_score": None,
                    },
                    synchronize_session=False,
                )
                fp.mark_complete(session, STAGE, case, current)
                session.commit()
            finally:
                session.close()
            continue

        # 3. compute in memory (no open write transaction)
        updates = _compute_updates(
            case, matrix, validation_settings, clustering_grid_workers
        )

        # 4. short write section (open AFTER compute)
        session = get_session()
        try:
            for run_id, fields in updates.items():
                session.query(ClusterRun).filter(ClusterRun.id == run_id).update(
                    fields, synchronize_session=False
                )
            fp.mark_complete(session, STAGE, case, current)
            session.commit()
        finally:
            session.close()

        log(f"validate:{case}", "done", level="ok")
