"""Phase 6b — clustering validation: filter, score, plateau."""

import json
import time
from types import SimpleNamespace
from typing import cast

import hdbscan.validity
import numpy as np
from sklearn.metrics import silhouette_score

from core import fingerprint as fp
from core.config import Settings
from core.console import progress
from core.database import ClusterRun, get_session
from core.log import event, scope, stage, warn
from core.pipeline import Stage
from modules.clustering.core import (
    CLUSTER_PARAM_COLS,
    DEFAULT_HDBSCAN_METRIC,
    ClusterParams,
    compute_clusters,
    load_user_matrix,
    resolve_hdbscan_metric,
)

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

ScoreOutcome = (
    tuple[float, float] | str
)  # (dbcv, silhouette) | "value_error" | "dbcv_fail"

_EMPTY_UPDATE: dict = {
    "passes_validation": False,
    "dbcv": None,
    "silhouette": None,
    "param_plateau_score": None,
}


# ── scoring ───────────────────────────────────────────────────────────────────


def _compute_row_scores(
    matrix: np.ndarray, params: dict, max_cluster_frac: float = 0.0
) -> ScoreOutcome:
    """Run clustering and metrics for one param set. Returns (dbcv, silhouette) or error token."""
    try:
        result = compute_clusters(
            matrix,
            ClusterParams.from_combo(params, max_cluster_frac=max_cluster_frac),
            random_state=int(params["random_state"]),
            return_nd_matrix=True,
        )
    except ValueError:
        return "value_error"
    if result.matrix_nd is None:
        return "value_error"

    x_nd = result.matrix_nd.astype(np.float64)
    labels = result.labels
    metric = resolve_hdbscan_metric(
        params.get("hdbscan_metric", DEFAULT_HDBSCAN_METRIC)
    )

    try:
        dbcv = float(
            cast(float, hdbscan.validity.validity_index(x_nd, labels, metric=metric))
        )
    except Exception:
        return "dbcv_fail"

    non_noise = labels != -1
    if len(np.unique(labels[non_noise])) >= 2:
        try:
            sil = float(
                silhouette_score(
                    x_nd[non_noise],
                    labels[non_noise],
                    metric=metric,
                    random_state=params.get("random_state"),
                )
            )
        except Exception:
            sil = 0.0
    else:
        sil = 0.0
    return dbcv, sil


# ── plateau ───────────────────────────────────────────────────────────────────


def _adjacent_numeric(vals: list, tv, cv) -> bool:
    """True if *tv* and *cv* are neighbours in the sorted distinct-value list."""
    if tv not in vals or cv not in vals:
        return False
    return abs(vals.index(tv) - vals.index(cv)) == 1


def _is_param_neighbor(target, cand, distinct: dict[str, list]) -> bool:
    """True if *cand* differs from *target* in exactly one parameter, with any
    numeric difference adjacent in that column's sorted distinct values."""
    diffs = 0
    for col in CLUSTER_PARAM_COLS:
        tv, cv = getattr(target, col), getattr(cand, col)
        if tv == cv:
            continue
        diffs += 1
        if diffs > 1:
            return False
        # categorical: any other value counts as adjacent
        if col in _NUMERIC_PARAM_COLS and not _adjacent_numeric(distinct[col], tv, cv):
            return False
    return diffs == 1


def _find_param_neighbors(target, candidates: list) -> list:
    """Rows that differ from *target* in exactly one parameter; numeric diffs
    must be adjacent in the sorted distinct-value list of that column."""
    all_rows = [target, *candidates]
    distinct: dict[str, list] = {
        col: sorted(
            {getattr(r, col) for r in all_rows if getattr(r, col, None) is not None}
        )
        for col in _NUMERIC_PARAM_COLS
    }
    return [
        cand
        for cand in candidates
        if cand.id != target.id and _is_param_neighbor(target, cand, distinct)
    ]


# ── fingerprint ───────────────────────────────────────────────────────────────


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
                "max_dominance": float(settings.max_dominance),
            },
            sort_keys=True,
        )
    )
    dependency = fp.stage_dependency_hash(session, Stage.CLUSTER_SEARCH, case)
    return fp.Fingerprint(data=data, config=config, dependency=dependency)


# ── phase orchestration ──────────────────────────────────────────────────────


def _load_snapshots(case: str) -> list[SimpleNamespace]:
    """Snapshot ClusterRun rows into detached namespaces (DB session closed)."""
    session = get_session()
    try:
        rows = (
            session.query(ClusterRun)
            .filter(ClusterRun.embedding_case == case)
            .order_by(ClusterRun.id)
            .all()
        )
        return [
            SimpleNamespace(
                id=r.id,
                noise_ratio=r.noise_ratio,
                n_clusters=r.n_clusters,
                max_size=r.max_size,
                params={col: getattr(r, col) for col in CLUSTER_PARAM_COLS},
                **{col: getattr(r, col) for col in CLUSTER_PARAM_COLS},
            )
            for r in rows
        ]
    finally:
        session.close()


def _passes_filter(snap: SimpleNamespace, n_total: int, settings) -> bool:
    max_dom = float(settings.max_dominance)
    if not (
        snap.noise_ratio <= float(settings.max_noise_ratio)
        and int(settings.min_clusters) <= snap.n_clusters <= int(settings.max_clusters)
    ):
        return False
    # >= 1.0 disables the dominance guard; skip it so reconstructed-assigned
    # float error cannot spuriously reject a run.
    if max_dom >= 1.0:
        return True
    # Recover the exact integer assigned count: noise_ratio is stored rounded
    # so n_total*(1-noise_ratio) drifts; n_total - round(noise) is exact.
    assigned = n_total - round(snap.noise_ratio * n_total)
    dominance = (snap.max_size / assigned) if assigned > 0 else 1.0
    return dominance <= max_dom


def _score_passing(
    case: str,
    matrix: np.ndarray,
    snapshots: list[SimpleNamespace],
    updates: dict[int, dict],
    max_cluster_frac: float,
) -> None:
    to_score = [s for s in snapshots if updates[s.id]["passes_validation"]]
    if not to_score:
        return
    with progress(len(to_score), f"validate score · {case}") as advance:
        for i, snap in enumerate(to_score):
            advance(0, detail=f"id={snap.id} ({i + 1}/{len(to_score)})")
            t0 = time.perf_counter()
            outcome = _compute_row_scores(matrix, snap.params, max_cluster_frac)
            elapsed = time.perf_counter() - t0
            if isinstance(outcome, str):  # "value_error" | "dbcv_fail"
                event(
                    "EXTRACT",
                    f"run_{snap.id}",
                    result="ERR",
                    stats={"time": elapsed, "err": outcome},
                )
                updates[snap.id]["passes_validation"] = False
                advance(1, detail=f"id={snap.id} skip")
                continue
            dbcv, sil = outcome
            updates[snap.id]["dbcv"] = dbcv
            updates[snap.id]["silhouette"] = sil
            event(
                "EXTRACT",
                f"run_{snap.id}",
                stats={"time": elapsed, "dbcv": round(dbcv, 3), "silh": round(sil, 3)},
            )
            advance(1, detail=f"id={snap.id} dbcv={dbcv:.4f}")


def _compute_plateau(
    case: str,
    snapshots: list[SimpleNamespace],
    updates: dict[int, dict],
) -> None:
    scored = [
        SimpleNamespace(
            id=s.id,
            dbcv=updates[s.id]["dbcv"],
            **{col: getattr(s, col) for col in CLUSTER_PARAM_COLS},
        )
        for s in snapshots
        if updates[s.id]["passes_validation"] and updates[s.id]["dbcv"] is not None
    ]
    if not scored:
        return

    event("SCAN", "plateau", stats={"n": len(scored)})
    with progress(len(scored), f"validate plateau · {case}") as advance:
        for proxy in scored:
            neighbors = _find_param_neighbors(proxy, scored)
            dbcv_vals = [n.dbcv for n in neighbors if n.dbcv is not None]
            plateau = float(np.mean(dbcv_vals)) if dbcv_vals else proxy.dbcv
            updates[proxy.id]["param_plateau_score"] = plateau
            event(
                "EXTRACT",
                f"run_{proxy.id}",
                stats={"plateau": round(plateau, 4), "neighbors": len(dbcv_vals)},
            )
            advance(
                1,
                detail=f"id={proxy.id} plateau={plateau:.4f} ({len(dbcv_vals)} neighbors)",
            )


@scope("clustering:validation")
def _compute_updates(
    case: str,
    matrix: np.ndarray,
    settings,
    max_cluster_frac: float = 0.0,
) -> dict[int, dict]:
    """Filter → score → plateau for all ClusterRun rows of *case*. No DB writes.

    Returns ``{run_id: {passes_validation, dbcv, silhouette, param_plateau_score}}``.
    """
    snapshots = _load_snapshots(case)
    if not snapshots:
        return {}

    n_total = matrix.shape[0]
    updates: dict[int, dict] = {
        s.id: {
            **_EMPTY_UPDATE,
            "passes_validation": _passes_filter(s, n_total, settings),
        }
        for s in snapshots
    }

    _score_passing(case, matrix, snapshots, updates, max_cluster_frac)
    _compute_plateau(case, snapshots, updates)

    n_pass = sum(1 for u in updates.values() if u["passes_validation"])
    event("WRITE", "filter", stats={"pass": n_pass, "fail": len(updates) - n_pass})
    return updates


# ── entry point ──────────────────────────────────────────────────────────────


def _check_fingerprint(case: str, settings) -> tuple[fp.Fingerprint, bool, str]:
    session = get_session()
    try:
        current = _fingerprint(session, case, settings)
        stale = fp.is_stale(session, STAGE, case, current)
        diff = fp.describe_diff(session, STAGE, case, current) if stale else ""
    finally:
        session.close()
    return current, stale, diff


def _seal_empty(case: str, current: fp.Fingerprint) -> None:
    session = get_session()
    try:
        session.query(ClusterRun).filter(ClusterRun.embedding_case == case).update(
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


def _seal_updates(case: str, current: fp.Fingerprint, updates: dict[int, dict]) -> None:
    session = get_session()
    try:
        for rid, fields in updates.items():
            session.query(ClusterRun).filter(ClusterRun.id == rid).update(
                fields, synchronize_session=False
            )
        fp.mark_complete(session, STAGE, case, current)
        session.commit()
    finally:
        session.close()


@stage("clustering:validation")
def validate_clustering(settings: Settings, cases: tuple[str, ...]) -> None:
    """Filter → score → plateau, fingerprint-gated per case."""
    validation = settings.validation
    max_cluster_frac = float(settings.search.hdbscan_max_cluster_frac)
    for case in cases:
        preprocess = settings.search.embedding_preprocess.get(case, "none")
        current, stale, diff = _check_fingerprint(case, validation)
        if not stale:
            event("SKIP", "fingerprint")
            continue
        warn("SCAN", "fingerprint", stats={"diff": diff})

        matrix, _ = load_user_matrix(case, preprocess=preprocess)
        if matrix.shape[0] == 0:
            _seal_empty(case, current)
            continue

        updates = _compute_updates(
            case, matrix, validation, max_cluster_frac=max_cluster_frac
        )
        _seal_updates(case, current, updates)

        n_pass = sum(1 for u in updates.values() if u["passes_validation"])
        event(
            "WRITE",
            f"validate:{case}",
            stats={
                "pass": n_pass,
                "fail": len(updates) - n_pass,
            },
        )
