"""Phase 6b — clustering validation: filter, score, composite, bootstrap, plateau."""
import hashlib
import json
import math
import os

import numpy as np
import hdbscan.validity
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sqlalchemy.orm import Session

from modules.database import ClusterRun, get_session
from modules.clustering import compute_clusters, load_user_matrix
from modules.services import log


_PARAM_COLS = [
    "umap_n_components", "umap_n_neighbors", "umap_min_dist", "umap_metric",
    "umap2d_n_neighbors", "umap2d_min_dist", "umap2d_metric",
    "hdbscan_min_cluster_size", "hdbscan_min_samples",
    "hdbscan_cluster_selection_method", "hdbscan_metric",
    "random_state",
]


def _compute_validation_config_hash() -> str:
    config = {
        "max_noise": os.environ.get("VALIDATION_MAX_NOISE_RATIO", "0.3"),
        "min_clusters": os.environ.get("VALIDATION_MIN_CLUSTERS", "3"),
        "max_clusters": os.environ.get("VALIDATION_MAX_CLUSTERS", "20"),
        "top_n_bootstrap": os.environ.get("VALIDATION_TOP_N_BOOTSTRAP", "20"),
        "bootstrap_n": os.environ.get("VALIDATION_BOOTSTRAP_N", "30"),
        "top_n_plateau": os.environ.get("VALIDATION_TOP_N_PLATEAU", "20"),
    }
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]


def _row_to_params(row: ClusterRun) -> dict:
    return {col: getattr(row, col) for col in _PARAM_COLS}


def _minmax(values: list[float]) -> list[float]:
    finite = [v for v in values if not math.isnan(v)]
    if not finite or max(finite) == min(finite):
        return [0.0 for _ in values]
    lo, hi = min(finite), max(finite)
    return [0.0 if math.isnan(v) else (v - lo) / (hi - lo) for v in values]


def _phase_filter(session: Session, case: str) -> None:
    max_noise = float(os.environ.get("VALIDATION_MAX_NOISE_RATIO", "0.3"))
    min_clusters = int(os.environ.get("VALIDATION_MIN_CLUSTERS", "3"))
    max_clusters = int(os.environ.get("VALIDATION_MAX_CLUSTERS", "20"))

    rows = (
        session.query(ClusterRun)
        .filter(
            ClusterRun.embedding_case == case,
            ClusterRun.in_current_grid == 1,
        )
        .all()
    )
    n_pass = 0
    for row in rows:
        passes = (
            row.noise_ratio <= max_noise
            and min_clusters <= row.n_clusters <= max_clusters
        )
        row.disqualified = 0 if passes else 1
        n_pass += int(passes)
    session.commit()

    log(f"validate:{case}", f"filter — {n_pass} passed, {len(rows) - n_pass} disqualified")


def _phase_score(session: Session, case: str, matrix: np.ndarray) -> None:
    rows = (
        session.query(ClusterRun)
        .filter(
            ClusterRun.embedding_case == case,
            ClusterRun.disqualified == 0,
            ClusterRun.dbcv.is_(None),
        )
        .all()
    )

    for i, row in enumerate(rows):
        params = _row_to_params(row)
        try:
            result = compute_clusters(matrix, return_nd_matrix=True, **params)
        except ValueError as exc:
            log(f"validate:{case}", f"score skip id={row.id} — {exc}", level="warn")
            row.disqualified = 1
            session.commit()
            continue

        X_nd = result.matrix_nd.astype(np.float64)
        labels = result.labels

        try:
            row.dbcv = float(hdbscan.validity.validity_index(
                X_nd, labels, metric=row.hdbscan_metric
            ))
        except Exception:
            log(f"validate:{case}", f"dbcv failed id={row.id} — disqualifying", level="err")
            row.disqualified = 1
            session.commit()
            continue

        non_noise = labels != -1
        unique_clusters = np.unique(labels[non_noise])
        if len(unique_clusters) >= 2:
            try:
                row.silhouette = float(silhouette_score(X_nd[non_noise], labels[non_noise]))
            except Exception:
                row.silhouette = 0.0
        else:
            row.silhouette = 0.0

        session.commit()
        dbcv_str = f"{row.dbcv:.4f}"
        sil_str = f"{row.silhouette:.4f}"
        log(f"validate:{case}", f"scored {i + 1}/{len(rows)} id={row.id} dbcv={dbcv_str} sil={sil_str}")


def _phase_composite(session: Session, case: str) -> None:
    rows = (
        session.query(ClusterRun)
        .filter(
            ClusterRun.embedding_case == case,
            ClusterRun.disqualified == 0,
            ClusterRun.dbcv.isnot(None),
        )
        .all()
    )
    if not rows:
        return

    dbcv_norm = _minmax([r.dbcv for r in rows])
    sil_norm = _minmax([r.silhouette if r.silhouette is not None else float("nan") for r in rows])
    stab_vals = [r.bootstrap_stability if r.bootstrap_stability is not None else 0.0 for r in rows]
    stab_norm = _minmax(stab_vals)

    for row, dn, sn, stn in zip(rows, dbcv_norm, sil_norm, stab_norm):
        row.composite_score = round(0.5 * dn + 0.2 * sn + 0.3 * stn, 6)
    session.commit()
    log(f"validate:{case}", f"composite — updated {len(rows)} rows")


def _phase_composite_final(session: Session, case: str) -> None:
    """Recompute composite_score using the full 4-factor formula after plateau is available.

    composite = 0.35*dbcv_norm + 0.15*sil_norm + 0.20*bootstrap_norm + 0.30*plateau_norm

    Only current-grid rows with param_plateau_score set are considered. bootstrap_stability=None → 0.0.
    """
    rows = (
        session.query(ClusterRun)
        .filter(
            ClusterRun.embedding_case == case,
            ClusterRun.disqualified == 0,
            ClusterRun.in_current_grid == 1,
            ClusterRun.param_plateau_score.isnot(None),
        )
        .all()
    )
    if not rows:
        return

    dbcv_norm = _minmax([r.dbcv if r.dbcv is not None else float("nan") for r in rows])
    sil_norm = _minmax([r.silhouette if r.silhouette is not None else float("nan") for r in rows])
    stab_vals = [r.bootstrap_stability if r.bootstrap_stability is not None else 0.0 for r in rows]
    stab_norm = _minmax(stab_vals)
    plateau_norm = _minmax([r.param_plateau_score for r in rows])

    for row, dn, sn, stn, pn in zip(rows, dbcv_norm, sil_norm, stab_norm, plateau_norm):
        row.composite_score = round(0.35 * dn + 0.15 * sn + 0.20 * stn + 0.30 * pn, 6)
    session.commit()
    log(f"validate:{case}", f"composite_final — updated {len(rows)} rows")


def _ari_non_noise(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    mask = (labels_a != -1) & (labels_b != -1)
    if mask.sum() < 2:
        return 0.0
    return float(adjusted_rand_score(labels_a[mask], labels_b[mask]))


def _phase_bootstrap(session: Session, case: str, matrix: np.ndarray) -> None:
    top_n = int(os.environ.get("VALIDATION_TOP_N_BOOTSTRAP", "20"))
    n_runs = int(os.environ.get("VALIDATION_BOOTSTRAP_N", "30"))
    n_rows = matrix.shape[0]

    top_ids = [
        r.id
        for r in (
            session.query(ClusterRun.id)
            .filter(
                ClusterRun.embedding_case == case,
                ClusterRun.disqualified == 0,
                ClusterRun.dbcv.isnot(None),
            )
            .order_by(ClusterRun.dbcv.desc())
            .limit(top_n)
            .all()
        )
    ]
    rows = (
        session.query(ClusterRun)
        .filter(
            ClusterRun.id.in_(top_ids),
            ClusterRun.bootstrap_stability.is_(None),
        )
        .all()
    )
    if not rows:
        log(f"validate:{case}", "bootstrap — nothing to do")
        return

    rng = np.random.default_rng(42)

    for row_i, row in enumerate(rows):
        params = _row_to_params(row)
        try:
            original = compute_clusters(matrix, **params)
        except ValueError as exc:
            log(f"validate:{case}", f"bootstrap skip id={row.id} — {exc}", level="warn")
            continue

        aris = []
        for _ in range(n_runs):
            idx = rng.integers(0, n_rows, size=n_rows)
            try:
                boot = compute_clusters(matrix[idx], **params)
            except ValueError:
                continue
            aris.append(_ari_non_noise(original.labels[idx], boot.labels))

        if not aris:
            log(f"validate:{case}", f"bootstrap all-failed id={row.id} — disqualifying", level="err")
            row.disqualified = 1
            session.commit()
            continue

        row.bootstrap_stability = float(np.mean(aris))
        row.bootstrap_n_runs = len(aris)
        session.commit()
        msg = (
            f"bootstrap {row_i + 1}/{len(rows)} id={row.id}"
            f" stability={row.bootstrap_stability:.4f} ({row.bootstrap_n_runs} runs)"
        )
        log(f"validate:{case}", msg)


_NUMERIC_PARAM_COLS = [
    "umap_n_components", "umap_n_neighbors", "umap_min_dist",
    "umap2d_n_neighbors", "umap2d_min_dist",
    "hdbscan_min_cluster_size", "hdbscan_min_samples", "random_state",
]


def _find_param_neighbors(target: ClusterRun, candidates: list[ClusterRun]) -> list[ClusterRun]:
    all_rows = [target] + candidates
    distinct: dict[str, list] = {}
    for col in _NUMERIC_PARAM_COLS:
        vals = sorted(set(getattr(r, col) for r in all_rows if getattr(r, col) is not None))
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


def _phase_plateau(session: Session, case: str) -> None:
    top_n = int(os.environ.get("VALIDATION_TOP_N_PLATEAU", "20"))

    top_rows = (
        session.query(ClusterRun)
        .filter(
            ClusterRun.embedding_case == case,
            ClusterRun.disqualified == 0,
            ClusterRun.composite_score.isnot(None),
            ClusterRun.param_plateau_score.is_(None),
        )
        .order_by(ClusterRun.composite_score.desc())
        .limit(top_n)
        .all()
    )
    if not top_rows:
        log(f"validate:{case}", "plateau — nothing to do")
        return

    all_rows = (
        session.query(ClusterRun)
        .filter(
            ClusterRun.embedding_case == case,
            ClusterRun.disqualified == 0,
            ClusterRun.composite_score.isnot(None),
        )
        .all()
    )

    for row in top_rows:
        neighbors = _find_param_neighbors(row, all_rows)
        scores = [n.composite_score for n in neighbors if n.composite_score is not None]
        row.param_plateau_score = float(np.mean(scores)) if scores else 0.0
    session.commit()
    log(f"validate:{case}", f"plateau — scored {len(top_rows)} top rows")


def _select_best(session: Session, case: str) -> ClusterRun | None:
    override = os.environ.get(f"CLUSTER_OVERRIDE_{case.upper()}")
    if override:
        row = session.get(ClusterRun, int(override))
        if row is None:
            raise ValueError(
                f"CLUSTER_OVERRIDE_{case.upper()}={override} but no ClusterRun with that id"
            )
        log(f"validate:{case}", f"override — using run id={row.id} (forced via env var)")
        return row

    rows = (
        session.query(ClusterRun)
        .filter(
            ClusterRun.embedding_case == case,
            ClusterRun.disqualified == 0,
            ClusterRun.composite_score.isnot(None),
            ClusterRun.param_plateau_score.isnot(None),
        )
        .all()
    )
    if not rows:
        log(f"validate:{case}", "select — no eligible runs", level="warn")
        return None

    best = max(rows, key=lambda r: r.composite_score)
    msg = (
        f"selected run id={best.id} composite={best.composite_score:.4f}"
        f" plateau={best.param_plateau_score:.4f}"
    )
    log(f"validate:{case}", msg, level="ok")
    return best


def validate_clustering() -> dict[str, dict | None]:
    """Phase 6b entry point. Runs all 5 validation phases per embedding case.

    Returns best params per case (ready to splat into cluster_users), or None
    if no eligible run exists for that case.
    """
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
            _phase_filter(session, case)
            _phase_score(session, case, matrix)
            _phase_composite(session, case)
            _phase_bootstrap(session, case, matrix)
            _phase_composite(session, case)
            _phase_plateau(session, case)
            _phase_composite_final(session, case)
            best = _select_best(session, case)
            result[case] = _row_to_params(best) if best is not None else None
        finally:
            session.close()
        log(f"validate:{case}", "done", level="ok")
    return result
