"""Grid search over UMAP + HDBSCAN hyperparameters; saves aggregate metrics to ClusterRun."""

import hashlib
import json
import time
from collections.abc import Iterable
from itertools import product

import numpy as np

from core import fingerprint as fp
from core.config import Settings
from core.console import progress
from core.database import (
    Clip,
    ClusterRun,
    UserEmbedding,
    clip_used_in_analysis,
    get_session,
)
from core.log import event, stage, warn
from core.pipeline import Stage
from modules.clustering.core import (
    CLUSTER_PARAM_COLS,
    DEFAULT_HDBSCAN_METRIC,
    ClusterParams,
    ClusterResult,
    compute_clusters,
    load_user_matrix,
)

STAGE = Stage.CLUSTER_SEARCH


def _load_grid(settings, cases: Iterable[str]) -> list[dict]:
    """Cartesian product of hyperparameter combos × cases.

    umap2d_{n_neighbors,min_dist} are fixed scalars; umap2d_metric is swept.
    HDBSCAN metric on pass-1 UMAP space is fixed (euclidean) and not swept.
    Empty hdbscan_min_samples yields one combo with None (HDBSCAN's default).
    """
    samples: list[int | None] = list(settings.hdbscan_min_samples) or [None]
    return [
        {
            "embedding_case": case,
            "umap_n_components": nc,
            "umap_n_neighbors": nn,
            "umap_min_dist": float(md),
            "umap_metric": um,
            "umap2d_n_neighbors": int(settings.umap2d_n_neighbors),
            "umap2d_min_dist": float(settings.umap2d_min_dist),
            "umap2d_metric": u2m,
            "hdbscan_min_cluster_size": mcs,
            "hdbscan_min_samples": ms,
            "hdbscan_cluster_selection_method": sel,
            "hdbscan_metric": DEFAULT_HDBSCAN_METRIC,
            "random_state": int(settings.random_state),
        }
        for case, nc, nn, md, um, u2m, mcs, ms, sel in product(
            list(cases),
            list(settings.umap_n_components),
            list(settings.umap_n_neighbors),
            list(settings.umap_min_dist),
            list(settings.umap_metrics),
            list(settings.umap2d_metrics),
            list(settings.hdbscan_min_cluster_size),
            samples,
            list(settings.hdbscan_selection),
        )
    ]


def _fingerprint(
    session,
    case: str,
    case_combos: list[dict],
    preprocess: str,
    max_cluster_frac: float,
) -> fp.Fingerprint:
    rows = (
        session.query(UserEmbedding.user_id, UserEmbedding.embedding)
        .join(Clip, Clip.user_id == UserEmbedding.user_id)
        .filter(UserEmbedding.embedding_case == case, *clip_used_in_analysis())
        .distinct()
        .order_by(UserEmbedding.user_id)
        .all()
    )
    data = fp.hash_rows((uid, hashlib.sha256(blob).hexdigest()) for uid, blob in rows)
    # preprocess + max_cluster_frac are applied at fit time (not stored on the
    # row), so they must enter the config hash explicitly.
    config = fp.hash_text(
        json.dumps(
            {
                "combos": sorted(
                    [{k: c[k] for k in CLUSTER_PARAM_COLS} for c in case_combos],
                    key=lambda d: tuple(str(d[k]) for k in CLUSTER_PARAM_COLS),
                ),
                "preprocess": preprocess,
                "max_cluster_frac": float(max_cluster_frac),
            },
            sort_keys=True,
            default=str,
        )
    )
    dependency = fp.stage_dependency_hash(session, Stage.USER_EMBEDDINGS, case)
    return fp.Fingerprint(data=data, config=config, dependency=dependency)


def _combo_to_row(combo: dict, result: ClusterResult) -> ClusterRun:
    sizes = result.cluster_sizes
    return ClusterRun(
        **combo,
        n_clusters=result.n_clusters,
        noise_ratio=round(result.noise_ratio, 4),
        min_size=min(sizes) if sizes else 0,
        median_size=int(np.median(sizes)) if sizes else 0,
        max_size=max(sizes) if sizes else 0,
    )


def _short(combo: dict) -> str:
    return (
        f"nc={combo['umap_n_components']} "
        f"nn={combo['umap_n_neighbors']} "
        f"mcs={combo['hdbscan_min_cluster_size']}"
    )


def _target(combo: dict) -> str:
    return f"n={combo['umap_n_neighbors']},mcs={combo['hdbscan_min_cluster_size']}"


def _fit_one(
    matrix: np.ndarray, combo: dict, max_cluster_frac: float
) -> tuple[ClusterResult | None, float, str]:
    """Run compute_clusters for one combo. Returns (result_or_None, elapsed, err)."""
    t0 = time.perf_counter()
    try:
        result = compute_clusters(
            matrix,
            ClusterParams.from_combo(combo, max_cluster_frac=max_cluster_frac),
            random_state=int(combo["random_state"]),
        )
    except ValueError as exc:
        return None, time.perf_counter() - t0, str(exc)
    return result, time.perf_counter() - t0, ""


def _check_fingerprint(
    case: str, combos: list[dict], preprocess: str, max_cluster_frac: float
) -> tuple[fp.Fingerprint, bool, str]:
    session = get_session()
    try:
        current = _fingerprint(session, case, combos, preprocess, max_cluster_frac)
        stale = fp.is_stale(session, STAGE, case, current)
        diff = fp.describe_diff(session, STAGE, case, current) if stale else ""
    finally:
        session.close()
    return current, stale, diff


def _seal_case(case: str, current: fp.Fingerprint, rows: list[ClusterRun]) -> None:
    session = get_session()
    try:
        session.query(ClusterRun).filter(ClusterRun.embedding_case == case).delete()
        if rows:
            session.bulk_save_objects(rows)
        fp.mark_complete(session, STAGE, case, current)
        session.commit()
    finally:
        session.close()


@stage("clustering:search")
def run_cluster_search(settings: Settings, cases: tuple[str, ...]) -> None:
    """Run grid search per embedding case, fingerprint-gated.

    On stale: wipe scoped ClusterRun rows, compute the full grid in memory,
    bulk-insert results, seal StageState. Compute runs outside any open
    write transaction. Empty matrices still seal an empty state so reruns
    short-circuit.
    """
    max_cluster_frac = float(settings.search.hdbscan_max_cluster_frac)
    by_case: dict[str, list[dict]] = {}
    for combo in _load_grid(settings.search, cases=cases):
        by_case.setdefault(combo["embedding_case"], []).append(combo)

    for case, combos in by_case.items():
        preprocess = settings.search.embedding_preprocess.get(case, "none")
        current, stale, diff = _check_fingerprint(
            case, combos, preprocess, max_cluster_frac
        )
        if not stale:
            event("SKIP", "fingerprint")
            continue
        warn("SCAN", "fingerprint", stats={"diff": diff})

        matrix, _ = load_user_matrix(case, preprocess=preprocess)
        new_rows: list[ClusterRun] = []
        if matrix.shape[0] > 0:
            with progress(len(combos), f"cluster search · {case}") as advance:
                for combo in combos:
                    result, elapsed, err = _fit_one(matrix, combo, max_cluster_frac)
                    if result is None:
                        event(
                            "EXTRACT",
                            _target(combo),
                            result="ERR",
                            stats={"time": elapsed, "err": err},
                        )
                        advance(1, detail=f"{_short(combo)} | skip {err[:48]}")
                        continue
                    new_rows.append(_combo_to_row(combo, result))
                    event(
                        "EXTRACT",
                        _target(combo),
                        stats={
                            "time": elapsed,
                            "k": result.n_clusters,
                            "noise": round(result.noise_ratio, 3),
                        },
                    )
                    advance(1, detail=f"{_short(combo)} | k={result.n_clusters}")

        _seal_case(case, current, new_rows)
        event("WRITE", f"search:{case}", stats={"runs": len(new_rows)})
