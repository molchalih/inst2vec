"""Grid search over UMAP + HDBSCAN hyperparameters; saves aggregate metrics to ClusterRun."""

import hashlib
import json
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product

import numpy as np

from core import fingerprint as fp
from core.config import Settings
from core.console import log, progress
from core.database import (
    Clip,
    ClusterRun,
    UserEmbedding,
    clip_used_in_analysis,
    get_session,
)
from core.pipeline import Stage
from modules.clustering.core import (
    CLUSTER_PARAM_COLS,
    DEFAULT_HDBSCAN_METRIC,
    compute_clusters,
    load_user_matrix,
)

STAGE = Stage.CLUSTER_SEARCH


def _load_grid(settings, cases: Iterable[str]) -> list[dict]:
    """Build cartesian product of hyperparameter combos from settings.

    umap2d_n_neighbors and umap2d_min_dist are fixed scalars (not swept);
    umap2d_metric is swept independently as a list.
    HDBSCAN distance on pass-1 UMAP space is fixed (euclidean); not swept.
    hdbscan_min_samples=None (HDBSCAN's default = min_cluster_size) is
    used when the configured list is empty, preserving prior behavior.
    """
    umap_n_components = list(settings.umap_n_components)
    umap_n_neighbors = list(settings.umap_n_neighbors)
    umap_min_dist = [float(x) for x in settings.umap_min_dist]
    umap_metrics = list(settings.umap_metrics)
    umap2d_n_neighbors = int(settings.umap2d_n_neighbors)
    umap2d_min_dist = float(settings.umap2d_min_dist)
    umap2d_metrics = list(settings.umap2d_metrics)
    hdbscan_min_sizes = list(settings.hdbscan_min_cluster_size)
    hdbscan_min_samples_list: list[int | None] = (
        list(settings.hdbscan_min_samples) if settings.hdbscan_min_samples else [None]
    )
    hdbscan_selection = list(settings.hdbscan_selection)
    random_state = int(settings.random_state)
    cases_list = list(cases)

    combos = []
    for case, nc, nn, md, um, u2m, mcs, ms, sel in product(
        cases_list,
        umap_n_components,
        umap_n_neighbors,
        umap_min_dist,
        umap_metrics,
        umap2d_metrics,
        hdbscan_min_sizes,
        hdbscan_min_samples_list,
        hdbscan_selection,
    ):
        combos.append(
            dict(
                embedding_case=case,
                umap_n_components=nc,
                umap_n_neighbors=nn,
                umap_min_dist=md,
                umap_metric=um,
                umap2d_n_neighbors=umap2d_n_neighbors,
                umap2d_min_dist=umap2d_min_dist,
                umap2d_metric=u2m,
                hdbscan_min_cluster_size=mcs,
                hdbscan_min_samples=ms,
                hdbscan_cluster_selection_method=sel,
                hdbscan_metric=DEFAULT_HDBSCAN_METRIC,
                random_state=random_state,
            )
        )
    return combos


def _fingerprint(session, case: str, case_combos: list[dict]) -> fp.Fingerprint:
    rows = (
        session.query(UserEmbedding.user_id, UserEmbedding.embedding)
        .join(Clip, Clip.user_id == UserEmbedding.user_id)
        .filter(
            UserEmbedding.embedding_case == case,
            *clip_used_in_analysis(),
        )
        .distinct()
        .order_by(UserEmbedding.user_id)
        .all()
    )
    data = fp.hash_rows((uid, hashlib.sha256(blob).hexdigest()) for uid, blob in rows)
    config = fp.hash_text(
        json.dumps(
            sorted(
                [{k: c[k] for k in CLUSTER_PARAM_COLS} for c in case_combos],
                key=lambda d: tuple(str(d[k]) for k in CLUSTER_PARAM_COLS),
            ),
            sort_keys=True,
            default=str,
        )
    )
    dependency = fp.stage_dependency_hash(session, Stage.USER_EMBEDDINGS, case)
    return fp.Fingerprint(data=data, config=config, dependency=dependency)


def _combo_to_row(combo: dict, result) -> ClusterRun:
    sizes = result.cluster_sizes
    return ClusterRun(
        **combo,
        n_clusters=result.n_clusters,
        noise_ratio=round(result.noise_ratio, 4),
        min_size=min(sizes) if sizes else 0,
        median_size=int(np.median(sizes)) if sizes else 0,
        max_size=max(sizes) if sizes else 0,
    )


def run_cluster_search(
    settings: Settings,
    cases: tuple[str, ...],
    clustering_grid_workers: int = 1,
) -> None:
    """Run grid search over hyperparameter combos per embedding case.

    Idempotent via modules.fingerprint: fingerprint per case, wipe scoped
    rows on stale, run full grid in memory, bulk-insert, mark_complete.
    Long compute runs outside the write transaction.

    ``cases`` is the tuple of embedding case names to search over (e.g.
    ``("video", "sandwich", "audio")``).  Grid hyperparameters are read
    from ``settings.search``.
    """
    combos = _load_grid(settings.search, cases=cases)
    grid_workers = max(1, clustering_grid_workers)

    combos_by_case: dict[str, list[dict]] = {}
    for combo in combos:
        combos_by_case.setdefault(combo["embedding_case"], []).append(combo)

    for case, case_combos in combos_by_case.items():
        # 1. fingerprint check (read-only session)
        session = get_session()
        try:
            current = _fingerprint(session, case, case_combos)
            stale = fp.is_stale(session, STAGE, case, current)
            diff = fp.describe_diff(session, STAGE, case, current) if stale else ""
        finally:
            session.close()

        scope = f"search:{case}"
        if not stale:
            log(scope, "SKIP", "fingerprint", "ok")
            continue

        log(scope, "SCAN", "fingerprint", "stale", stats={"diff": diff})

        # 2. load inputs (read-only)
        matrix, _ = load_user_matrix(case)

        # 3. compute outside any write transaction
        new_rows: list[ClusterRun] = []
        t_stage = time.perf_counter()
        if matrix.shape[0] > 0:
            with progress(len(case_combos), f"cluster search · {case}") as advance:
                if grid_workers == 1:
                    for combo in case_combos:
                        short = (
                            f"nc={combo['umap_n_components']} "
                            f"nn={combo['umap_n_neighbors']} "
                            f"mcs={combo['hdbscan_min_cluster_size']}"
                        )
                        target = (
                            f"n={combo['umap_n_neighbors']},"
                            f"mcs={combo['hdbscan_min_cluster_size']}"
                        )
                        p = {k: v for k, v in combo.items() if k != "embedding_case"}
                        t0 = time.perf_counter()
                        try:
                            result = compute_clusters(matrix, **p)
                        except ValueError as exc:
                            log(
                                scope,
                                "FIT",
                                target,
                                "ERR",
                                stats={
                                    "time": time.perf_counter() - t0,
                                    "err": str(exc),
                                },
                            )
                            advance(1, detail=f"{short} | skip {str(exc)[:48]}")
                            continue
                        new_rows.append(_combo_to_row(combo, result))
                        log(
                            scope,
                            "FIT",
                            target,
                            "ok",
                            stats={
                                "time": time.perf_counter() - t0,
                                "k": result.n_clusters,
                                "noise": round(result.noise_ratio, 3),
                            },
                        )
                        advance(1, detail=f"{short} | k={result.n_clusters}")
                else:

                    def run_one(c: dict, *, _matrix=matrix):
                        p = {k: v for k, v in c.items() if k != "embedding_case"}
                        t = time.perf_counter()
                        return (
                            c,
                            compute_clusters(_matrix, **p),
                            time.perf_counter() - t,
                        )

                    with ThreadPoolExecutor(max_workers=grid_workers) as ex:
                        futures = {ex.submit(run_one, c): c for c in case_combos}
                        for fut in as_completed(futures):
                            combo = futures[fut]
                            short = (
                                f"nc={combo['umap_n_components']} "
                                f"nn={combo['umap_n_neighbors']} "
                                f"mcs={combo['hdbscan_min_cluster_size']}"
                            )
                            target = (
                                f"n={combo['umap_n_neighbors']},"
                                f"mcs={combo['hdbscan_min_cluster_size']}"
                            )
                            try:
                                c_done, result, duration = fut.result()
                            except ValueError as exc:
                                log(
                                    scope,
                                    "FIT",
                                    target,
                                    "ERR",
                                    stats={"err": str(exc)},
                                )
                                advance(1, detail=f"{short} | skip {str(exc)[:48]}")
                                continue
                            new_rows.append(_combo_to_row(c_done, result))
                            log(
                                scope,
                                "FIT",
                                target,
                                "ok",
                                stats={
                                    "time": duration,
                                    "k": result.n_clusters,
                                    "noise": round(result.noise_ratio, 3),
                                },
                            )
                            advance(1, detail=f"{short} | k={result.n_clusters}")

        # 4. short write section
        session = get_session()
        try:
            session.query(ClusterRun).filter(ClusterRun.embedding_case == case).delete()
            if new_rows:
                session.bulk_save_objects(new_rows)
            fp.mark_complete(session, STAGE, case, current)
            session.commit()
        finally:
            session.close()
        log(
            scope,
            "SEAL",
            "search",
            "ok",
            stats={
                "runs": len(new_rows),
                "time": time.perf_counter() - t_stage,
            },
        )
