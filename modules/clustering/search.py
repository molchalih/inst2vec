"""Grid search over UMAP + HDBSCAN hyperparameters; saves aggregate metrics to ClusterRun."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product

import numpy as np

from modules.clustering.core import (
    DEFAULT_HDBSCAN_METRIC,
    compute_clusters,
    load_user_matrix,
)
from modules.console import log, progress
from modules.database import Base, ClusterRun, get_engine, get_session

_PARAM_KEYS = (
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


def _load_grid(settings) -> list[dict]:
    """Build cartesian product of hyperparameter combos from settings.

    umap2d_n_neighbors and umap2d_min_dist are fixed scalars (not swept);
    umap2d_metric is swept independently as a list.
    HDBSCAN distance on pass-1 UMAP space is fixed (euclidean); not swept.
    """
    umap_n_components = list(settings.umap_n_components)
    umap_n_neighbors = list(settings.umap_n_neighbors)
    umap_min_dist = [float(x) for x in settings.umap_min_dist]
    umap_metrics = list(settings.umap_metrics)
    umap2d_n_neighbors = int(settings.umap2d_n_neighbors)
    umap2d_min_dist = float(settings.umap2d_min_dist)
    umap2d_metrics = list(settings.umap2d_metrics)
    hdbscan_min_sizes = list(settings.hdbscan_min_cluster_size)
    hdbscan_selection = list(settings.hdbscan_selection)
    random_state = int(settings.random_state)
    cases = ["video", "sandwich", "audio"]

    combos = []
    for case, nc, nn, md, um, u2m, mcs, sel in product(
        cases,
        umap_n_components,
        umap_n_neighbors,
        umap_min_dist,
        umap_metrics,
        umap2d_metrics,
        hdbscan_min_sizes,
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
                hdbscan_min_samples=None,
                hdbscan_cluster_selection_method=sel,
                hdbscan_metric=DEFAULT_HDBSCAN_METRIC,
                random_state=random_state,
            )
        )
    return combos


def run_cluster_search(settings, clustering_grid_workers: int = 1) -> None:
    """Run grid search over all hyperparameter combos; save metrics to ClusterRun.

    Idempotent at the (case, params) level: skips combos already present.
    Stale handling (data/grid changes) is added in a later task via the
    fingerprint layer.
    """
    Base.metadata.create_all(get_engine())
    combos = _load_grid(settings)
    grid_workers = max(1, clustering_grid_workers)

    combos_by_case: dict[str, list[dict]] = {}
    for combo in combos:
        combos_by_case.setdefault(combo["embedding_case"], []).append(combo)

    total_new = 0
    total_skipped = 0

    for case, case_combos in combos_by_case.items():
        matrix, _ = load_user_matrix(case)
        if matrix.shape[0] == 0:
            log(
                f"cluster_search:{case}",
                f"no embeddings — skipping {len(case_combos)} combos",
                level="warn",
            )
            total_skipped += len(case_combos)
            continue

        with progress(len(case_combos), f"cluster search · {case}") as advance:
            pending: list[dict] = []
            session = get_session()
            try:
                for combo in case_combos:
                    short = (
                        f"nc={combo['umap_n_components']} nn={combo['umap_n_neighbors']} "
                        f"mcs={combo['hdbscan_min_cluster_size']}"
                    )
                    if session.query(ClusterRun).filter_by(**combo).first():
                        total_skipped += 1
                        advance(1, detail=f"{short} | cached")
                    else:
                        pending.append(combo)
            finally:
                session.close()

            def run_one(c: dict, *, _matrix=matrix):
                p = {k: v for k, v in c.items() if k != "embedding_case"}
                return c, compute_clusters(_matrix, **p)

            def persist(combo, result) -> None:
                nonlocal total_new
                sizes = result.cluster_sizes
                session = get_session()
                try:
                    if session.query(ClusterRun).filter_by(**combo).first():
                        return
                    session.add(
                        ClusterRun(
                            **combo,
                            n_clusters=result.n_clusters,
                            noise_ratio=round(result.noise_ratio, 4),
                            min_size=min(sizes) if sizes else 0,
                            median_size=int(np.median(sizes)) if sizes else 0,
                            max_size=max(sizes) if sizes else 0,
                        )
                    )
                    session.commit()
                    total_new += 1
                finally:
                    session.close()

            if grid_workers == 1:
                for combo in pending:
                    short = (
                        f"nc={combo['umap_n_components']} nn={combo['umap_n_neighbors']} "
                        f"mcs={combo['hdbscan_min_cluster_size']}"
                    )
                    p = {k: v for k, v in combo.items() if k != "embedding_case"}
                    try:
                        result = compute_clusters(matrix, **p)
                    except ValueError as exc:
                        log(f"cluster_search:{case}", f"skipping — {exc}", level="warn")
                        total_skipped += 1
                        advance(1, detail=f"{short} | skip {str(exc)[:48]}")
                        continue
                    persist(combo, result)
                    advance(1, detail=f"{short} | k={result.n_clusters} new")
            else:
                with ThreadPoolExecutor(max_workers=grid_workers) as ex:
                    futures = {ex.submit(run_one, c): c for c in pending}
                    for fut in as_completed(futures):
                        combo = futures[fut]
                        short = (
                            f"nc={combo['umap_n_components']} nn={combo['umap_n_neighbors']} "
                            f"mcs={combo['hdbscan_min_cluster_size']}"
                        )
                        try:
                            c_done, result = fut.result()
                        except ValueError as exc:
                            log(
                                f"cluster_search:{case}",
                                f"skipping — {exc}",
                                level="warn",
                            )
                            total_skipped += 1
                            advance(1, detail=f"{short} | skip {str(exc)[:48]}")
                            continue
                        persist(c_done, result)
                        advance(1, detail=f"{short} | k={result.n_clusters} new")

    log(
        "cluster_search", f"done — {total_new} new, {total_skipped} skipped", level="ok"
    )
