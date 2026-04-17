"""Grid search over UMAP + HDBSCAN hyperparameters; saves aggregate metrics to ClusterRun."""
import os
from itertools import product

import numpy as np
from sqlalchemy.exc import IntegrityError

from modules.database import Base, engine, get_session, ClusterRun
from modules.clustering import compute_clusters, load_user_matrix


def _parse_ints(val: str) -> list[int]:
    return [int(x) for x in val.split()]


def _parse_floats(val: str) -> list[float]:
    return [float(x) for x in val.split()]


def _parse_strs(val: str) -> list[str]:
    return val.split()


def _load_grid() -> list[dict]:
    """Build cartesian product of hyperparameter combos from env vars.

    umap2d_n_neighbors and umap2d_min_dist are fixed scalars (not swept);
    umap2d_metric is swept independently as a list.
    """
    umap_n_components  = _parse_ints(os.environ.get("CLUSTERING_UMAP_N_COMPONENTS", "15"))
    umap_n_neighbors   = _parse_ints(os.environ.get("CLUSTERING_UMAP_N_NEIGHBORS", "15"))
    umap_min_dist      = _parse_floats(os.environ.get("CLUSTERING_UMAP_MIN_DIST", "0.0"))
    umap_metrics       = _parse_strs(os.environ.get("CLUSTERING_UMAP_METRICS", "cosine"))
    umap2d_n_neighbors = int(os.environ.get("CLUSTERING_UMAP2D_N_NEIGHBORS", "15"))
    umap2d_min_dist    = float(os.environ.get("CLUSTERING_UMAP2D_MIN_DIST", "0.1"))
    umap2d_metrics     = _parse_strs(os.environ.get("CLUSTERING_UMAP2D_METRICS", "cosine"))
    hdbscan_min_sizes  = _parse_ints(os.environ.get("CLUSTERING_HDBSCAN_MIN_CLUSTER_SIZE", "15"))
    hdbscan_selection  = _parse_strs(os.environ.get("CLUSTERING_HDBSCAN_SELECTION", "eom"))
    hdbscan_metrics    = _parse_strs(os.environ.get("CLUSTERING_HDBSCAN_METRICS", "euclidean"))
    random_state       = int(os.environ.get("CLUSTERING_RANDOM_STATE", "42"))
    cases              = ["video", "sandwich", "audio"]

    combos = []
    for case, nc, nn, md, um, u2m, mcs, sel, hm in product(
        cases, umap_n_components, umap_n_neighbors, umap_min_dist, umap_metrics,
        umap2d_metrics, hdbscan_min_sizes, hdbscan_selection, hdbscan_metrics,
    ):
        combos.append(dict(
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
            hdbscan_metric=hm,
            random_state=random_state,
        ))
    return combos


def run_cluster_search() -> None:
    """Run grid search over all hyperparameter combos from env; save metrics to ClusterRun.

    Idempotent: skips any combo already present in the DB. Groups combos by
    embedding_case so the user embedding matrix is loaded once per case.
    """
    Base.metadata.create_all(engine)
    combos = _load_grid()

    combos_by_case: dict[str, list[dict]] = {}
    for combo in combos:
        combos_by_case.setdefault(combo["embedding_case"], []).append(combo)

    total_new = 0
    total_skipped = 0

    for case, case_combos in combos_by_case.items():
        matrix, _ = load_user_matrix(case)
        if matrix.shape[0] == 0:
            print(f"[cluster_search:{case}] no embeddings — skipping {len(case_combos)} combos")
            total_skipped += len(case_combos)
            continue

        for combo in case_combos:
            session = get_session()
            try:
                if session.query(ClusterRun).filter_by(**combo).first():
                    total_skipped += 1
                    continue
            finally:
                session.close()

            params = {k: v for k, v in combo.items() if k != "embedding_case"}
            try:
                result = compute_clusters(matrix, **params)
            except ValueError as exc:
                print(f"[cluster_search:{case}] skipping — {exc}")
                total_skipped += 1
                continue

            sizes = result.cluster_sizes
            row = ClusterRun(
                **combo,
                n_clusters=result.n_clusters,
                noise_ratio=round(result.noise_ratio, 4),
                min_size=min(sizes) if sizes else 0,
                median_size=int(np.median(sizes)) if sizes else 0,
                max_size=max(sizes) if sizes else 0,
            )
            session = get_session()
            try:
                session.add(row)
                session.commit()
                total_new += 1
            except IntegrityError:
                session.rollback()
                total_skipped += 1
            finally:
                session.close()

    print(f"[cluster_search] done — {total_new} new, {total_skipped} skipped")


def validate_clustering() -> dict[str, dict]:
    """Select best clustering params per embedding_case from ClusterRun results.

    Contract (not yet implemented):
    - Query ClusterRun for all completed runs.
    - For each embedding_case, select the param combo with the lowest noise_ratio
      where n_clusters falls within a configurable target range.
    - Return {embedding_case: params_dict} where params_dict is suitable for
      passing directly as **kwargs to cluster_users().

    Implement this once run_cluster_search() has populated enough ClusterRun rows
    to make a meaningful selection.
    """
    raise NotImplementedError(
        "validate_clustering() is not implemented. "
        "Run run_cluster_search() first to populate ClusterRun, "
        "then implement selection logic here and pass best params to cluster_users()."
    )
