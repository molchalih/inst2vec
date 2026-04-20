"""Grid search over UMAP + HDBSCAN hyperparameters; saves aggregate metrics to ClusterRun."""
import hashlib
import os
from itertools import product

import numpy as np
from sqlalchemy.exc import IntegrityError

from modules.console import progress
from modules.database import Base, engine, get_session, ClusterRun
from modules.clustering import compute_clusters, load_user_matrix
from modules.services import log


def _parse_ints(val: str) -> list[int]:
    return [int(x) for x in val.split()]


def _parse_floats(val: str) -> list[float]:
    return [float(x) for x in val.split()]


def _parse_strs(val: str) -> list[str]:
    return val.split()


_PARAM_KEYS = (
    "umap_n_components", "umap_n_neighbors", "umap_min_dist", "umap_metric",
    "umap2d_n_neighbors", "umap2d_min_dist", "umap2d_metric",
    "hdbscan_min_cluster_size", "hdbscan_min_samples",
    "hdbscan_cluster_selection_method", "hdbscan_metric", "random_state",
)


def _compute_dataset_fingerprint(user_pks: list[int]) -> str:
    """Compute a deterministic SHA-256 fingerprint of the dataset.

    The fingerprint is based on sorted user PKs, ensuring it's order-independent.
    """
    payload = ",".join(str(pk) for pk in sorted(user_pks)).encode()
    return hashlib.sha256(payload).hexdigest()


def _combo_key(combo: dict) -> frozenset:
    """Extract the hyperparameter key from a combo, excluding embedding_case.

    Returns a frozenset of (key, value) tuples for all params except embedding_case.
    This is used to check if a stored row matches the current grid configuration.
    """
    return frozenset((k, combo[k]) for k in _PARAM_KEYS)


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

    At the start of each run: marks existing rows as in_current_grid=0/disqualified=1
    if they belong to a combo not in the current grid, or were computed on a different
    dataset (fingerprint mismatch). New rows get in_current_grid=1 and dataset_fingerprint.

    Idempotent: skips any combo already present in the DB with matching fingerprint.
    Groups combos by embedding_case so the user embedding matrix is loaded once per case.
    """
    Base.metadata.create_all(engine)
    combos = _load_grid()

    combos_by_case: dict[str, list[dict]] = {}
    for combo in combos:
        combos_by_case.setdefault(combo["embedding_case"], []).append(combo)

    total_new = 0
    total_skipped = 0

    for case, case_combos in combos_by_case.items():
        matrix, user_pks = load_user_matrix(case)
        if matrix.shape[0] == 0:
            log(f"cluster_search:{case}", f"no embeddings — skipping {len(case_combos)} combos", level="warn")
            total_skipped += len(case_combos)
            continue

        fingerprint = _compute_dataset_fingerprint(user_pks)
        current_keys = {_combo_key(c) for c in case_combos}

        # Invalidate stale rows: wrong grid params or dataset changed
        session = get_session()
        try:
            existing_rows = session.query(ClusterRun).filter(
                ClusterRun.embedding_case == case
            ).all()
            for row in existing_rows:
                row_key = frozenset(
                    (k, getattr(row, k)) for k in _PARAM_KEYS
                )
                in_grid = row_key in current_keys
                fp_match = row.dataset_fingerprint == fingerprint
                if in_grid and fp_match:
                    row.in_current_grid = 1
                else:
                    row.in_current_grid = 0
                    row.disqualified = 1
            session.commit()
        finally:
            session.close()

        with progress(len(case_combos), f"cluster search · {case}") as advance:
            for combo in case_combos:
                short = (
                    f"nc={combo['umap_n_components']} nn={combo['umap_n_neighbors']} "
                    f"mcs={combo['hdbscan_min_cluster_size']}"
                )
                params = {k: v for k, v in combo.items() if k != "embedding_case"}
                session = get_session()
                try:
                    # Skip if already computed for the current fingerprint
                    if session.query(ClusterRun).filter_by(**combo).filter(
                        ClusterRun.dataset_fingerprint == fingerprint
                    ).first():
                        total_skipped += 1
                        advance(1, detail=f"{short} | cached")
                        continue

                    try:
                        result = compute_clusters(matrix, **params)
                    except ValueError as exc:
                        log(f"cluster_search:{case}", f"skipping — {exc}", level="warn")
                        total_skipped += 1
                        advance(1, detail=f"{short} | skip {str(exc)[:48]}")
                        continue

                    sizes = result.cluster_sizes
                    row = ClusterRun(
                        **combo,
                        n_clusters=result.n_clusters,
                        noise_ratio=round(result.noise_ratio, 4),
                        min_size=min(sizes) if sizes else 0,
                        median_size=int(np.median(sizes)) if sizes else 0,
                        max_size=max(sizes) if sizes else 0,
                        in_current_grid=1,
                        dataset_fingerprint=fingerprint,
                    )
                    session.add(row)
                    session.commit()
                    total_new += 1
                    advance(1, detail=f"{short} | k={result.n_clusters} new")
                except IntegrityError:
                    session.rollback()
                    # A stale row with the same hyperparams exists (non-None hdbscan_min_samples
                    # case — SQLite unique constraint fires). Update it in-place instead.
                    stale_row = session.query(ClusterRun).filter_by(**combo).first()
                    if stale_row is not None:
                        try:
                            result = compute_clusters(matrix, **params)
                        except ValueError as exc:
                            log(f"cluster_search:{case}", f"update-skip — {exc}", level="warn")
                            total_skipped += 1
                            advance(1, detail=f"{short} | upd-skip {str(exc)[:40]}")
                            continue
                        sizes = result.cluster_sizes
                        stale_row.n_clusters = result.n_clusters
                        stale_row.noise_ratio = round(result.noise_ratio, 4)
                        stale_row.min_size = min(sizes) if sizes else 0
                        stale_row.median_size = int(np.median(sizes)) if sizes else 0
                        stale_row.max_size = max(sizes) if sizes else 0
                        stale_row.in_current_grid = 1
                        stale_row.dataset_fingerprint = fingerprint
                        stale_row.disqualified = None
                        stale_row.dbcv = None
                        stale_row.silhouette = None
                        stale_row.composite_score = None
                        stale_row.param_plateau_score = None
                        session.commit()
                        total_new += 1
                        advance(1, detail=f"{short} | k={result.n_clusters} updated")
                    else:
                        total_skipped += 1
                        advance(1, detail=f"{short} | integrity skip")
                finally:
                    session.close()

    log("cluster_search", f"done — {total_new} new, {total_skipped} skipped", level="ok")
