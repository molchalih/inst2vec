"""Tests for modules.clustering.assign."""

from __future__ import annotations

import numpy as np

from modules.clustering import assign_clusters
from modules.database import (
    Base,
    Clip,
    ClusterRun,
    StageState,
    User,
    UserCluster,
    UserEmbedding,
    get_engine,
    get_session,
)


def _seed_case(session, case: str, n_users: int = 30, *, with_best_run: bool):
    for uid in range(n_users):
        session.merge(User(id=uid))
        session.merge(
            Clip(id=1000 + uid, user_id=uid, is_selected=True, is_downloaded=True)
        )
        session.merge(
            UserEmbedding(
                user_id=uid,
                embedding_case=case,
                embedding=np.random.default_rng(uid)
                .standard_normal(8)
                .astype(np.float32)
                .tobytes(),
            )
        )
    if with_best_run:
        session.add(
            ClusterRun(
                embedding_case=case,
                umap_n_components=3,
                umap_n_neighbors=5,
                umap_min_dist=0.1,
                umap_metric="cosine",
                umap2d_n_neighbors=5,
                umap2d_min_dist=0.1,
                umap2d_metric="cosine",
                hdbscan_min_cluster_size=5,
                hdbscan_min_samples=None,
                hdbscan_cluster_selection_method="eom",
                hdbscan_metric="euclidean",
                random_state=42,
                n_clusters=3,
                noise_ratio=0.05,
                min_size=5,
                median_size=10,
                max_size=15,
                passes_validation=True,
                dbcv=0.5,
                silhouette=0.4,
                param_plateau_score=0.48,
            )
        )
    session.commit()


def _clear():
    Base.metadata.create_all(get_engine())
    session = get_session()
    try:
        for m in (UserCluster, ClusterRun, StageState, UserEmbedding, Clip, User):
            session.query(m).delete()
        session.commit()
    finally:
        session.close()


def test_assign_without_validation_state_does_not_seal():
    """If cluster_validation StageState absent: assign skips without sealing."""
    _clear()
    session = get_session()
    try:
        _seed_case(session, "video", with_best_run=True)
    finally:
        session.close()
    # NOTE: no StageState("cluster_validation", "video") written
    assign_clusters()
    session = get_session()
    try:
        assert session.get(StageState, ("cluster_assign", "video")) is None
        assert session.query(UserCluster).filter_by(embedding_case="video").count() == 0
    finally:
        session.close()


def test_assign_seals_empty_clusters_when_no_best_run():
    """If validation state exists but no best run: delete UserClusters, seal."""
    from modules import fingerprint as fp

    _clear()
    session = get_session()
    try:
        _seed_case(session, "video", with_best_run=False)
        fp.mark_complete(
            session,
            "cluster_validation",
            "video",
            fp.Fingerprint(data="d", config="c", dependency="x"),
        )
        session.commit()
    finally:
        session.close()

    assign_clusters()
    session = get_session()
    try:
        assert session.get(StageState, ("cluster_assign", "video")) is not None
        assert session.query(UserCluster).filter_by(embedding_case="video").count() == 0
    finally:
        session.close()


def test_assign_creates_user_clusters_for_best_run():
    """Best run exists: UserCluster rows materialized."""
    from modules import fingerprint as fp

    _clear()
    session = get_session()
    try:
        _seed_case(session, "video", with_best_run=True)
        fp.mark_complete(
            session,
            "cluster_validation",
            "video",
            fp.Fingerprint(data="d", config="c", dependency="x"),
        )
        session.commit()
    finally:
        session.close()

    assign_clusters()
    session = get_session()
    try:
        n = session.query(UserCluster).filter_by(embedding_case="video").count()
        assert n == 30
    finally:
        session.close()


def test_unchanged_fingerprint_skips_assign():
    """Second assign call with unchanged validation state is a no-op."""
    from modules import fingerprint as fp

    _clear()
    session = get_session()
    try:
        _seed_case(session, "video", with_best_run=True)
        fp.mark_complete(
            session,
            "cluster_validation",
            "video",
            fp.Fingerprint(data="d", config="c", dependency="x"),
        )
        session.commit()
    finally:
        session.close()

    assign_clusters()
    session = get_session()
    try:
        first = {
            (uc.user_id, uc.cluster_id)
            for uc in session.query(UserCluster).filter_by(embedding_case="video").all()
        }
    finally:
        session.close()

    assign_clusters()
    session = get_session()
    try:
        second = {
            (uc.user_id, uc.cluster_id)
            for uc in session.query(UserCluster).filter_by(embedding_case="video").all()
        }
    finally:
        session.close()
    assert first == second
