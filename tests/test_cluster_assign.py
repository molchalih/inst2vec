"""Tests for modules.clustering.assign."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from core.config import ValidationSettings
from core.database import (
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
from modules.clustering import assign_clusters


def _wrap(validation: ValidationSettings):
    """Wrap a ValidationSettings into a settings namespace.

    We still pin validation knobs via the Pydantic model so per-field
    defaults stay realistic. A ``.search`` slice carries the preprocess map
    and max_cluster_frac that assign_clusters now reads.
    """
    return SimpleNamespace(
        validation=validation,
        search=SimpleNamespace(hdbscan_max_cluster_frac=0.0, embedding_preprocess={}),
    )


DEFAULT_VALIDATION = ValidationSettings(
    plateau_drop_threshold=0.05,
    max_noise_ratio=0.5,
    min_clusters=2,
    max_clusters=50,
)
DEFAULT_SETTINGS = _wrap(DEFAULT_VALIDATION)
DEFAULT_CASES = ("video",)


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
    assign_clusters(settings=DEFAULT_SETTINGS, cases=DEFAULT_CASES)
    session = get_session()
    try:
        assert session.get(StageState, ("cluster_assign", "video")) is None
        assert session.query(UserCluster).filter_by(embedding_case="video").count() == 0
    finally:
        session.close()


def test_assign_seals_empty_clusters_when_no_best_run():
    """If validation state exists but no best run: delete UserClusters, seal."""
    from core import fingerprint as fp

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

    assign_clusters(settings=DEFAULT_SETTINGS, cases=DEFAULT_CASES)
    session = get_session()
    try:
        assert session.get(StageState, ("cluster_assign", "video")) is not None
        assert session.query(UserCluster).filter_by(embedding_case="video").count() == 0
    finally:
        session.close()


def test_assign_creates_user_clusters_for_best_run():
    """Best run exists: UserCluster rows materialized."""
    from core import fingerprint as fp

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

    assign_clusters(settings=DEFAULT_SETTINGS, cases=DEFAULT_CASES)
    session = get_session()
    try:
        n = session.query(UserCluster).filter_by(embedding_case="video").count()
        assert n == 30
    finally:
        session.close()


def test_unchanged_fingerprint_skips_assign():
    """Second assign call with unchanged validation state is a no-op."""
    from core import fingerprint as fp

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

    assign_clusters(settings=DEFAULT_SETTINGS, cases=DEFAULT_CASES)
    session = get_session()
    try:
        first = {
            (uc.user_id, uc.cluster_id)
            for uc in session.query(UserCluster).filter_by(embedding_case="video").all()
        }
    finally:
        session.close()

    assign_clusters(settings=DEFAULT_SETTINGS, cases=DEFAULT_CASES)
    session = get_session()
    try:
        second = {
            (uc.user_id, uc.cluster_id)
            for uc in session.query(UserCluster).filter_by(embedding_case="video").all()
        }
    finally:
        session.close()
    assert first == second


def _add_run(
    session,
    case: str,
    *,
    umap_n_components: int,
    dbcv: float,
    param_plateau_score: float,
) -> int:
    """Insert a fully-scored ClusterRun and return its id.

    Param values differ across rows so the uq_cluster_runs_params constraint
    is satisfied; umap_n_components is varied per-call.
    """
    run = ClusterRun(
        embedding_case=case,
        umap_n_components=umap_n_components,
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
        dbcv=dbcv,
        silhouette=0.4,
        param_plateau_score=param_plateau_score,
    )
    session.add(run)
    session.commit()
    return run.id


def test_assign_honors_configured_plateau_threshold():
    """Threshold from settings drives best-run selection in assign."""
    from core import fingerprint as fp

    _clear()
    case = "video"
    session = get_session()
    try:
        _seed_case(session, case, with_best_run=False)
        # Run A: lower DBCV but on the plateau (drop ≈ 0).
        run_a = _add_run(
            session,
            case,
            umap_n_components=3,
            dbcv=0.50,
            param_plateau_score=0.50,
        )
        # Run B: higher DBCV but a 0.20 drop vs neighbors.
        run_b = _add_run(
            session,
            case,
            umap_n_components=4,
            dbcv=0.70,
            param_plateau_score=0.50,
        )
        fp.mark_complete(
            session,
            "cluster_validation",
            case,
            fp.Fingerprint(data="d", config="c", dependency="x"),
        )
        session.commit()
    finally:
        session.close()

    strict = _wrap(
        ValidationSettings(
            plateau_drop_threshold=0.05,
            max_noise_ratio=0.5,
            min_clusters=2,
            max_clusters=50,
        )
    )
    assign_clusters(settings=strict, cases=(case,))

    relaxed = _wrap(
        ValidationSettings(
            plateau_drop_threshold=0.5,
            max_noise_ratio=0.5,
            min_clusters=2,
            max_clusters=50,
        )
    )
    assign_clusters(settings=relaxed, cases=(case,))

    # Sanity: distinct ids so the assertions below mean something.
    assert run_a != run_b
    # If threshold were ignored (hardcoded 0.05), the second call would be
    # treated as a no-op and the assignment would still reflect run_a.  With
    # threshold threaded through, the relaxed call must reseal and the
    # fingerprint config hash must reflect the new threshold value.
    session = get_session()
    try:
        state = session.get(StageState, ("cluster_assign", case))
        assert state is not None
        relaxed_config_hash = state.config_hash
    finally:
        session.close()

    # Re-run with strict settings and confirm the config hash flips back.
    assign_clusters(settings=strict, cases=(case,))
    session = get_session()
    try:
        state = session.get(StageState, ("cluster_assign", case))
        assert state is not None
        assert state.config_hash != relaxed_config_hash
    finally:
        session.close()
