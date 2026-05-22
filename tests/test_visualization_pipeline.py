"""Tests for modules.visualization.pipeline.run_visualization."""

from __future__ import annotations

from core import fingerprint as fp
from core.database import (
    Base,
    StageState,
    User,
    UserCluster,
    Visualization,
    VisualizationCluster,
    VisualizationUser,
    get_engine,
    get_session,
)
from modules.visualization.pipeline import run_visualization


def _clear() -> None:
    Base.metadata.create_all(get_engine())
    session = get_session()
    try:
        for m in (
            VisualizationUser,
            VisualizationCluster,
            Visualization,
            UserCluster,
            StageState,
            User,
        ):
            session.query(m).delete()
        session.commit()
    finally:
        session.close()


def _seed_user_clusters(
    case: str, n_per_cluster: int = 5, n_clusters: int = 3, noise: int = 2
) -> None:
    """Seed UserCluster rows and the upstream cluster_assign StageState."""
    session = get_session()
    try:
        uid = 0
        for cid in range(n_clusters):
            for _ in range(n_per_cluster):
                session.merge(User(id=uid))
                session.add(
                    UserCluster(
                        user_id=uid,
                        embedding_case=case,
                        cluster_id=cid,
                        umap_x=float(cid * 10 + uid * 0.1),
                        umap_y=float(cid * 10 - uid * 0.1),
                    )
                )
                uid += 1
        for _ in range(noise):
            session.merge(User(id=uid))
            session.add(
                UserCluster(
                    user_id=uid,
                    embedding_case=case,
                    cluster_id=-1,
                    umap_x=float(-uid),
                    umap_y=float(uid),
                )
            )
            uid += 1
        fp.mark_complete(
            session,
            "cluster_assign",
            case,
            fp.Fingerprint(data="d", config="c", dependency="x"),
        )
        session.commit()
    finally:
        session.close()


def test_run_visualization_skips_when_upstream_state_missing():
    _clear()
    # No StageState("cluster_assign", "video") seeded.
    run_visualization(cases=("video",))
    session = get_session()
    try:
        assert session.get(StageState, ("visualization", "video")) is None
        assert session.query(Visualization).count() == 0
    finally:
        session.close()


def test_run_visualization_materializes_rows_for_seeded_case():
    _clear()
    _seed_user_clusters("video", n_per_cluster=5, n_clusters=3, noise=2)
    run_visualization(cases=("video",))
    session = get_session()
    try:
        viz = session.get(Visualization, "video")
        assert viz is not None
        assert viz.label == "Visual"
        # 3 clusters * 5 users + 2 noise = 17 user rows total.
        assert viz.size == 17
        assert (
            session.query(VisualizationUser).filter_by(embedding_case="video").count()
            == 17
        )
        # Noise excluded from cluster table; 3 real clusters remain.
        assert (
            session.query(VisualizationCluster)
            .filter_by(embedding_case="video")
            .count()
            == 3
        )
        # StageState seal written.
        assert session.get(StageState, ("visualization", "video")) is not None
    finally:
        session.close()


def test_run_visualization_is_idempotent_when_data_unchanged():
    _clear()
    _seed_user_clusters("video", n_per_cluster=5, n_clusters=3, noise=2)
    run_visualization(cases=("video",))
    session = get_session()
    try:
        first_hash = session.get(StageState, ("visualization", "video")).data_hash
        first_users = {
            (u.user_id, u.cluster_id)
            for u in session.query(VisualizationUser)
            .filter_by(embedding_case="video")
            .all()
        }
    finally:
        session.close()

    run_visualization(cases=("video",))
    session = get_session()
    try:
        second_hash = session.get(StageState, ("visualization", "video")).data_hash
        second_users = {
            (u.user_id, u.cluster_id)
            for u in session.query(VisualizationUser)
            .filter_by(embedding_case="video")
            .all()
        }
    finally:
        session.close()
    assert first_hash == second_hash
    assert first_users == second_users


def test_run_visualization_re_exports_when_upstream_row_mutates():
    _clear()
    _seed_user_clusters("video", n_per_cluster=5, n_clusters=3, noise=2)
    run_visualization(cases=("video",))
    session = get_session()
    try:
        first_hash = session.get(StageState, ("visualization", "video")).data_hash
        # Move user 0 to a wildly different position.
        target = (
            session.query(UserCluster)
            .filter_by(embedding_case="video", user_id=0)
            .one()
        )
        target.umap_x = 9999.0
        session.commit()
    finally:
        session.close()

    run_visualization(cases=("video",))
    session = get_session()
    try:
        second_hash = session.get(StageState, ("visualization", "video")).data_hash
        new_x = (
            session.query(VisualizationUser)
            .filter_by(embedding_case="video", user_id=0)
            .one()
            .x
        )
    finally:
        session.close()
    assert first_hash != second_hash
    assert new_x == 9999.0


def test_run_visualization_handles_empty_case():
    """Upstream sealed but no UserCluster rows → size=0 Visualization row."""
    _clear()
    session = get_session()
    try:
        fp.mark_complete(
            session,
            "cluster_assign",
            "video",
            fp.Fingerprint(data="d", config="c", dependency="x"),
        )
        session.commit()
    finally:
        session.close()

    run_visualization(cases=("video",))
    session = get_session()
    try:
        viz = session.get(Visualization, "video")
        assert viz is not None
        assert viz.size == 0
        assert session.query(VisualizationUser).count() == 0
        assert session.query(VisualizationCluster).count() == 0
        assert session.get(StageState, ("visualization", "video")) is not None
    finally:
        session.close()


def test_run_visualization_writes_db_rows_even_when_case_hidden():
    """expose_to_viewer=False does not skip the DB write (only the export)."""
    _clear()
    _seed_user_clusters("gemini", n_per_cluster=4, n_clusters=2, noise=1)
    run_visualization(cases=("gemini",))
    session = get_session()
    try:
        viz = session.get(Visualization, "gemini")
        assert viz is not None
        assert viz.label == "Gemini"
        assert viz.size == 9
        assert (
            session.query(VisualizationCluster)
            .filter_by(embedding_case="gemini")
            .count()
            == 2
        )
    finally:
        session.close()
