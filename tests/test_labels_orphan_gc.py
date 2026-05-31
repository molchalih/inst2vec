"""Orphan GC for clip_labels / cluster_labels.

A clip that becomes unselected, a user removed from data.csv, or a
cluster that vanishes after re-clustering must not leave dead label
rows behind. Stage-1 / stage-2 fingerprints are config+dep only with
respect to *missing* rows, so no drift fires — GC has to run unconditionally
at the top of ``modules.labels.pipeline.run``.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from core.database import (
    Clip,
    ClipLabel,
    ClusterLabel,
    User,
    UserCluster,
    init_db,
)
from core.database.engine import get_engine
from modules.labels.gc import purge_orphans


def _make_user(session: Session, *, user_id: int) -> User:
    user = User(id=user_id)
    session.add(user)
    return user


def _make_clip(session: Session, *, clip_id: int, user_id: int, selected: bool) -> Clip:
    clip = Clip(
        id=clip_id,
        user_id=user_id,
        is_selected=selected,
        is_downloaded=True,
    )
    session.add(clip)
    return clip


def test_purge_orphans_removes_unselected_clip_labels() -> None:
    init_db("sqlite:///:memory:", "sqlite:///:memory:")
    with Session(get_engine()) as session:
        _make_user(session, user_id=1)
        _make_clip(session, clip_id=10, user_id=1, selected=True)
        _make_clip(session, clip_id=11, user_id=1, selected=False)
        session.add_all(
            [
                ClipLabel(clip_id=10, label_case="video", status="success", attempts=1),
                ClipLabel(clip_id=11, label_case="video", status="success", attempts=1),
            ]
        )
        session.commit()

        clips_deleted, clusters_deleted = purge_orphans(session)
        session.commit()

        remaining = sorted(
            (r.clip_id, r.label_case) for r in session.query(ClipLabel).all()
        )
        assert remaining == [(10, "video")]
        assert clips_deleted == 1
        assert clusters_deleted == 0


def test_purge_orphans_removes_stale_cluster_labels() -> None:
    init_db("sqlite:///:memory:", "sqlite:///:memory:")
    with Session(get_engine()) as session:
        _make_user(session, user_id=1)
        session.add(
            UserCluster(
                user_id=1,
                embedding_case="video",
                cluster_id=0,
                umap_x=0.0,
                umap_y=0.0,
                centrality=1.0,
            )
        )
        session.add_all(
            [
                ClusterLabel(
                    embedding_case="video", cluster_id=0, status="success", attempts=1
                ),
                ClusterLabel(
                    embedding_case="video", cluster_id=7, status="success", attempts=1
                ),
                ClusterLabel(
                    embedding_case="audio", cluster_id=0, status="success", attempts=1
                ),
            ]
        )
        session.commit()

        _clips_deleted, clusters_deleted = purge_orphans(session)
        session.commit()

        remaining = sorted(
            (r.embedding_case, r.cluster_id) for r in session.query(ClusterLabel).all()
        )
        assert remaining == [("video", 0)]
        assert clusters_deleted == 2


def test_purge_orphans_is_idempotent() -> None:
    init_db("sqlite:///:memory:", "sqlite:///:memory:")
    with Session(get_engine()) as session:
        _make_user(session, user_id=1)
        _make_clip(session, clip_id=10, user_id=1, selected=True)
        _make_clip(session, clip_id=11, user_id=1, selected=False)
        session.add_all(
            [
                ClipLabel(clip_id=10, label_case="video", status="success", attempts=1),
                ClipLabel(clip_id=11, label_case="video", status="success", attempts=1),
            ]
        )
        session.commit()

        first = purge_orphans(session)
        session.commit()
        assert first == (1, 0)

        second = purge_orphans(session)
        session.commit()
        assert second == (0, 0)

        remaining = sorted(
            (r.clip_id, r.label_case) for r in session.query(ClipLabel).all()
        )
        assert remaining == [(10, "video")]


def test_purge_orphans_treats_noise_cluster_as_dead() -> None:
    """HDBSCAN noise label (cluster_id=-1) is not a live cluster.

    Any cluster_labels row keyed on a negative cluster_id must be
    purged even if a UserCluster with cluster_id=-1 exists.
    """
    init_db("sqlite:///:memory:", "sqlite:///:memory:")
    with Session(get_engine()) as session:
        _make_user(session, user_id=1)
        session.add(
            UserCluster(
                user_id=1,
                embedding_case="video",
                cluster_id=-1,
                centrality=0.0,
                umap_x=0.0,
                umap_y=0.0,
            )
        )
        session.add(
            ClusterLabel(
                embedding_case="video",
                cluster_id=-1,
                status="success",
                attempts=1,
            )
        )
        session.commit()

        _, clusters_deleted = purge_orphans(session)
        session.commit()

        assert clusters_deleted == 1
        assert session.query(ClusterLabel).count() == 0


def test_purge_orphans_no_selected_clips_deletes_all_clip_labels() -> None:
    init_db("sqlite:///:memory:", "sqlite:///:memory:")
    with Session(get_engine()) as session:
        _make_user(session, user_id=1)
        _make_clip(session, clip_id=10, user_id=1, selected=False)
        session.add(
            ClipLabel(clip_id=10, label_case="video", status="success", attempts=1)
        )
        session.commit()

        clips_deleted, _ = purge_orphans(session)
        session.commit()

        assert clips_deleted == 1
        assert session.query(ClipLabel).count() == 0


def test_purge_orphans_no_live_clusters_deletes_all_cluster_labels() -> None:
    init_db("sqlite:///:memory:", "sqlite:///:memory:")
    with Session(get_engine()) as session:
        session.add(
            ClusterLabel(
                embedding_case="video", cluster_id=0, status="success", attempts=1
            )
        )
        session.commit()

        _, clusters_deleted = purge_orphans(session)
        session.commit()

        assert clusters_deleted == 1
        assert session.query(ClusterLabel).count() == 0


def test_purge_drops_label_for_selected_but_undownloaded_clip() -> None:
    """A selected-but-undownloaded clip is not in analysis; its stray label goes.

    Locks gc to the canonical ``clip_used_in_analysis()`` predicate
    (is_selected AND is_downloaded), matching the clip pass.
    """
    init_db("sqlite:///:memory:", "sqlite:///:memory:")
    with Session(get_engine()) as session:
        _make_user(session, user_id=1)
        session.add(Clip(id=10, user_id=1, is_selected=True, is_downloaded=False))
        session.add(
            ClipLabel(clip_id=10, label_case="video", status="success", attempts=1)
        )
        session.commit()

        purge_orphans(session)
        session.commit()

        assert session.get(ClipLabel, (10, "video")) is None
