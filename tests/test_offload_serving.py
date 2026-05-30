"""Decomposing offload: main DB → normalised serving tables.

The offload calls the version-6 builders, decomposes each payload dict into
``serving_*`` rows, and writes them idempotently (delete-then-insert per
run). These tests assert row counts/ordering match the builder payloads,
re-running is stable, and dropping a case prunes its rows.
"""

from __future__ import annotations

from core.database import (
    ServingCluster,
    ServingRun,
    ServingUser,
    ServingWeightedTag,
    get_serving_session,
    get_session,
    init_serving_db,
)
from modules.visualization import export as export_mod
from tests.test_visualization_export import (
    _clear,
    _seed_case_with_mir,
    _settings,
)


def _bundle(tmp_path, case="video"):
    session = get_session()
    try:
        return export_mod.build_case_payloads(
            session, settings_viz=_settings(tmp_path).visualization, case=case
        )
    finally:
        session.close()


def _serving(tmp_path):
    init_serving_db(f"sqlite:///{tmp_path / 'serving.db'}")


def test_offload_populates_run_users_clusters(tmp_path):
    from scripts.offload_serving import offload

    _clear()
    _seed_case_with_mir("video", n_users=4)
    _serving(tmp_path)
    offload(_settings(tmp_path), cases=("video",))

    bundle = _bundle(tmp_path)
    with get_serving_session() as s:
        runs = s.query(ServingRun).all()
        assert [r.run_id for r in runs] == ["video"]
        assert runs[0].size == bundle.manifest_entry["size"]
        assert s.query(ServingUser).count() == len(bundle.users["users"])
        assert s.query(ServingCluster).count() == len(bundle.clusters["clusters"])


def test_offload_weighted_tag_order_matches_payload(tmp_path):
    from scripts.offload_serving import offload

    _clear()
    _seed_case_with_mir("video", n_users=4)
    _serving(tmp_path)
    offload(_settings(tmp_path), cases=("video",))

    bundle = _bundle(tmp_path)
    # Pick a cluster with a non-empty genre_top and compare ordered labels.
    cid, detail = next(
        (cid, d) for cid, d in bundle.cluster_details.items() if d["genre_top"]
    )
    with get_serving_session() as s:
        rows = (
            s.query(ServingWeightedTag)
            .filter_by(
                run_id="video", owner_kind="cluster", owner_id=cid, field="genre"
            )
            .order_by(ServingWeightedTag.ord)
            .all()
        )
    assert [r.label for r in rows] == [g["label"] for g in detail["genre_top"]]
    assert [r.weight for r in rows] == [g["weight"] for g in detail["genre_top"]]


def test_offload_is_idempotent(tmp_path):
    from scripts.offload_serving import offload

    _clear()
    _seed_case_with_mir("video", n_users=4)
    _serving(tmp_path)
    offload(_settings(tmp_path), cases=("video",))
    with get_serving_session() as s:
        first = (s.query(ServingUser).count(), s.query(ServingWeightedTag).count())
    offload(_settings(tmp_path), cases=("video",))
    with get_serving_session() as s:
        second = (s.query(ServingUser).count(), s.query(ServingWeightedTag).count())
    assert first == second


def _add_case_for_existing_users(case: str, *, n_users: int = 4) -> None:
    """Add a second embedding-case view (Visualization + viz/cluster rows)
    over the SAME users/clips a prior `_seed_case_with_mir("video")` laid down.

    Embedding cases share clips; only UserCluster / Visualization* rows are
    per-case, so this mirrors how a real multi-case DB looks without colliding
    on the globally-unique clip ids.
    """
    from core.database import (
        UserCluster,
        Visualization,
        VisualizationCluster,
        VisualizationUser,
    )

    session = get_session()
    try:
        session.merge(
            Visualization(
                embedding_case=case, label="Combined", size=n_users, source_hash="h2"
            )
        )
        for uid in range(n_users):
            session.add(
                UserCluster(
                    user_id=uid,
                    embedding_case=case,
                    cluster_id=uid % 2,
                    umap_x=float(uid),
                    umap_y=float(-uid),
                )
            )
            session.add(
                VisualizationUser(
                    user_id=uid,
                    embedding_case=case,
                    x=float(uid),
                    y=float(-uid),
                    cluster_id=uid % 2,
                )
            )
        for cid in range(2):
            session.add(
                VisualizationCluster(
                    embedding_case=case,
                    cluster_id=cid,
                    cx=float(cid),
                    cy=float(cid * 2),
                    rx=1.0,
                    ry=2.0,
                    angle=0.0,
                    size=n_users // 2,
                    label=f"Cluster {cid + 1}",
                )
            )
        session.commit()
    finally:
        session.close()


def test_offload_prunes_dropped_case(tmp_path):
    from scripts.offload_serving import offload

    _clear()
    _seed_case_with_mir("video", n_users=4)
    _add_case_for_existing_users("sandwich", n_users=4)
    _serving(tmp_path)
    offload(_settings(tmp_path), cases=("video", "sandwich"))
    with get_serving_session() as s:
        assert {r.run_id for r in s.query(ServingRun).all()} == {"video", "sandwich"}

    # Drop sandwich from the main DB and re-offload; its serving rows go away.
    _clear()
    _seed_case_with_mir("video", n_users=4)
    offload(_settings(tmp_path), cases=("video", "sandwich"))
    with get_serving_session() as s:
        assert {r.run_id for r in s.query(ServingRun).all()} == {"video"}
        assert s.query(ServingUser).filter_by(run_id="sandwich").count() == 0
