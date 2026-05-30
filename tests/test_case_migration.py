"""Tests for the maest→auditory re-key (no recompute) + audio GC migration.

The migration must:
  * Re-key every ``maest`` row across the audited tables to ``auditory``
    WITHOUT touching the stored bytes (no recompute).
  * Adopt the embed + cluster_labels seals (their config_hash folds the
    literal case name, so a naive scope_key rename would leave a stale
    config_hash and force recompute).
  * Re-derive the visualization seal's dependency_hash from the adopted
    upstream rows so visualization also stays frozen.
  * DELETE every ``audio`` row (audio is removed, not renamed).
  * Be idempotent (a second run is a no-op).
"""

from __future__ import annotations

from core import fingerprint as fp
from core.config import LabelsSettings
from core.database import (
    Base,
    Clip,
    ClipEmbedding,
    ClipLabel,
    ClusterLabel,
    ClusterRun,
    StageState,
    User,
    UserCluster,
    UserEmbedding,
    Visualization,
    VisualizationCluster,
    VisualizationUser,
    get_engine,
    get_session,
)
from core.database.case_migration import (
    legacy_cluster_labels_payload,
    run_case_migration,
)
from modules.labels.state import cluster_labels_config_payload

EMBED_STAGE = "clip_embeddings"
CLUSTER_LABELS_STAGE = "cluster_labels"
LABELS_STAGE = "labels"
VIZ_STAGE = "visualization"

# Hashes the production wiring computes from the renamed registry/config and
# hands to the migration as the adopted seals. The test only needs stable,
# distinct sentinels.
ADOPTED_EMBED_CONFIG = "adopted-embed-config-hash"
ADOPTED_CLUSTER_LABELS_CONFIG = "adopted-cluster-labels-config-hash"

# The expected-legacy hashes: what the OLD maest recipe produces under CURRENT
# settings. ``_seed_case`` seals the maest config_hash to these so the common
# (no-drift) path adopts the auditory hash. Tests that simulate a drifted
# recipe override the stored maest config_hash to something else.
LEGACY_EMBED_CONFIG = "embed-config-maest"
LEGACY_CLUSTER_LABELS_CONFIG = "cl-config-maest"


def _fresh_db():
    """Drop + recreate the shared in-memory schema and return a session."""
    engine = get_engine()
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    return get_session()


def _seed_case(session, case: str) -> None:
    """Seed one row per audited table for ``case`` with case-distinct content."""
    uid = 1 if case == "maest" else 2
    cid = 10 if case == "maest" else 20
    session.merge(User(id=uid))
    session.merge(Clip(id=cid, user_id=uid, is_selected=True, is_downloaded=True))
    session.merge(
        ClipEmbedding(
            clip_id=cid,
            embedding_case=case,
            embedding=f"vec-{case}".encode(),
            source_hash=f"src-{case}",
        )
    )
    session.merge(
        UserEmbedding(
            user_id=uid,
            embedding_case=case,
            embedding=f"uvec-{case}".encode(),
            source_hash=f"usrc-{case}",
        )
    )
    session.merge(
        UserCluster(
            user_id=uid,
            embedding_case=case,
            cluster_id=0,
            umap_x=1.0,
            umap_y=2.0,
            centrality=0.5,
        )
    )
    session.add(
        ClusterRun(
            embedding_case=case,
            umap_n_components=2,
            umap_n_neighbors=15,
            umap_min_dist=0.1,
            umap_metric="euclidean",
            umap2d_n_neighbors=15,
            umap2d_min_dist=0.1,
            umap2d_metric="euclidean",
            hdbscan_min_cluster_size=5,
            hdbscan_min_samples=None,
            hdbscan_cluster_selection_method="eom",
            hdbscan_metric="euclidean",
            random_state=42,
            n_clusters=3,
            noise_ratio=0.1,
            min_size=2,
            median_size=5,
            max_size=9,
        )
    )
    session.merge(
        Visualization(
            embedding_case=case, label=case.title(), size=1, source_hash=f"vh-{case}"
        )
    )
    session.merge(
        VisualizationUser(
            user_id=uid, embedding_case=case, x=1.0, y=2.0, cluster_id=0, centrality=0.5
        )
    )
    session.merge(
        VisualizationCluster(
            embedding_case=case,
            cluster_id=0,
            cx=1.0,
            cy=2.0,
            rx=1.0,
            ry=1.0,
            angle=0.0,
            size=1,
            label=case.title(),
        )
    )
    session.merge(
        ClipLabel(clip_id=cid, label_case=case, status="success", payload={"k": case})
    )
    session.merge(
        ClusterLabel(
            embedding_case=case, cluster_id=0, status="success", payload={"k": case}
        )
    )
    # Seals. Use case as scope_key. config_hash for embed + cluster_labels
    # folds the case name (sentinel here); the rest are case-independent.
    session.merge(
        StageState(
            stage_name=EMBED_STAGE,
            scope_key=case,
            data_hash="embed-data",
            config_hash=f"embed-config-{case}",
            dependency_hash="embed-dep",
        )
    )
    session.merge(
        StageState(
            stage_name="user_embeddings",
            scope_key=case,
            data_hash="ue-data",
            config_hash="ue-config",
            dependency_hash="ue-dep",
        )
    )
    session.merge(
        StageState(
            stage_name="cluster_assign",
            scope_key=case,
            data_hash="ca-data",
            config_hash="ca-config",
            dependency_hash="ca-dep",
        )
    )
    session.merge(
        StageState(
            stage_name=CLUSTER_LABELS_STAGE,
            scope_key=case,
            data_hash="cl-data",
            config_hash=f"cl-config-{case}",
            dependency_hash="cl-dep",
        )
    )
    # Visualization seal: dependency chains to cluster_assign + cluster_labels.
    viz_dep = fp.compose_hashes(
        fp.stage_dependency_hash(session, "cluster_assign", case),
        fp.stage_dependency_hash(session, CLUSTER_LABELS_STAGE, case),
    )
    session.merge(
        StageState(
            stage_name=VIZ_STAGE,
            scope_key=case,
            data_hash="viz-data",
            config_hash=fp.hash_text(""),
            dependency_hash=viz_dep,
        )
    )
    session.commit()


def _run(
    session,
    *,
    embed_legacy=LEGACY_EMBED_CONFIG,
    cluster_labels_legacy=LEGACY_CLUSTER_LABELS_CONFIG,
):
    run_case_migration(
        session,
        embed_config_hash=ADOPTED_EMBED_CONFIG,
        embed_legacy_config_hash=embed_legacy,
        cluster_labels_config_hash=ADOPTED_CLUSTER_LABELS_CONFIG,
        cluster_labels_legacy_config_hash=cluster_labels_legacy,
    )


_CASE_TABLES = [
    (ClipEmbedding, "embedding_case"),
    (UserEmbedding, "embedding_case"),
    (UserCluster, "embedding_case"),
    (ClusterRun, "embedding_case"),
    (Visualization, "embedding_case"),
    (VisualizationUser, "embedding_case"),
    (VisualizationCluster, "embedding_case"),
    (ClipLabel, "label_case"),
    (ClusterLabel, "embedding_case"),
]


def _count(session, model, attr, value) -> int:
    return session.query(model).filter(getattr(model, attr) == value).count()


def test_rekey_preserves_bytes_and_counts():
    session = _fresh_db()
    _seed_case(session, "maest")
    _seed_case(session, "audio")

    pre_maest_counts = {
        model.__name__: _count(session, model, attr, "maest")
        for model, attr in _CASE_TABLES
    }
    pre_vec = (
        session.query(ClipEmbedding).filter_by(embedding_case="maest").one().embedding
    )

    _run(session)

    for model, attr in _CASE_TABLES:
        assert _count(session, model, attr, "maest") == 0, model.__name__
        assert _count(session, model, attr, "audio") == 0, model.__name__
        assert (
            _count(session, model, attr, "auditory") == pre_maest_counts[model.__name__]
        ), model.__name__

    post_vec = (
        session.query(ClipEmbedding)
        .filter_by(embedding_case="auditory")
        .one()
        .embedding
    )
    assert post_vec == pre_vec  # byte-identical re-key, not recompute
    session.close()


def test_audio_rows_fully_deleted():
    session = _fresh_db()
    _seed_case(session, "audio")
    _run(session)
    for model, attr in _CASE_TABLES:
        assert _count(session, model, attr, "audio") == 0, model.__name__
    # audio seals gone for every stage.
    for stage_name in (
        EMBED_STAGE,
        "user_embeddings",
        "cluster_assign",
        CLUSTER_LABELS_STAGE,
        VIZ_STAGE,
    ):
        assert session.get(StageState, (stage_name, "audio")) is None, stage_name
    session.close()


def test_embed_and_cluster_labels_seals_adopted():
    session = _fresh_db()
    _seed_case(session, "maest")
    _run(session)

    assert session.get(StageState, (EMBED_STAGE, "maest")) is None
    embed = session.get(StageState, (EMBED_STAGE, "auditory"))
    assert embed is not None
    assert embed.config_hash == ADOPTED_EMBED_CONFIG
    # data/dependency unchanged (candidate set + deps unchanged).
    assert embed.data_hash == "embed-data"
    assert embed.dependency_hash == "embed-dep"

    cl = session.get(StageState, (CLUSTER_LABELS_STAGE, "auditory"))
    assert cl is not None
    assert cl.config_hash == ADOPTED_CLUSTER_LABELS_CONFIG
    assert cl.data_hash == "cl-data"
    session.close()


def test_viz_seal_stays_frozen_after_adoption():
    """Visualization must SKIP post-migration: its stored dependency_hash must
    equal what ``visualization._fingerprint`` would recompute from the adopted
    auditory upstream seals."""
    session = _fresh_db()
    _seed_case(session, "maest")
    _run(session)

    viz = session.get(StageState, (VIZ_STAGE, "auditory"))
    assert viz is not None
    expected_dep = fp.compose_hashes(
        fp.stage_dependency_hash(session, "cluster_assign", "auditory"),
        fp.stage_dependency_hash(session, CLUSTER_LABELS_STAGE, "auditory"),
    )
    assert viz.dependency_hash == expected_dep
    assert viz.config_hash == fp.hash_text("")
    session.close()


def test_idempotent_second_run_is_noop():
    session = _fresh_db()
    _seed_case(session, "maest")
    _seed_case(session, "audio")
    _run(session)
    snapshot = {
        (s.stage_name, s.scope_key): (s.data_hash, s.config_hash, s.dependency_hash)
        for s in session.query(StageState).all()
    }
    auditory_counts = {
        model.__name__: _count(session, model, attr, "auditory")
        for model, attr in _CASE_TABLES
    }

    _run(session)  # second run

    after = {
        (s.stage_name, s.scope_key): (s.data_hash, s.config_hash, s.dependency_hash)
        for s in session.query(StageState).all()
    }
    assert after == snapshot
    for model, attr in _CASE_TABLES:
        assert (
            _count(session, model, attr, "auditory") == auditory_counts[model.__name__]
        )
    session.close()


def test_clean_db_is_noop():
    """No maest/audio rows → migration touches nothing."""
    session = _fresh_db()
    _seed_case(session, "auditory")  # already migrated
    _run(session)
    for model, attr in _CASE_TABLES:
        assert _count(session, model, attr, "auditory") == 1, model.__name__
    session.close()


def test_matching_legacy_embed_hash_adopts_seal_no_recompute():
    """(a) Stored maest embed config == expected legacy hash → adopt the auditory
    seal so the next run SKIPs (no recompute)."""
    session = _fresh_db()
    _seed_case(session, "maest")
    _run(session)  # _seed_case seals embed config to LEGACY_EMBED_CONFIG

    embed = session.get(StageState, (EMBED_STAGE, "auditory"))
    assert embed is not None
    assert embed.config_hash == ADOPTED_EMBED_CONFIG
    session.close()


def test_stale_legacy_embed_hash_drops_seal_and_wipes_rows():
    """(b) Stored maest embed config != expected legacy hash (recipe drifted) →
    do NOT write an auditory seal AND wipe the re-keyed ClipEmbedding rows.

    The embed runner does not wipe on a missing seal (it incremental-diffs the
    config-independent per-clip source_hash), so leaving the re-keyed rows would
    let it seal stale vectors under the new auditory config. Wiping them forces a
    full re-embed. Other re-keyed rows (e.g. UserEmbedding) are left to self-heal
    via their own dependency chains."""
    session = _fresh_db()
    _seed_case(session, "maest")
    # Simulate a drifted MAEST recipe: the stored maest seal no longer matches
    # what the legacy recipe would now produce.
    _run(session, embed_legacy="stale-different-embed-config")

    # ClipEmbedding rows wiped so the next run re-embeds every clip from scratch.
    assert _count(session, ClipEmbedding, "embedding_case", "auditory") == 0
    assert _count(session, ClipEmbedding, "embedding_case", "maest") == 0
    # NO auditory embed seal exists → fingerprint drift recomputes.
    assert session.get(StageState, (EMBED_STAGE, "auditory")) is None
    assert session.get(StageState, (EMBED_STAGE, "maest")) is None
    session.close()


def test_embed_seal_adopted_preserves_rekeyed_rows():
    """Counterpart to (b): on the no-drift path the embed seal is adopted and the
    re-keyed ClipEmbedding rows are PRESERVED (no recompute)."""
    session = _fresh_db()
    _seed_case(session, "maest")
    _run(session)  # default embed_legacy matches the seeded maest config

    assert _count(session, ClipEmbedding, "embedding_case", "auditory") == 1
    embed = session.get(StageState, (EMBED_STAGE, "auditory"))
    assert embed is not None and embed.config_hash == ADOPTED_EMBED_CONFIG
    session.close()


def test_legacy_cluster_labels_payload_reconstructs_old_maest_payload():
    """The reconstructed legacy payload must equal what the OLD maest recipe
    hashed (prompt body under the ``maest`` key), so a drift-free rename adopts
    the cluster-labels seal. Hashing ``..., "maest")`` directly under current
    settings reads an empty prompt and would wrongly drop the seal."""
    body = "Summarize the acoustic character of this cluster in two sentences."

    # OLD settings: the cluster prompt body lived under the "maest" key.
    old = LabelsSettings(cluster_case_prompts={"maest": body})
    # CURRENT settings: the SAME body was moved verbatim to the "auditory" key.
    new = LabelsSettings(cluster_case_prompts={"auditory": body})

    auditory_payload = cluster_labels_config_payload(new, "auditory")
    reconstructed = legacy_cluster_labels_payload(auditory_payload)

    # Reconstruction matches the original maest payload → seal adopted.
    assert reconstructed == cluster_labels_config_payload(old, "maest")
    # The naive direct lookup reads an empty prompt and would NOT match.
    assert cluster_labels_config_payload(new, "maest") != cluster_labels_config_payload(
        old, "maest"
    )


def test_legacy_cluster_labels_payload_detects_prompt_drift():
    """If the prompt body actually changed during the move, the reconstructed
    legacy payload must NOT match the old seal so the stage recomputes."""
    old = LabelsSettings(cluster_case_prompts={"maest": "original body"})
    new = LabelsSettings(cluster_case_prompts={"auditory": "edited body"})

    reconstructed = legacy_cluster_labels_payload(
        cluster_labels_config_payload(new, "auditory")
    )
    assert reconstructed != cluster_labels_config_payload(old, "maest")


def test_stale_legacy_cluster_labels_hash_drops_seal_so_drift_recomputes():
    """(c) Stored maest cluster_labels config != expected legacy hash (prompt /
    generator knobs drifted) → do NOT write an auditory cluster_labels seal so
    stale prompt output is regenerated; viz dependency then drifts too."""
    session = _fresh_db()
    _seed_case(session, "maest")
    _run(session, cluster_labels_legacy="stale-different-cluster-labels-config")

    # Cluster-label rows re-keyed, but the seal is dropped (no recompute-skip).
    assert _count(session, ClusterLabel, "embedding_case", "auditory") == 1
    assert session.get(StageState, (CLUSTER_LABELS_STAGE, "auditory")) is None
    assert session.get(StageState, (CLUSTER_LABELS_STAGE, "maest")) is None

    # Visualization's re-derived dependency hash must reflect the now-absent
    # cluster_labels seal so visualization recomputes alongside it.
    viz = session.get(StageState, (VIZ_STAGE, "auditory"))
    assert viz is not None
    expected_dep = fp.compose_hashes(
        fp.stage_dependency_hash(session, "cluster_assign", "auditory"),
        fp.stage_dependency_hash(session, CLUSTER_LABELS_STAGE, "auditory"),
    )
    assert viz.dependency_hash == expected_dep
    session.close()
