"""Integration tests for the per-entity detail-file export pass."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from core.database import (
    AudioMIR,
    Base,
    Clip,
    StageState,
    User,
    UserCluster,
    UserStats,
    Visualization,
    VisualizationCluster,
    VisualizationUser,
    get_engine,
    get_session,
)
from modules.visualization.export import (
    _genre_only_pairs,
    export_visualization_json,
)
from modules.visualization.schema import SCHEMA_VERSION


def test_genre_only_pairs_reduces_flattened_labels_to_leaf():
    # MIR flattens ``Parent---Child`` to ``Parent Child``; the viewer shows
    # only the leaf, including for multi-word parents.
    assert _genre_only_pairs("Electronic House, Hip Hop Trap", "0.8, 0.2") == [
        ("House", 0.8),
        ("Trap", 0.2),
    ]


def test_genre_only_pairs_passes_unknown_labels_through():
    assert _genre_only_pairs("totally unknown", "0.5") == [("totally unknown", 0.5)]


def _clear() -> None:
    Base.metadata.create_all(get_engine())
    session = get_session()
    try:
        for m in (
            AudioMIR,
            Clip,
            UserStats,
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


def _seed_case_with_mir(case: str = "video", *, n_users: int = 4) -> None:
    session = get_session()
    try:
        session.merge(
            Visualization(
                embedding_case=case,
                label="Visual",
                size=n_users,
                source_hash="h",
            )
        )
        for uid in range(n_users):
            session.merge(User(id=uid, follower_count=12_000 + uid * 1000))
            session.merge(
                UserStats(
                    user_id=uid,
                    n_clips=3,
                    median_plays=10_000.0,
                    approx_clips_per_week=2.0,
                    top_to_median_plays_ratio=1.5,
                    median_video_duration=20.0,
                    clip_time_span_days=180.0,
                )
            )
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
            for cidx in range(3):
                clip_id = uid * 10 + cidx
                session.add(
                    Clip(
                        id=clip_id,
                        user_id=uid,
                        is_selected=True,
                        is_speech_detected=False,
                        speech_language=None,
                        caption_language="en",
                    )
                )
                session.add(
                    AudioMIR(
                        clip_id=clip_id,
                        is_mir_extracted=True,
                        is_music_detected=True,
                        approachability=0.6,
                        engagement=0.7,
                        danceability=0.5,
                        is_happy=True,
                        is_electronic=True,
                        is_tonal=True,
                        genre_labels="house, techno",
                        genre_scores="0.8, 0.2",
                        instrument_labels="synth",
                        instrument_scores="0.9",
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


def _settings(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        visualization=SimpleNamespace(
            export_dir=tmp_path,
            default_case="video",
            distinctiveness_z_min=0.0,
            distinctiveness_top_k=3,
            genre_top_k=5,
            instrument_top_k=3,
            languages_top_k=3,
            edge_percentile=66,
        )
    )


def test_per_entity_files_written_with_correct_ids(tmp_path):
    _clear()
    _seed_case_with_mir("video", n_users=4)
    export_visualization_json(_settings(tmp_path), cases=("video",))

    cluster_dir = tmp_path / "runs" / "video" / "clusters"
    user_dir = tmp_path / "runs" / "video" / "users"
    assert {p.stem for p in cluster_dir.glob("*.json")} == {"0", "1"}
    assert {p.stem for p in user_dir.glob("*.json")} == {"0", "1", "2", "3"}

    sample = json.loads((cluster_dir / "0.json").read_text())
    assert sample["version"] == SCHEMA_VERSION
    assert sample["cluster_id"] == 0
    assert sample["follower_bucket"].endswith("k")


def test_bulk_rows_carry_has_detail_flags(tmp_path):
    _clear()
    _seed_case_with_mir("video", n_users=2)
    export_visualization_json(_settings(tmp_path), cases=("video",))
    users = json.loads((tmp_path / "runs" / "video" / "users.json").read_text())
    assert all(u[4] is True for u in users["users"])

    clusters = json.loads((tmp_path / "runs" / "video" / "clusters.json").read_text())
    assert all(c["has_detail"] is True for c in clusters["clusters"])

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["runs"][0]["details_available"] is True


def test_creator_without_mir_data_has_no_detail_file(tmp_path):
    _clear()
    _seed_case_with_mir("video", n_users=2)
    # Add a third user with no Clip/AudioMIR rows.
    session = get_session()
    try:
        session.merge(User(id=99, follower_count=1000))
        session.add(
            UserCluster(
                user_id=99,
                embedding_case="video",
                cluster_id=0,
                umap_x=5.0,
                umap_y=5.0,
            )
        )
        session.add(
            VisualizationUser(
                user_id=99,
                embedding_case="video",
                x=5.0,
                y=5.0,
                cluster_id=0,
            )
        )
        session.commit()
    finally:
        session.close()

    export_visualization_json(_settings(tmp_path), cases=("video",))
    user_dir = tmp_path / "runs" / "video" / "users"
    assert not (user_dir / "99.json").exists()

    users = json.loads((tmp_path / "runs" / "video" / "users.json").read_text())
    row_99 = next(u for u in users["users"] if u[0] == 99)
    assert row_99[4] is False


def test_failed_mir_clips_excluded_from_user_detail(tmp_path):
    """A user whose selected clips all have failed AudioMIR rows must not get a detail file."""
    _clear()
    _seed_case_with_mir("video", n_users=2)
    session = get_session()
    try:
        session.merge(User(id=99, follower_count=1000))
        session.merge(
            UserStats(
                user_id=99,
                n_clips=2,
                median_plays=5_000.0,
                approx_clips_per_week=1.0,
                top_to_median_plays_ratio=1.2,
                median_video_duration=15.0,
                clip_time_span_days=90.0,
            )
        )
        session.add(
            UserCluster(
                user_id=99,
                embedding_case="video",
                cluster_id=0,
                umap_x=5.0,
                umap_y=5.0,
            )
        )
        session.add(
            VisualizationUser(
                user_id=99,
                embedding_case="video",
                x=5.0,
                y=5.0,
                cluster_id=0,
            )
        )
        for cidx in range(2):
            clip_id = 9900 + cidx
            session.add(
                Clip(
                    id=clip_id,
                    user_id=99,
                    is_selected=True,
                    is_speech_detected=False,
                    speech_language=None,
                    caption_language="en",
                )
            )
            session.add(
                AudioMIR(
                    clip_id=clip_id,
                    is_mir_extracted=False,
                    mir_error="no_audio_file",
                )
            )
        session.commit()
    finally:
        session.close()

    export_visualization_json(_settings(tmp_path), cases=("video",))
    user_dir = tmp_path / "runs" / "video" / "users"
    assert not (user_dir / "99.json").exists()

    users = json.loads((tmp_path / "runs" / "video" / "users.json").read_text())
    row_99 = next(u for u in users["users"] if u[0] == 99)
    assert row_99[4] is False


def test_non_mir_user_contributes_to_cluster_aggregates(tmp_path):
    """Users without MIR clips still feed cluster-level follower / posting aggregates."""
    _clear()
    _seed_case_with_mir("video", n_users=2)
    session = get_session()
    try:
        session.merge(User(id=77, follower_count=500_000))
        session.merge(
            UserStats(
                user_id=77,
                n_clips=4,
                median_plays=80_000.0,
                approx_clips_per_week=3.0,
                top_to_median_plays_ratio=2.0,
                median_video_duration=25.0,
                clip_time_span_days=120.0,
            )
        )
        session.add(
            UserCluster(
                user_id=77,
                embedding_case="video",
                cluster_id=0,
                umap_x=0.5,
                umap_y=0.5,
            )
        )
        session.add(
            VisualizationUser(
                user_id=77,
                embedding_case="video",
                x=0.5,
                y=0.5,
                cluster_id=0,
            )
        )
        session.commit()
    finally:
        session.close()

    export_visualization_json(_settings(tmp_path), cases=("video",))
    cluster_detail = json.loads(
        (tmp_path / "runs" / "video" / "clusters" / "0.json").read_text()
    )
    # User 77 has 500k followers which falls in a non-zero follower bucket;
    # earlier behavior dropped this user from members entirely.
    assert cluster_detail["follower_bucket"] != "<1k"
    assert cluster_detail["posting"]["median_plays"] > 0


def test_user_detail_reports_selected_clip_count_not_mir_subset(tmp_path):
    """When some selected clips lack AudioMIR rows, exported n_clips reflects all selected."""
    _clear()
    _seed_case_with_mir("video", n_users=2)
    session = get_session()
    try:
        # Add one more selected clip to user 0 without an AudioMIR row.
        session.add(
            Clip(
                id=999,
                user_id=0,
                is_selected=True,
                is_speech_detected=False,
                speech_language=None,
                caption_language="en",
            )
        )
        session.commit()
    finally:
        session.close()

    export_visualization_json(_settings(tmp_path), cases=("video",))
    user_detail = json.loads(
        (tmp_path / "runs" / "video" / "users" / "0.json").read_text()
    )
    # User 0 has 3 MIR-backed clips + 1 selected clip with no AudioMIR = 4 selected total.
    assert user_detail["n_clips"] == 4


def test_reexport_prunes_stale_entity_files(tmp_path):
    _clear()
    _seed_case_with_mir("video", n_users=4)
    export_visualization_json(_settings(tmp_path), cases=("video",))
    user_dir = tmp_path / "runs" / "video" / "users"
    assert (user_dir / "3.json").exists()

    # Remove user 3 from the cohort entirely and re-export.
    session = get_session()
    try:
        session.query(VisualizationUser).filter_by(user_id=3).delete()
        session.query(UserCluster).filter_by(user_id=3).delete()
        session.query(AudioMIR).filter(AudioMIR.clip_id.in_([30, 31, 32])).delete(
            synchronize_session=False
        )
        session.query(Clip).filter_by(user_id=3).delete()
        session.query(UserStats).filter_by(user_id=3).delete()
        session.query(User).filter_by(id=3).delete()
        viz = session.get(Visualization, "video")
        if viz is not None:
            viz.size = 3
        session.commit()
    finally:
        session.close()

    export_visualization_json(_settings(tmp_path), cases=("video",))
    assert not (user_dir / "3.json").exists()
    assert (user_dir / "0.json").exists()
