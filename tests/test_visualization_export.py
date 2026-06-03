"""Tests for modules.visualization.export.export_visualization_json.

Covers both the bulk JSON pass (manifest, users.json, clusters.json) and the
per-entity detail pass (clusters/N.json, users/N.json) that depends on
AudioMIR / UserStats / Clip rows.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from core.contract import SCHEMA_VERSION
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


def _seed_visualization(
    case: str, label: str, *, n_users: int = 6, n_clusters: int = 2
) -> None:
    """Seed Visualization + Visualization{User,Cluster} rows without MIR data."""
    session = get_session()
    try:
        session.merge(
            Visualization(
                embedding_case=case,
                label=label,
                size=n_users,
                source_hash="hash-" + case,
            )
        )
        for uid in range(n_users):
            session.merge(User(id=uid))
            cid = uid % n_clusters
            session.add(
                VisualizationUser(
                    user_id=uid,
                    embedding_case=case,
                    x=float(uid),
                    y=float(-uid),
                    cluster_id=cid,
                )
            )
        for cid in range(n_clusters):
            session.add(
                VisualizationCluster(
                    embedding_case=case,
                    cluster_id=cid,
                    cx=float(cid),
                    cy=float(cid * 2),
                    rx=1.0,
                    ry=2.0,
                    angle=0.0,
                    size=n_users // n_clusters,
                    label=f"Cluster {cid + 1}",
                )
            )
        session.commit()
    finally:
        session.close()


def _seed_case_with_mir(case: str = "video", *, n_users: int = 4) -> None:
    """Like _seed_visualization but also lays down UserStats + Clip + AudioMIR
    rows so the per-entity detail pass has something to materialize."""
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


def _settings(tmp_path: Path, default_case: str = "video") -> SimpleNamespace:
    return SimpleNamespace(
        visualization=SimpleNamespace(
            export_dir=tmp_path,
            default_case=default_case,
            distinctiveness_z_min=0.0,
            distinctiveness_top_k=3,
            genre_top_k=5,
            instrument_top_k=3,
            languages_top_k=3,
            edge_percentile=66,
        )
    )


def test_export_writes_manifest_users_clusters_for_exposed_case(tmp_path):
    _clear()
    _seed_visualization("video", "Visual", n_users=6, n_clusters=2)
    export_visualization_json(_settings(tmp_path), cases=("video",))

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["version"] == SCHEMA_VERSION
    assert manifest["default_run_id"] == "video"
    assert manifest["runs"] == [
        {
            "id": "video",
            "case": "video",
            "label": "Visual",
            "size": 6,
            "details_available": True,
        }
    ]

    users = json.loads((tmp_path / "runs" / "video" / "users.json").read_text())
    assert users["version"] == SCHEMA_VERSION
    assert users["run_id"] == "video"
    assert users["bounds"] == {"minX": 0.0, "maxX": 5.0, "minY": -5.0, "maxY": 0.0}
    assert len(users["users"]) == 6
    # Tuple shape: [id, x, y, cluster_id, has_detail, centrality]
    assert all(len(u) == 6 for u in users["users"])
    # No MIR data seeded → has_detail must be False for every row.
    assert all(u[4] is False for u in users["users"])
    assert all(0.0 <= float(u[5]) <= 1.0 for u in users["users"])

    clusters = json.loads((tmp_path / "runs" / "video" / "clusters.json").read_text())
    assert clusters["version"] == SCHEMA_VERSION
    assert clusters["run_id"] == "video"
    assert len(clusters["clusters"]) == 2
    assert {c["label"] for c in clusters["clusters"]} == {"Cluster 1", "Cluster 2"}
    assert all("has_detail" in c for c in clusters["clusters"])


def test_export_filters_hidden_cases(tmp_path):
    _clear()
    _seed_visualization("video", "Visual")
    _seed_visualization("hidden", "Hidden")
    export_visualization_json(_settings(tmp_path), cases=("video", "hidden"))

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert [r["case"] for r in manifest["runs"]] == ["video"]
    assert not (tmp_path / "runs" / "hidden").exists()


def test_export_skips_cases_with_no_db_row(tmp_path):
    _clear()
    _seed_visualization("video", "Visual")
    # "sandwich" is in cases but has no Visualization row.
    export_visualization_json(_settings(tmp_path), cases=("video", "sandwich"))

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert [r["case"] for r in manifest["runs"]] == ["video"]
    assert not (tmp_path / "runs" / "sandwich").exists()


def test_export_no_visible_runs_skips_manifest(tmp_path):
    _clear()
    _seed_visualization("hidden", "Hidden")  # hidden-only
    export_visualization_json(_settings(tmp_path), cases=("hidden",))

    assert not (tmp_path / "manifest.json").exists()


def test_export_no_runs_removes_stale_manifest_and_run_dirs(tmp_path):
    _clear()
    stale_manifest = tmp_path / "manifest.json"
    stale_manifest.write_text(json.dumps({"version": "stale", "runs": [{"id": "old"}]}))
    stale_case = tmp_path / "runs" / "old"
    stale_case.mkdir(parents=True)
    (stale_case / "users.json").write_text("{}")

    export_visualization_json(_settings(tmp_path), cases=("video",))

    assert not stale_manifest.exists()
    assert not stale_case.exists()


def test_export_prunes_run_dirs_for_dropped_cases(tmp_path):
    _clear()
    _seed_visualization("video", "Visual")
    _seed_visualization("sandwich", "Sandwich")
    export_visualization_json(_settings(tmp_path), cases=("video", "sandwich"))
    assert (tmp_path / "runs" / "sandwich").exists()

    _clear()
    _seed_visualization("video", "Visual")
    export_visualization_json(_settings(tmp_path), cases=("video", "sandwich"))

    assert (tmp_path / "runs" / "video").exists()
    assert not (tmp_path / "runs" / "sandwich").exists()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert [r["case"] for r in manifest["runs"]] == ["video"]


def test_export_empty_case_writes_degenerate_bounds(tmp_path):
    _clear()
    session = get_session()
    try:
        session.merge(
            Visualization(
                embedding_case="video",
                label="Visual",
                size=0,
                source_hash="empty",
            )
        )
        session.commit()
    finally:
        session.close()

    export_visualization_json(_settings(tmp_path), cases=("video",))
    users = json.loads((tmp_path / "runs" / "video" / "users.json").read_text())
    assert users["users"] == []
    # Default safe box (non-zero extent) so the frontend's stretchRun
    # never divides by zero.
    assert users["bounds"] == {"minX": -1.0, "maxX": 1.0, "minY": -1.0, "maxY": 1.0}


def test_genre_only_pairs_reduces_flattened_labels_to_leaf():
    # MIR flattens ``Parent---Child`` to ``Parent Child``; the viewer shows
    # only the leaf, including for multi-word parents.
    assert _genre_only_pairs("Electronic House, Hip Hop Trap", "0.8, 0.2") == [
        ("House", 0.8),
        ("Trap", 0.2),
    ]


def test_genre_only_pairs_passes_unknown_labels_through():
    assert _genre_only_pairs("totally unknown", "0.5") == [("totally unknown", 0.5)]


def test_per_entity_files_written_with_correct_ids(tmp_path):
    _clear()
    _seed_case_with_mir("video", n_users=4)
    export_visualization_json(_settings(tmp_path), cases=("video",))

    user_dir = tmp_path / "runs" / "video" / "users"
    assert {p.stem for p in user_dir.glob("*.json")} == {"0", "1", "2", "3"}

    # Cluster main detail ships as one per-run bundle (no per-cluster <id>.json).
    bundle = json.loads(
        (tmp_path / "runs" / "video" / "clusters-detail.json").read_text()
    )
    assert bundle["version"] == SCHEMA_VERSION
    assert bundle["run_id"] == "video"
    assert {c["cluster_id"] for c in bundle["clusters"]} == {0, 1}
    sample = next(c for c in bundle["clusters"] if c["cluster_id"] == 0)
    assert sample["follower_bucket"].endswith("k")
    assert "version" not in sample
    assert "label" not in sample
    assert "label_modality" in sample


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
    bundle = json.loads(
        (tmp_path / "runs" / "video" / "clusters-detail.json").read_text()
    )
    cluster_detail = next(c for c in bundle["clusters"] if c["cluster_id"] == 0)
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
