"""Tests for the per-user ``clips`` JSON block in ``export_visualization_json``.

Regression coverage for the per-case ``ClipLabel`` fan-out fix: the
``_render_user_clips_block`` helper must filter ``ClipLabel`` by
``label_case == case`` (otherwise a clip with N modality-specific label
rows would appear N times in the export) and must read the
case-appropriate ``observable_*_tags`` / ``one_sentence_*_reading``
payload keys (the case-agnostic ``aesthetic_tags`` /
``community_signalling_tags`` keys are stable per SPEC).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from core.database import (
    AudioMIR,
    Base,
    Clip,
    ClipLabel,
    ClusterLabel,
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
from modules.visualization.export import export_visualization_json


def _settings(tmp_path: Path, default_case: str = "video") -> SimpleNamespace:
    return SimpleNamespace(
        visualization=SimpleNamespace(
            export_dir=str(tmp_path),
            default_case=default_case,
            distinctiveness_z_min=0.0,
            distinctiveness_top_k=3,
            genre_top_k=5,
            instrument_top_k=3,
            languages_top_k=3,
            edge_percentile=66,
        )
    )


def _clear() -> None:
    Base.metadata.create_all(get_engine())
    session = get_session()
    try:
        for m in (
            ClusterLabel,
            ClipLabel,
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


def _seed_one_user_one_clip(case: str, *, clip_id: int = 100) -> None:
    """Seed the minimum rows needed for ``export_visualization_json`` to
    materialise a single user's detail file under ``runs/{case}/users/``.

    Visualization rows exist for both ``video`` and ``audio`` so a single
    seed can drive a per-case export under either case without re-seeding.
    """
    session = get_session()
    try:
        for c in ("video", "audio"):
            session.merge(
                Visualization(
                    embedding_case=c,
                    label=c.capitalize(),
                    size=1,
                    source_hash="hash-" + c,
                )
            )
            session.add(
                UserCluster(
                    user_id=0,
                    embedding_case=c,
                    cluster_id=0,
                    umap_x=0.0,
                    umap_y=0.0,
                )
            )
            session.add(
                VisualizationUser(
                    user_id=0,
                    embedding_case=c,
                    x=0.0,
                    y=0.0,
                    cluster_id=0,
                )
            )
            session.add(
                VisualizationCluster(
                    embedding_case=c,
                    cluster_id=0,
                    cx=0.0,
                    cy=0.0,
                    rx=1.0,
                    ry=1.0,
                    angle=0.0,
                    size=1,
                    label="Cluster 1",
                )
            )
        session.merge(User(id=0, follower_count=10_000))
        session.merge(
            UserStats(
                user_id=0,
                n_clips=1,
                median_plays=10_000.0,
                approx_clips_per_week=1.0,
                top_to_median_plays_ratio=1.0,
                median_video_duration=15.0,
                clip_time_span_days=30.0,
            )
        )
        session.add(
            Clip(
                id=clip_id,
                user_id=0,
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
                approachability=0.5,
                engagement=0.5,
                danceability=0.5,
                genre_labels="house",
                genre_scores="1.0",
                instrument_labels="synth",
                instrument_scores="1.0",
            )
        )
        session.commit()
    finally:
        session.close()


def _add_clip_label(
    *,
    clip_id: int,
    label_case: str,
    payload: dict,
) -> None:
    session = get_session()
    try:
        session.add(
            ClipLabel(
                clip_id=clip_id,
                label_case=label_case,
                status="success",
                validation="ok",
                payload=payload,
                warnings=[],
                attempts=1,
            )
        )
        session.commit()
    finally:
        session.close()


def _video_payload(sentence: str = "video reading") -> dict:
    return {
        "observable_visual_tags": [{"tag": "lamp", "evidence": "frame 1"}],
        "aesthetic_tags": [
            {"tag": "moody", "grounded_in": ["lamp"], "confidence": "medium"}
        ],
        "community_signalling_tags": [
            {"tag": "indie", "grounded_in": ["moody"], "confidence": "low"}
        ],
        "one_sentence_visual_reading": sentence,
    }


def _audio_payload(sentence: str = "audio reading") -> dict:
    return {
        "observable_audio_tags": [{"tag": "kick", "evidence": "0:01"}],
        "aesthetic_tags": [
            {"tag": "warm", "grounded_in": ["kick"], "confidence": "medium"}
        ],
        "community_signalling_tags": [
            {"tag": "house", "grounded_in": ["warm"], "confidence": "low"}
        ],
        "one_sentence_audio_reading": sentence,
    }


def test_user_clips_block_for_audio_case_filters_by_label_case(tmp_path):
    """Seeding both a video and an audio ClipLabel row for the same clip
    must yield exactly one clip entry in the audio export, sourced from
    the audio payload (no 2× fan-out, no visual-key leakage)."""
    _clear()
    _seed_one_user_one_clip("audio", clip_id=100)
    _add_clip_label(clip_id=100, label_case="video", payload=_video_payload())
    _add_clip_label(
        clip_id=100,
        label_case="audio",
        payload=_audio_payload("the audio one"),
    )

    export_visualization_json(
        _settings(tmp_path, default_case="audio"), cases=("audio",)
    )

    detail = json.loads((tmp_path / "runs" / "audio" / "users" / "0.json").read_text())
    clips = detail["clips"]
    assert len(clips) == 1
    assert clips[0]["clip_id"] == 100
    assert clips[0]["sentence"] == "the audio one"
    assert clips[0]["tags"]["observable"] == [{"tag": "kick", "evidence": "0:01"}]


def test_user_clips_block_for_video_case_remains_unchanged(tmp_path):
    """Same seed, but exporting the video case must yield the video row's
    one_sentence_visual_reading — not the audio row's."""
    _clear()
    _seed_one_user_one_clip("video", clip_id=100)
    _add_clip_label(
        clip_id=100, label_case="video", payload=_video_payload("the video one")
    )
    _add_clip_label(clip_id=100, label_case="audio", payload=_audio_payload())

    export_visualization_json(
        _settings(tmp_path, default_case="video"), cases=("video",)
    )

    detail = json.loads((tmp_path / "runs" / "video" / "users" / "0.json").read_text())
    clips = detail["clips"]
    assert len(clips) == 1
    assert clips[0]["clip_id"] == 100
    assert clips[0]["sentence"] == "the video one"
    assert clips[0]["tags"]["observable"] == [{"tag": "lamp", "evidence": "frame 1"}]


def test_user_clips_block_skips_clip_with_no_label_for_this_case(tmp_path):
    """A clip with only a video ClipLabel must be OMITTED from the audio
    export — the join is inner on label_case, not outer."""
    _clear()
    _seed_one_user_one_clip("audio", clip_id=100)
    _add_clip_label(clip_id=100, label_case="video", payload=_video_payload())

    export_visualization_json(
        _settings(tmp_path, default_case="audio"), cases=("audio",)
    )

    # The user detail file is still written (MIR-backed clip is present in
    # ``self_member.clips``), but the per-case ``clips`` block must be
    # EMPTY: the video-only label row must not bleed into the audio export
    # via an outer join, confirming inner-join semantics on label_case.
    detail = json.loads((tmp_path / "runs" / "audio" / "users" / "0.json").read_text())
    assert detail["clips"] == []
