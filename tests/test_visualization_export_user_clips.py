"""Tests for the per-user ``clips`` JSON block in ``export_visualization_json``.

Covers two behaviours of ``_render_user_clips_block``:

* Per-case fan-out: ``ClipLabel`` rows are joined by ``label_case`` so a
  clip with N modality-specific rows does not appear N times.
* Cross-case fallback: the visual ``ClipLabel`` payload is
  case-agnostic content (it describes the clip's frames), so for
  stage-1-skipped cases (audio/sandwich/maest/gemini) the per-clip
  block falls back to ``label_case="video"`` rows rather than
  returning empty — the creator pane shows the same per-clip tags in
  every run.
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


def test_user_clips_block_falls_back_to_video_for_stage1_skipped_case(tmp_path):
    """Audio is stage-1-skipped — the cluster pass synthesises directly
    from raw signals and the pipeline writes no per-clip audio
    ``ClipLabel`` rows. But the per-clip visual tags describe the
    clip's frames regardless of which embedding case is being viewed,
    so the audio export must surface the video ``ClipLabel`` payload
    in the per-clip block. Any stray audio ``ClipLabel`` row is
    ignored — only the video reading is shown.
    """
    _clear()
    _seed_one_user_one_clip("audio", clip_id=100)
    _add_clip_label(
        clip_id=100, label_case="video", payload=_video_payload("the video one")
    )
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
    assert clips[0]["sentence"] == "the video one"
    assert clips[0]["tags"]["observable"] == [{"tag": "lamp", "evidence": "frame 1"}]


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


def test_user_clips_block_omits_clip_with_no_video_label_for_stage1_skipped_case(
    tmp_path,
):
    """For a stage-1-skipped case, the per-clip block falls back to
    the video ``ClipLabel``. A clip with no video label at all must be
    omitted (inner join on ``label_case``, no outer fallback to other
    cases)."""
    _clear()
    _seed_one_user_one_clip("audio", clip_id=100)
    # Only an audio ClipLabel exists; there is no video label to fall
    # back to, so the per-case block must be empty.
    _add_clip_label(clip_id=100, label_case="audio", payload=_audio_payload())

    export_visualization_json(
        _settings(tmp_path, default_case="audio"), cases=("audio",)
    )

    detail = json.loads((tmp_path / "runs" / "audio" / "users" / "0.json").read_text())
    assert detail["clips"] == []
