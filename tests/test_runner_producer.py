"""The producer builds one job per (case, target clip) with the video_key,
audio_key, and fps/max_frames probed from the local file."""

from __future__ import annotations

from types import SimpleNamespace

from modules.embeddings.runner import build_jobs_for_case


def test_build_jobs_sets_video_key_and_sampling(monkeypatch):
    monkeypatch.setattr(
        "modules.embeddings.runner.adaptive_sampling", lambda *a: (2.0, 96)
    )
    monkeypatch.setattr("modules.embeddings.runner.os.path.exists", lambda p: True)
    clips = [
        SimpleNamespace(id=1),
        SimpleNamespace(id=2),
    ]
    from modules.embeddings.cases import CASE_REGISTRY

    jobs = build_jobs_for_case(
        CASE_REGISTRY["video"],
        clips,
        texts={1: None, 2: None},
        video_dir="/videos",
        adaptive_max_frames=96,
        adaptive_default_fps=2.0,
    )
    by_id = {j["clip_id"]: j for j in jobs}
    assert by_id[1]["video_key"] == "1.mp4"
    assert by_id[2]["video_key"] == "2.mp4"
    assert by_id[1]["fps"] == 2.0 and by_id[1]["max_frames"] == 96


def test_build_jobs_sets_audio_key_for_audio_dependent_case(monkeypatch):
    monkeypatch.setattr("modules.embeddings.runner.os.path.exists", lambda p: True)
    clips = [SimpleNamespace(id=7)]
    from modules.embeddings.cases import CASE_REGISTRY

    jobs = build_jobs_for_case(
        CASE_REGISTRY[
            "auditory"
        ],  # requires_video=False, dependency has _audio_file_stat
        clips,
        texts={7: None},
        video_dir="/videos",
        adaptive_max_frames=96,
        adaptive_default_fps=2.0,
    )
    job = jobs[0]
    assert job["audio_key"] == "7.mp3"
    assert job["video_key"] is None
    assert job["fps"] is None and job["max_frames"] is None
