"""Per-case stage-1 fingerprint isolation.

Drifting one case's stage-1 prompt MUST wipe only that case's
``ClipLabel`` rows — every other case's rows are preserved.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import SimpleNamespace

from sqlalchemy import create_engine
from sqlalchemy import event as _sqla_event
from sqlalchemy.orm import Session

from core.config import LabelsSettings, Secrets
from core.database import (
    AudioMIR,
    Base,
    Clip,
    ClipLabel,
    ClusterLabel,
    User,
    UserCluster,
)
from core.database.engine import _apply_sqlite_pragmas
from modules.embeddings.cases import default_cases
from modules.labels.cases import REGISTRY
from modules.labels.clip_pass import run_case
from tests._clustering_helpers import seed_audio_mir
from tests.test_labels_pipeline import _DispatchingGen, _patch_dispatching


def _engine():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


def _settings(tmp_path) -> SimpleNamespace:
    return SimpleNamespace(
        paths=SimpleNamespace(
            video_dir=str(tmp_path),
            video_for=lambda cid: tmp_path / f"{cid}.mp4",
        )
    )


def _labels(prompts: dict[str, str]) -> LabelsSettings:
    return LabelsSettings(
        case_prompts=prompts,
        cluster_case_prompts={k: "x" for k in prompts},
    )


@dataclass
class _FakeGen:
    response: str = ""
    video_calls: list = field(default_factory=list)
    text_calls: list = field(default_factory=list)

    def run(self, video_path, prompt: str) -> str:
        self.video_calls.append((str(video_path), prompt))
        return self.response

    def run_many(self, video_paths, prompt: str) -> list[str]:
        return [self.run(vp, prompt) for vp in video_paths]

    def run_text(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        seed: int | None = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        schema: dict | None = None,
    ) -> str:
        self.text_calls.append((prompt, max_new_tokens))
        return self.response


def _video_json() -> str:
    return json.dumps(
        {
            "observable_visual_tags": [
                {"tag": "warm kitchen scene", "evidence": "lamp"},
                {"tag": "shallow depth field", "evidence": "blur"},
                {"tag": "handheld framing", "evidence": "drift"},
            ],
            "aesthetic_tags": [
                {
                    "tag": "soft domestic vignette",
                    "grounded_in": ["warm kitchen scene"],
                    "confidence": "medium",
                },
                {
                    "tag": "warm-toned framing",
                    "grounded_in": ["warm kitchen scene"],
                    "confidence": "low",
                },
                {
                    "tag": "intimate handheld register",
                    "grounded_in": ["handheld framing"],
                    "confidence": "high",
                },
            ],
            "community_signalling_tags": [
                {
                    "tag": "slow-living domestic taste",
                    "grounded_in": ["soft domestic vignette"],
                    "confidence": "low",
                },
                {
                    "tag": "homecore aesthetic register",
                    "grounded_in": ["warm-toned framing"],
                    "confidence": "medium",
                },
                {
                    "tag": "personal-diary reels register",
                    "grounded_in": ["intimate handheld register"],
                    "confidence": "low",
                },
            ],
            "one_sentence_visual_reading": (
                "tight handheld kitchen vignette with warm domestic palette"
            ),
        }
    )


def _seed_full_clip(eng) -> None:
    with Session(eng) as s:
        s.add(User(id=1, is_selected=True))
        s.add(
            Clip(
                id=1,
                user_id=1,
                is_selected=True,
                is_downloaded=True,
                caption_clean="warm kitchen scene",
                caption_language="en",
                is_speech_detected=True,
                speech_transcription="a calm slow narration",
                speech_language="en",
            )
        )
        s.add(
            AudioMIR(
                clip_id=1,
                is_music_detected=True,
                genre_labels="lofi, downtempo",
                moodtheme_labels="relaxed",
                instrument_labels="piano",
            )
        )
        s.commit()


def _all_prompts(
    *, video="VP", spoken="AP", sandwich="SP", auditory="MP", textual="TP"
):
    return {
        "video": video,
        "spoken": spoken,
        "sandwich": sandwich,
        "auditory": auditory,
        "textual": textual,
    }


def test_video_prompt_drift_wipes_only_video_rows(tmp_path):
    eng = _engine()
    _seed_full_clip(eng)
    # Pre-seed spoken + sandwich rows directly so we can verify they survive.
    with Session(eng) as s:
        s.add(
            ClipLabel(
                clip_id=1,
                label_case="spoken",
                status="success",
                validation="ok",
                payload={"keep": "spoken"},
                warnings=[],
                attempts=1,
            )
        )
        s.add(
            ClipLabel(
                clip_id=1,
                label_case="sandwich",
                status="success",
                validation="ok",
                payload={"keep": "sandwich"},
                warnings=[],
                attempts=1,
            )
        )
        s.commit()
    gen = _FakeGen(response=_video_json())
    settings = _settings(tmp_path)
    with Session(eng) as s:
        run_case(
            session=s,
            settings=settings,
            labels=_labels(_all_prompts()),
            generator=gen,
            spec=REGISTRY["video"],
        )
    with Session(eng) as s:
        run_case(
            session=s,
            settings=settings,
            labels=_labels(_all_prompts(video="DIFFERENT")),
            generator=gen,
            spec=REGISTRY["video"],
        )
    with Session(eng) as s:
        assert s.get(ClipLabel, (1, "spoken")).payload == {"keep": "spoken"}
        assert s.get(ClipLabel, (1, "sandwich")).payload == {"keep": "sandwich"}
        video_row = s.get(ClipLabel, (1, "video"))
        assert video_row is not None
        assert video_row.status == "success"


# ---------------------------------------------------------------------------
# Phase F2 — per-case CLUSTER drift isolation (end-to-end via pipeline.run)
# ---------------------------------------------------------------------------


def _shared_engine(monkeypatch):
    """Build the shared in-memory engine that ``get_engine()`` returns."""
    eng = create_engine("sqlite:///:memory:")
    _sqla_event.listen(eng, "connect", _apply_sqlite_pragmas)
    Base.metadata.create_all(eng)
    import core.database.engine as engine_mod

    monkeypatch.setattr(engine_mod, "_main_engine", eng)
    return eng


def _secrets() -> Secrets:
    return Secrets(
        database_url="sqlite:///:memory:",
        identity_db_url="sqlite:///:memory:",
        hiker_api_key="",
        huggingface_token="",
    )


def _full_pipeline_settings(tmp_path):
    from core.config import _load_settings

    s = _load_settings()
    s.labels.max_attempts = 1
    s.paths.video_dir = str(tmp_path)
    return s


def _seed_full_clip_for_pipeline(eng, *, tmp_path) -> None:
    """Seed user + clip + speech + MIR + a video file + per-case clusters.

    Sandwich needs the video stage-1 row to exist before the sandwich
    runner gates: we DON'T pre-seed it — the pipeline runs the video
    case first, producing the row in time.
    """
    with Session(eng) as s:
        s.add(User(id=1, is_selected=True))
        s.add(
            Clip(
                id=1,
                user_id=1,
                is_selected=True,
                is_downloaded=True,
                caption_clean="warm kitchen scene",
                caption_language="en",
                is_speech_detected=True,
                speech_transcription="a calm slow narration",
                speech_language="en",
            )
        )
        s.commit()
        seed_audio_mir(s, clip_id=1)
    (tmp_path / "1.mp4").write_bytes(b"\x00")


def _seed_clusters_for_cases(eng, cases: tuple[str, ...]) -> None:
    with Session(eng) as s:
        for case in cases:
            s.add(
                UserCluster(
                    user_id=1,
                    embedding_case=case,
                    cluster_id=0,
                    umap_x=0.0,
                    umap_y=0.0,
                    centrality=0.9,
                )
            )
        s.commit()


def _cluster_payloads_by_case(eng) -> dict[str, dict | None]:
    with Session(eng) as s:
        rows = s.query(ClusterLabel).all()
        return {r.embedding_case: r.payload for r in rows}


def test_spoken_case_prompt_drift_invalidates_only_spoken_cluster_labels(
    monkeypatch, tmp_path
):
    """Mutating ``case_prompts.spoken`` cascades through stage-1 (spoken)
    → stage-2 (spoken) and MUST leave every other case's ``ClusterLabel``
    rows untouched.
    """
    eng = _shared_engine(monkeypatch)
    _seed_full_clip_for_pipeline(eng, tmp_path=tmp_path)
    settings = _full_pipeline_settings(tmp_path)
    cases = default_cases(settings)
    assert "spoken" in cases  # precondition
    _seed_clusters_for_cases(eng, cases)

    gen = _DispatchingGen()
    with _patch_dispatching(gen):
        from modules.labels import run as run_labels

        run_labels(settings, _secrets())

    before = _cluster_payloads_by_case(eng)
    survivors = {c: before[c] for c in before if c != "spoken"}
    assert "spoken" in before and before["spoken"] is not None

    # Drift both spoken prompts to force the cluster-pass dep to change.
    settings.labels.case_prompts = dict(settings.labels.case_prompts)
    settings.labels.case_prompts["spoken"] = (
        settings.labels.case_prompts["spoken"] + "\n# spoken clip drift"
    )
    settings.labels.cluster_case_prompts = dict(settings.labels.cluster_case_prompts)
    settings.labels.cluster_case_prompts["spoken"] = (
        settings.labels.cluster_case_prompts["spoken"] + "\n# spoken cluster drift"
    )

    gen2 = _DispatchingGen()
    with _patch_dispatching(gen2):
        from modules.labels import run as run_labels

        run_labels(settings, _secrets())

    after = _cluster_payloads_by_case(eng)
    # Other cases' payloads are byte-equal to the pre-drift snapshot.
    for case, payload in survivors.items():
        assert after.get(case) == payload, (
            f"case={case} cluster payload changed unexpectedly"
        )
    # Spoken cluster row was wiped + re-created.
    assert after.get("spoken") is not None


def test_cluster_case_prompt_drift_invalidates_only_that_cases_cluster_labels(
    monkeypatch, tmp_path
):
    """Mutating only ``cluster_case_prompts.video`` invalidates the video
    cluster row (config drift on stage-2) while every other case's
    ``ClusterLabel`` rows survive.
    """
    eng = _shared_engine(monkeypatch)
    _seed_full_clip_for_pipeline(eng, tmp_path=tmp_path)
    settings = _full_pipeline_settings(tmp_path)
    cases = default_cases(settings)
    _seed_clusters_for_cases(eng, cases)

    gen = _DispatchingGen()
    with _patch_dispatching(gen):
        from modules.labels import run as run_labels

        run_labels(settings, _secrets())

    before = _cluster_payloads_by_case(eng)
    # Stamp a sentinel into the video cluster payload so a wipe-then-rewrite
    # is detectable byte-wise.
    with Session(eng) as s:
        row = s.get(ClusterLabel, ("video", 0))
        assert row is not None
        row.payload = {"sentinel": True}
        s.commit()

    settings.labels.cluster_case_prompts = dict(settings.labels.cluster_case_prompts)
    settings.labels.cluster_case_prompts["video"] = (
        settings.labels.cluster_case_prompts["video"] + "\n# video cluster drift"
    )

    gen2 = _DispatchingGen()
    with _patch_dispatching(gen2):
        from modules.labels import run as run_labels

        run_labels(settings, _secrets())

    after = _cluster_payloads_by_case(eng)
    # Video row was wiped (sentinel gone) and re-created.
    assert after.get("video") is not None
    assert after["video"] != {"sentinel": True}
    # Every other case is byte-equal to pre-drift.
    for case in before:
        if case == "video":
            continue
        assert after.get(case) == before[case], (
            f"case={case} cluster payload changed unexpectedly"
        )
