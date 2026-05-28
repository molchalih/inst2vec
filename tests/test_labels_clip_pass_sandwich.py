"""Stage-1 generic runner — ``sandwich`` case coverage.

Sandwich is the only case with a *cross-case* upstream dependency: it
consumes the video case's ``ClipLabel.payload`` and composes
``stage_dependency_hash(LABELS, "video")`` into its own fingerprint so a
re-run of the video case stage-1 invalidates sandwich stage-1.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import SimpleNamespace

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from core.config import LabelsSettings
from core.database import AudioMIR, Base, Clip, ClipLabel, StageState, User
from modules.labels.cases import REGISTRY
from modules.labels.clip_pass import run_case
from modules.labels.state import STAGE_LABELS


def _labels(**overrides) -> LabelsSettings:
    base = dict(
        case_prompts={"sandwich": "SANDWICH_PROMPT"},
        cluster_case_prompts={"sandwich": "cluster prompt"},
    )
    base.update(overrides)
    return LabelsSettings(**base)


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        paths=SimpleNamespace(video_for=lambda cid: f"/tmp/{cid}.mp4")
    )


def _engine():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


def _clean_sandwich_json() -> str:
    return json.dumps(
        {
            "observable_multimodal_tags": [
                {"tag": "warm kitchen on screen", "evidence": "visual + caption"},
                {"tag": "calm narration voice", "evidence": "speech transcript"},
                {"tag": "midtempo lo-fi loop", "evidence": "music description"},
            ],
            "aesthetic_tags": [
                {
                    "tag": "soft domestic vignette",
                    "grounded_in": ["warm kitchen on screen"],
                    "confidence": "medium",
                },
                {
                    "tag": "intimate spoken register",
                    "grounded_in": ["calm narration voice"],
                    "confidence": "low",
                },
                {
                    "tag": "lo-fi domestic mood",
                    "grounded_in": ["midtempo lo-fi loop"],
                    "confidence": "high",
                },
            ],
            "community_signalling_tags": [
                {
                    "tag": "slow-living domestic voice",
                    "grounded_in": ["soft domestic vignette"],
                    "confidence": "medium",
                },
                {
                    "tag": "personal-diary creator register",
                    "grounded_in": ["intimate spoken register"],
                    "confidence": "low",
                },
                {
                    "tag": "lofi creator audio palette",
                    "grounded_in": ["lo-fi domestic mood"],
                    "confidence": "low",
                },
            ],
            "one_sentence_multimodal_reading": (
                "warm domestic kitchen scene over a soft narrated lo-fi instrumental"
            ),
        }
    )


@dataclass
class _FakeGen:
    response: str = ""
    video_calls: list = field(default_factory=list)
    text_calls: list = field(default_factory=list)

    def run(self, video_path, prompt: str) -> str:
        self.video_calls.append((str(video_path), prompt))
        return self.response

    def run_text(self, prompt: str, *, max_new_tokens: int) -> str:
        self.text_calls.append((prompt, max_new_tokens))
        return self.response


def _seed_clip(eng) -> None:
    with Session(eng) as s:
        s.add(User(id=1, is_selected=True))
        s.add(
            Clip(
                id=1,
                user_id=1,
                is_selected=True,
                is_downloaded=True,
                caption_clean="a warm kitchen scene",
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


def _seed_video_label(eng, payload: dict) -> None:
    with Session(eng) as s:
        s.add(
            ClipLabel(
                clip_id=1,
                label_case="video",
                status="success",
                validation="ok",
                payload=payload,
                warnings=[],
                attempts=1,
            )
        )
        s.commit()


def test_sandwich_uses_video_label_payload_in_prompt():
    eng = _engine()
    _seed_clip(eng)
    visual_payload = {
        "observable_visual_tags": [{"tag": "warm kitchen", "evidence": "lamp"}],
        "one_sentence_visual_reading": "warm domestic kitchen scene",
    }
    _seed_video_label(eng, visual_payload)
    gen = _FakeGen(response=_clean_sandwich_json())
    with Session(eng) as s:
        run_case(
            session=s,
            settings=_settings(),
            labels=_labels(),
            generator=gen,
            spec=REGISTRY["sandwich"],
        )
    assert gen.video_calls == []
    assert len(gen.text_calls) == 1
    prompt, _ = gen.text_calls[0]
    # Visual payload JSON is embedded in the prompt body via sandwich_input.
    assert "VISUAL_OBSERVATIONS=" in prompt
    assert "warm domestic kitchen scene" in prompt
    # Sandwich text from build_sandwich_text is also present.
    assert "a calm slow narration" in prompt


def test_sandwich_missing_video_label_marks_failed_missing_video_label():
    eng = _engine()
    _seed_clip(eng)
    # No video ClipLabel seeded: sandwich must short-circuit.
    gen = _FakeGen(response=_clean_sandwich_json())
    with Session(eng) as s:
        run_case(
            session=s,
            settings=_settings(),
            labels=_labels(max_attempts=1),
            generator=gen,
            spec=REGISTRY["sandwich"],
        )
    assert gen.text_calls == []
    with Session(eng) as s:
        row = s.get(ClipLabel, (1, "sandwich"))
        assert row is not None
        assert row.status == "failed"
        assert row.error == "missing_video_label"


def test_sandwich_with_video_label_writes_success_row_with_label_case_sandwich():
    eng = _engine()
    _seed_clip(eng)
    _seed_video_label(
        eng,
        {
            "observable_visual_tags": [{"tag": "warm kitchen", "evidence": "lamp"}],
            "one_sentence_visual_reading": "warm scene",
        },
    )
    gen = _FakeGen(response=_clean_sandwich_json())
    with Session(eng) as s:
        run_case(
            session=s,
            settings=_settings(),
            labels=_labels(),
            generator=gen,
            spec=REGISTRY["sandwich"],
        )
    with Session(eng) as s:
        row = s.get(ClipLabel, (1, "sandwich"))
        assert row is not None
        assert row.label_case == "sandwich"
        assert row.status == "success"
        assert row.validation == "ok"
        expected = {
            "observable_multimodal_tags",
            "aesthetic_tags",
            "community_signalling_tags",
            "one_sentence_multimodal_reading",
        }
        assert set(row.payload) == expected
        # Sandwich is non-video → source_hash stamped from the composed input.
        assert isinstance(row.source_hash, str) and len(row.source_hash) == 64


def test_sandwich_dep_drift_wipes_existing_rows():
    """Drifting the video case's StageState causes sandwich rows to be wiped.

    With ``check_dependency=True`` in the gate call, a change to the video
    case's StageState row (simulating a video-case re-run) must trigger
    ``on_drift`` for the sandwich scope, deleting the stale ClipLabel rows
    before re-running the clips.
    """
    eng = _engine()
    _seed_clip(eng)
    video_payload = {
        "observable_visual_tags": [{"tag": "warm kitchen", "evidence": "lamp"}],
        "one_sentence_visual_reading": "warm scene",
    }
    _seed_video_label(eng, video_payload)
    # Seed a video StageState so sandwich's dep hash is non-trivial.
    with Session(eng) as s:
        s.merge(
            StageState(
                stage_name=STAGE_LABELS,
                scope_key="video",
                data_hash="dA",
                config_hash="cA",
                dependency_hash="depA",
            )
        )
        s.commit()
    settings = _settings()
    labels = _labels()
    spec = REGISTRY["sandwich"]
    gen = _FakeGen(response=_clean_sandwich_json())
    # First run: no sandwich StageState yet → gate skips wipe, runs clip,
    # marks complete.
    with Session(eng) as s:
        run_case(
            session=s,
            settings=settings,
            labels=labels,
            generator=gen,
            spec=spec,
        )
    with Session(eng) as s:
        row = s.get(ClipLabel, (1, "sandwich"))
        assert row is not None and row.status == "success"
    # Stamp a sentinel payload directly so we can detect a wipe unambiguously.
    with Session(eng) as s:
        row = s.get(ClipLabel, (1, "sandwich"))
        row.payload = {"sentinel": True}
        s.commit()
    # Drift the video StageState — sandwich's dependency hash changes.
    with Session(eng) as s:
        video_ss = s.get(StageState, (STAGE_LABELS, "video"))
        video_ss.config_hash = "cB"
        s.commit()
    gen2 = _FakeGen(response=_clean_sandwich_json())
    # Second run: gate detects dep drift → wipes sandwich rows → re-runs clip.
    with Session(eng) as s:
        run_case(
            session=s,
            settings=settings,
            labels=labels,
            generator=gen2,
            spec=spec,
        )
    with Session(eng) as s:
        row = s.get(ClipLabel, (1, "sandwich"))
        # Row was wiped and re-created from fresh generator output.
        assert row is not None
        assert row.payload != {"sentinel": True}
        assert row.status == "success"
