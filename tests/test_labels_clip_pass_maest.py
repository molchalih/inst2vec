"""Stage-1 generic runner — ``maest`` (music) case coverage."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import SimpleNamespace

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from core.config import LabelsSettings
from core.database import AudioMIR, Base, Clip, ClipLabel, User
from modules.labels.cases import REGISTRY
from modules.labels.clip_pass import run_case


def _labels(**overrides) -> LabelsSettings:
    base = dict(
        case_prompts={"maest": "MAEST_PROMPT"},
        cluster_case_prompts={"maest": "cluster prompt"},
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


def _clean_music_json() -> str:
    return json.dumps(
        {
            "observable_music_tags": [
                {"tag": "lofi mid-tempo loop", "evidence": "lofi"},
                {"tag": "warm piano lead", "evidence": "piano"},
                {"tag": "relaxed mood bed", "evidence": "relaxed"},
            ],
            "aesthetic_tags": [
                {
                    "tag": "soft instrumental palette",
                    "grounded_in": ["lofi mid-tempo loop"],
                    "confidence": "medium",
                },
                {
                    "tag": "warm tonal register",
                    "grounded_in": ["warm piano lead"],
                    "confidence": "low",
                },
                {
                    "tag": "mellow listening mood",
                    "grounded_in": ["relaxed mood bed"],
                    "confidence": "high",
                },
            ],
            "community_signalling_tags": [
                {
                    "tag": "lofi listening register",
                    "grounded_in": ["soft instrumental palette"],
                    "confidence": "medium",
                },
                {
                    "tag": "study-music creator palette",
                    "grounded_in": ["warm tonal register"],
                    "confidence": "low",
                },
                {
                    "tag": "personal-mood music register",
                    "grounded_in": ["mellow listening mood"],
                    "confidence": "low",
                },
            ],
            "one_sentence_music_reading": (
                "warm midtempo lo-fi piano bed in a relaxed instrumental palette"
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


def _seed_music(eng) -> None:
    with Session(eng) as s:
        s.add(User(id=1, is_selected=True))
        s.add(Clip(id=1, user_id=1, is_selected=True, is_downloaded=True))
        s.add(
            AudioMIR(
                clip_id=1,
                is_music_detected=True,
                genre_labels="lofi, downtempo",
                moodtheme_labels="relaxed, mellow",
                instrument_labels="piano",
            )
        )
        s.commit()


def _seed_no_music(eng) -> None:
    with Session(eng) as s:
        s.add(User(id=1, is_selected=True))
        s.add(Clip(id=1, user_id=1, is_selected=True, is_downloaded=True))
        s.add(AudioMIR(clip_id=1, is_music_detected=False))
        s.commit()


def test_maest_happy_path_uses_verbalize_mir():
    eng = _engine()
    _seed_music(eng)
    gen = _FakeGen(response=_clean_music_json())
    with Session(eng) as s:
        run_case(
            session=s,
            settings=_settings(),
            labels=_labels(),
            generator=gen,
            spec=REGISTRY["maest"],
        )
    assert gen.video_calls == []
    assert len(gen.text_calls) == 1
    prompt, _ = gen.text_calls[0]
    # ``verbalize_mir`` always prefixes ``Music:`` — that's our wire signal.
    assert prompt.startswith("MAEST_PROMPT\n\nMusic:")
    assert "lofi" in prompt and "piano" in prompt
    with Session(eng) as s:
        row = s.get(ClipLabel, (1, "maest"))
        assert row is not None
        assert row.label_case == "maest"
        assert row.status == "success"
        expected = {
            "observable_music_tags",
            "aesthetic_tags",
            "community_signalling_tags",
            "one_sentence_music_reading",
        }
        assert set(row.payload) == expected


def test_maest_no_music_marks_failed_no_music():
    eng = _engine()
    _seed_no_music(eng)
    gen = _FakeGen(response=_clean_music_json())
    with Session(eng) as s:
        run_case(
            session=s,
            settings=_settings(),
            labels=_labels(max_attempts=1),
            generator=gen,
            spec=REGISTRY["maest"],
        )
    assert gen.text_calls == []
    with Session(eng) as s:
        row = s.get(ClipLabel, (1, "maest"))
        assert row is not None
        assert row.status == "failed"
        assert row.error == "no_music"
