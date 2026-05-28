"""Stage-1 generic runner — ``audio`` case coverage.

Asserts the audio case routes to ``run_text``, that input composition
comes from ``modules.embeddings.text.build_audio_text``, and that the
runner marks ``status='failed', error='no_input'`` when the case
adapter signals no input is available.
"""

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
        case_prompts={"audio": "AUDIO_PROMPT"},
        cluster_case_prompts={"audio": "cluster prompt"},
    )
    base.update(overrides)
    return LabelsSettings(**base)


def _settings() -> SimpleNamespace:
    # video_for is unused for audio, but ``run_case`` reads ``settings.paths``
    # so we keep it present-and-safe.
    return SimpleNamespace(
        paths=SimpleNamespace(video_for=lambda cid: f"/tmp/{cid}.mp4")
    )


def _engine():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


def _clean_audio_json() -> str:
    return json.dumps(
        {
            "observable_audio_tags": [
                {"tag": "calm narration voice", "evidence": "speech transcript"},
                {"tag": "midtempo lo-fi loop", "evidence": "music description"},
                {"tag": "soft ambient room tone", "evidence": "background hum"},
            ],
            "aesthetic_tags": [
                {
                    "tag": "intimate spoken register",
                    "grounded_in": ["calm narration voice"],
                    "confidence": "medium",
                },
                {
                    "tag": "lo-fi domestic mood",
                    "grounded_in": ["midtempo lo-fi loop"],
                    "confidence": "low",
                },
                {
                    "tag": "muted ambient palette",
                    "grounded_in": ["soft ambient room tone"],
                    "confidence": "high",
                },
            ],
            "community_signalling_tags": [
                {
                    "tag": "slow-living spoken voice",
                    "grounded_in": ["intimate spoken register"],
                    "confidence": "medium",
                },
                {
                    "tag": "lofi creator audio palette",
                    "grounded_in": ["lo-fi domestic mood"],
                    "confidence": "low",
                },
                {
                    "tag": "personal-diary listening register",
                    "grounded_in": ["muted ambient palette"],
                    "confidence": "low",
                },
            ],
            "one_sentence_audio_reading": (
                "soft narrated voice over an unhurried lo-fi instrumental bed"
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


def _seed_speech_and_music(eng) -> None:
    with Session(eng) as s:
        s.add(User(id=1, is_selected=True))
        s.add(
            Clip(
                id=1,
                user_id=1,
                is_selected=True,
                is_downloaded=True,
                is_speech_detected=True,
                speech_transcription="a calm slow narration about the day",
                speech_language="en",
            )
        )
        s.add(
            AudioMIR(
                clip_id=1,
                is_music_detected=True,
                genre_labels="lofi, downtempo",
                moodtheme_labels="relaxed, gentle",
                instrument_labels="piano",
            )
        )
        s.commit()


def _seed_silent_clip(eng) -> None:
    with Session(eng) as s:
        s.add(User(id=1, is_selected=True))
        s.add(
            Clip(
                id=1,
                user_id=1,
                is_selected=True,
                is_downloaded=True,
                is_speech_detected=False,
            )
        )
        s.commit()


def test_audio_happy_path_writes_row_with_required_keys():
    eng = _engine()
    _seed_speech_and_music(eng)
    gen = _FakeGen(response=_clean_audio_json())
    with Session(eng) as s:
        run_case(
            session=s,
            settings=_settings(),
            labels=_labels(),
            generator=gen,
            spec=REGISTRY["audio"],
        )
    # ``run_text`` is the only branch the audio case may take.
    assert gen.video_calls == []
    assert len(gen.text_calls) == 1
    prompt, _max_new = gen.text_calls[0]
    assert prompt.startswith("AUDIO_PROMPT\n\n")
    # Speech transcript and MIR verbalization both make it into the prompt.
    assert "calm slow narration" in prompt
    assert "Music:" in prompt and "lofi" in prompt
    with Session(eng) as s:
        row = s.get(ClipLabel, (1, "audio"))
        assert row is not None
        assert row.label_case == "audio"
        assert row.status == "success"
        assert row.validation == "ok"
        expected = {
            "observable_audio_tags",
            "aesthetic_tags",
            "community_signalling_tags",
            "one_sentence_audio_reading",
        }
        assert set(row.payload) == expected
        # Non-video cases stamp the source hash from the case adapter input.
        assert isinstance(row.source_hash, str) and len(row.source_hash) == 64


def test_audio_no_input_marks_failed_no_input():
    eng = _engine()
    _seed_silent_clip(eng)
    gen = _FakeGen(response=_clean_audio_json())
    with Session(eng) as s:
        run_case(
            session=s,
            settings=_settings(),
            labels=_labels(max_attempts=1),
            generator=gen,
            spec=REGISTRY["audio"],
        )
    assert gen.text_calls == []
    with Session(eng) as s:
        row = s.get(ClipLabel, (1, "audio"))
        assert row is not None
        assert row.status == "failed"
        assert row.error == "no_input"


def test_audio_does_not_call_run_with_video():
    eng = _engine()
    _seed_speech_and_music(eng)
    gen = _FakeGen(response=_clean_audio_json())
    with Session(eng) as s:
        run_case(
            session=s,
            settings=_settings(),
            labels=_labels(),
            generator=gen,
            spec=REGISTRY["audio"],
        )
    assert gen.video_calls == []
