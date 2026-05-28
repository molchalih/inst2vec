"""Stage-1 generic runner — ``video`` case coverage.

Mirrors ``tests/test_labels_pipeline.py`` style: an in-memory sqlite
engine, a thin ``_FakeGen`` stand-in for ``LabelsGenerator``, and a
minimal ``LabelsSettings`` + ``paths`` stub. Exercises the generic
``run_case`` runner directly so the video case is the integration
contract for the rest of Phase B.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from core.config import LabelsSettings
from core.database import Base, Clip, ClipLabel, User
from modules.labels.cases import REGISTRY
from modules.labels.clip_pass import run_case


def _labels(**overrides) -> LabelsSettings:
    base = dict(
        case_prompts={
            "video": "VIDEO_PROMPT",
            "audio": "AUDIO_PROMPT",
        },
        cluster_case_prompts={"video": "cluster prompt"},
    )
    base.update(overrides)
    return LabelsSettings(**base)


def _settings(tmp_path: Path) -> SimpleNamespace:
    paths = SimpleNamespace(
        video_dir=str(tmp_path),
        video_for=lambda cid: tmp_path / f"{cid}.mp4",
    )
    return SimpleNamespace(paths=paths)


def _engine():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


def _seed(eng, *, clip_ids: tuple[int, ...]) -> None:
    with Session(eng) as s:
        s.add(User(id=1, is_selected=True))
        for cid in clip_ids:
            s.add(Clip(id=cid, user_id=1, is_selected=True, is_downloaded=True))
        s.commit()


def _clean_video_json() -> str:
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


@dataclass
class _FakeGen:
    """Records ``run`` / ``run_text`` calls and returns a canned JSON body."""

    response: str = ""
    video_calls: list[tuple] = field(default_factory=list)
    text_calls: list[tuple] = field(default_factory=list)

    def run(self, video_path, prompt: str) -> str:
        self.video_calls.append((str(video_path), prompt))
        return self.response

    def run_text(self, prompt: str, *, max_new_tokens: int) -> str:
        self.text_calls.append((prompt, max_new_tokens))
        return self.response


def test_run_case_video_uses_run_with_video_path(tmp_path):
    eng = _engine()
    _seed(eng, clip_ids=(1, 2))
    gen = _FakeGen(response=_clean_video_json())
    settings = _settings(tmp_path)
    with Session(eng) as s:
        run_case(
            session=s,
            settings=settings,
            labels=_labels(),
            generator=gen,
            spec=REGISTRY["video"],
        )
    assert gen.text_calls == []
    assert len(gen.video_calls) == 2
    seen = sorted(p for (p, _) in gen.video_calls)
    assert seen == [str(tmp_path / "1.mp4"), str(tmp_path / "2.mp4")]
    # Prompt body for video case must be the case-specific prompt verbatim.
    assert all(prompt == "VIDEO_PROMPT" for (_, prompt) in gen.video_calls)


def test_run_case_video_writes_ClipLabel_with_label_case_video(tmp_path):
    eng = _engine()
    _seed(eng, clip_ids=(1,))
    gen = _FakeGen(response=_clean_video_json())
    with Session(eng) as s:
        run_case(
            session=s,
            settings=_settings(tmp_path),
            labels=_labels(),
            generator=gen,
            spec=REGISTRY["video"],
        )
    with Session(eng) as s:
        row = s.get(ClipLabel, (1, "video"))
        assert row is not None
        assert row.label_case == "video"
        assert row.status == "success"
        assert row.validation == "ok"
        assert row.payload is not None
        # Video case must not stamp a ``source_hash`` (frames live on disk).
        assert row.source_hash is None


def test_run_case_video_idempotent_rerun_no_model_calls(tmp_path):
    eng = _engine()
    _seed(eng, clip_ids=(1, 2))
    gen = _FakeGen(response=_clean_video_json())
    settings = _settings(tmp_path)
    labels = _labels()
    with Session(eng) as s:
        run_case(
            session=s,
            settings=settings,
            labels=labels,
            generator=gen,
            spec=REGISTRY["video"],
        )
    first = list(gen.video_calls)
    with Session(eng) as s:
        run_case(
            session=s,
            settings=settings,
            labels=labels,
            generator=gen,
            spec=REGISTRY["video"],
        )
    assert len(first) == 2
    # Second run must find every row terminal and never invoke the generator.
    assert gen.video_calls == first


def test_run_case_video_prompt_drift_wipes_only_video_case(tmp_path):
    eng = _engine()
    _seed(eng, clip_ids=(1,))
    # Pre-seed an unrelated audio-case ClipLabel that the video drift must not
    # touch (the per-case wipe deletes only ``label_case == "video"``).
    with Session(eng) as s:
        s.add(
            ClipLabel(
                clip_id=1,
                label_case="audio",
                status="success",
                validation="ok",
                payload={"keep": True},
                warnings=[],
                attempts=1,
            )
        )
        s.commit()
    gen = _FakeGen(response=_clean_video_json())
    settings = _settings(tmp_path)
    with Session(eng) as s:
        run_case(
            session=s,
            settings=settings,
            labels=_labels(),
            generator=gen,
            spec=REGISTRY["video"],
        )
    # Drift the video case prompt only and rerun.
    drifted = _labels(
        case_prompts={
            "video": "DIFFERENT_VIDEO_PROMPT",
            "audio": "AUDIO_PROMPT",
        }
    )
    with Session(eng) as s:
        run_case(
            session=s,
            settings=settings,
            labels=drifted,
            generator=gen,
            spec=REGISTRY["video"],
        )
    with Session(eng) as s:
        audio_row = s.get(ClipLabel, (1, "audio"))
        video_row = s.get(ClipLabel, (1, "video"))
        assert audio_row is not None and audio_row.payload == {"keep": True}
        # Video case re-ran with the new prompt body.
        assert video_row is not None and video_row.status == "success"
    assert any(prompt == "DIFFERENT_VIDEO_PROMPT" for (_, prompt) in gen.video_calls)
