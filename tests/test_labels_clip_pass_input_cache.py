"""The per-clip text adapter must run exactly once per clip per case."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from unittest.mock import MagicMock

from sqlalchemy.orm import Session

from core.config import LabelsSettings
from core.database import Clip, User, init_db
from core.database.engine import get_engine
from modules.labels import clip_pass
from modules.labels.cases import SPOKEN_CASE


def _seed(session: Session) -> None:
    session.add(User(id=1))
    for cid in (10, 11):
        session.add(
            Clip(
                id=cid,
                user_id=1,
                is_selected=True,
                is_downloaded=True,
                is_speech_detected=True,
                speech_transcription="some words",
                speech_language="en",
            )
        )
    session.commit()


def test_spoken_input_adapter_runs_once_per_clip(monkeypatch) -> None:
    init_db("sqlite:///:memory:", "sqlite:///:memory:")
    with Session(get_engine()) as session:
        _seed(session)
        calls: Counter[int] = Counter()
        original = SPOKEN_CASE.clip_input

        def counting_adapter(clip, mir_row, visual_payload):
            calls[clip.id] += 1
            return original(clip, mir_row, visual_payload)

        spec = replace(SPOKEN_CASE, clip_input=counting_adapter)

        labels = LabelsSettings(
            case_prompts={"spoken": "PROMPT"},
            cluster_case_prompts={"spoken": "C"},
            max_attempts=1,
        )
        settings = MagicMock()
        settings.labels = labels
        settings.paths.video_for = lambda cid: f"/nope/{cid}.mp4"

        gen = MagicMock()
        gen.run_text.return_value = "{}"  # validator will reject — fine.

        clip_pass.run_case(
            session=session,
            settings=settings,
            labels=labels,
            generator=gen,
            spec=spec,
        )

        # Two pending clips × 1 adapter call (fingerprint + generation share cache).
        assert calls[10] == 1
        assert calls[11] == 1
