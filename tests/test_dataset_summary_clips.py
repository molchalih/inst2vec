import os
import sys

from IPython.display import Markdown
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.database import Base, Clip, User


def _make_engine():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


def test_clips_summary_to_markdown_renders_curated_metrics():
    eng = _make_engine()
    with Session(eng) as s:
        s.add_all(
            [
                User(id=1, parse_status="success", is_eligible=True),
                User(id=2, parse_status="success", is_eligible=False),
                Clip(
                    id=11,
                    user_id=1,
                    caption_text="hello",
                    caption_language="en",
                    caption_translation="hi",
                    like_count=10,
                    comment_count=2,
                    reshare_count=1,
                    play_count=100,
                    speech_transcription="hello",
                    speech_language="en",
                    speech_translation="hello",
                    is_speech_detected=1,
                    is_selected=True,
                    is_downloaded=True,
                ),
                Clip(
                    id=12,
                    user_id=1,
                    like_count=20,
                    comment_count=2,
                    reshare_count=1,
                    play_count=200,
                    is_speech_detected=0,
                    is_selected=True,
                    is_downloaded=True,
                ),
                Clip(
                    id=13,
                    user_id=2,
                    caption_text="bonjour",
                    caption_language="fr",
                    caption_translation="hello",
                    like_count=10_000,
                    comment_count=100,
                    reshare_count=200,
                    play_count=10_000,
                    speech_transcription="bonjour",
                    speech_language="fr",
                    speech_translation="hello",
                    is_speech_detected=1,
                    is_selected=False,
                ),
            ]
        )
        s.commit()

    from docs.reporting.tables.clips import (
        clips_summary_to_markdown,
    )

    out = clips_summary_to_markdown(eng)

    assert out.startswith("| Metric | Value |")
    assert "| total clips | 3 |" in out
    assert "| kept clips | 2 (66.7%) |" in out
    assert "| with caption | 1 (50.0%) |" in out
    assert "| with caption translation | 1 (50.0%) |" in out
    assert "| with speech | 1 (50.0%) |" in out
    assert "| with speech translation | 1 (50.0%) |" in out
    assert "| median views | 150 |" in out
    assert "| mean views | 150.0 |" in out
    assert "| min–max views | 100-200 |" in out

    assert "| Clips disqualified |" not in out
    assert "| Clips with caption language |" not in out
    assert "| Clips with speech transcription |" not in out
    assert "| Clips with speech language |" not in out
    assert "| Clips linked to music row |" not in out
    assert "| Like count" not in out
    assert "| Comment count" not in out
    assert "| Reshare count" not in out


def test_clips_summary_to_markdown_uses_dash_for_missing_numeric_values():
    eng = _make_engine()
    with Session(eng) as s:
        s.add(User(id=1, parse_status="success", is_eligible=True))
        s.add(Clip(id=11, user_id=1, is_selected=True, is_downloaded=True))
        s.commit()

    from docs.reporting.tables.clips import (
        clips_summary_to_markdown,
    )

    out = clips_summary_to_markdown(eng)

    assert "| median views | - |" in out


def test_render_clips_summary_returns_markdown_object():
    eng = _make_engine()
    with Session(eng) as s:
        s.add(User(id=1, parse_status="success", is_eligible=True))
        s.add(Clip(id=11, user_id=1, is_selected=True, is_downloaded=True))
        s.commit()

    from docs.quarto_helpers import render_clips_summary
    from docs.reporting.tables.clips import (
        clips_summary_to_markdown,
    )

    rendered = render_clips_summary(eng=eng)

    assert isinstance(rendered, Markdown)
    assert rendered.data == clips_summary_to_markdown(eng)
