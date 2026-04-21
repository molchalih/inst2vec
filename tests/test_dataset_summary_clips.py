import os
import sys

from IPython.display import Markdown
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from modules.database import Base, Clip, User


def _make_engine():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


def test_clips_summary_to_markdown_renders_curated_metrics():
    eng = _make_engine()
    with Session(eng) as s:
        s.add_all(
            [
                User(pk=1, username="alpha", parse_status="success", user_disqualified=0),
                User(pk=2, username="beta", parse_status="success", user_disqualified=1),
                Clip(
                    pk=11,
                    user_pk=1,
                    caption_text="hello",
                    caption_language="en",
                    like_count=10,
                    comment_count=1,
                    reshare_count=0,
                    play_count=100,
                    music_id=7,
                    has_music=1,
                    speech_transcription="hello",
                    speech_language="en",
                    has_speech=1,
                    disqualified=0,
                ),
                Clip(
                    pk=12,
                    user_pk=1,
                    like_count=20,
                    comment_count=2,
                    reshare_count=1,
                    play_count=200,
                    has_music=0,
                    has_speech=0,
                    disqualified=1,
                ),
                Clip(
                    pk=13,
                    user_pk=2,
                    caption_text="bonjour",
                    caption_language="fr",
                    caption_translation="hello",
                    like_count=30,
                    comment_count=3,
                    reshare_count=2,
                    play_count=300,
                    speech_transcription="bonjour",
                    speech_language="fr",
                    speech_translation="hello",
                    has_speech=1,
                    disqualified=0,
                ),
            ]
        )
        s.commit()

    from generators.dataset_summary_clips import clips_summary_to_markdown

    out = clips_summary_to_markdown(eng)

    assert out.startswith("| Metric | Value |")
    assert "| Total clips | 3 |" in out
    assert "| Clips kept | 2 (66.7%) |" in out
    assert "| Clips disqualified | 1 (33.3%) |" in out
    assert "| Clips with caption text | 2 (66.7%) |" in out
    assert "| Clips with caption language | 2 (66.7%) |" in out
    assert "| Clips with caption translation | 1 (33.3%) |" in out
    assert "| Clips with speech | 2 (66.7%) |" in out
    assert "| Clips with speech transcription | 2 (66.7%) |" in out
    assert "| Clips with speech language | 2 (66.7%) |" in out
    assert "| Clips with speech translation | 1 (33.3%) |" in out
    assert "| Clips with music | 1 (33.3%) |" in out
    assert "| Clips linked to music row | 1 (33.3%) |" in out
    assert "| Play count (median, mean, min-max) | 200, 200.0, 100-300 |" in out
    assert "| Like count (median, mean, min-max) | 20, 20.0, 10-30 |" in out
    assert "| Comment count (median, mean, min-max) | 2, 2.0, 1-3 |" in out
    assert "| Reshare count (median, mean, min-max) | 1, 1.0, 0-2 |" in out


def test_clips_summary_to_markdown_uses_dash_for_missing_numeric_values():
    eng = _make_engine()
    with Session(eng) as s:
        s.add(User(pk=1, username="alpha", parse_status="success", user_disqualified=0))
        s.add(Clip(pk=11, user_pk=1, disqualified=0))
        s.commit()

    from generators.dataset_summary_clips import clips_summary_to_markdown

    out = clips_summary_to_markdown(eng)

    assert "| Play count (median, mean, min-max) | - |" in out


def test_render_clips_summary_returns_markdown_object():
    eng = _make_engine()
    with Session(eng) as s:
        s.add(User(pk=1, username="alpha", parse_status="success", user_disqualified=0))
        s.add(Clip(pk=11, user_pk=1, disqualified=0))
        s.commit()

    from docs.quarto_helpers import render_clips_summary
    from generators.dataset_summary_clips import clips_summary_to_markdown

    rendered = render_clips_summary(eng=eng)

    assert isinstance(rendered, Markdown)
    assert rendered.data == clips_summary_to_markdown(eng)
