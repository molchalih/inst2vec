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
                User(
                    id=1, username="alpha", parse_status="success", user_disqualified=0
                ),
                User(
                    id=2, username="beta", parse_status="success", user_disqualified=1
                ),
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
                    music_id=7,
                    has_music=1,
                    speech_transcription="hello",
                    speech_language="en",
                    speech_translation="hello",
                    has_speech=1,
                    disqualified=0,
                ),
                Clip(
                    id=12,
                    user_id=1,
                    like_count=20,
                    comment_count=2,
                    reshare_count=1,
                    play_count=200,
                    has_music=0,
                    has_speech=0,
                    disqualified=0,
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
                    has_speech=1,
                    has_music=1,
                    music_id=8,
                    disqualified=1,
                ),
            ]
        )
        s.commit()

    from generators.dataset_summary_clips import clips_summary_to_markdown

    out = clips_summary_to_markdown(eng)

    assert out.startswith("| Metric | Value |")
    assert r"| $N$ | 3 |" in out
    assert r"| $N_{\mathrm{kept}}$ | 2 (66.7%) |" in out
    assert r"| $N_{\mathrm{caption}}$ | 1 (50.0%) |" in out
    assert r"| $N_{\mathrm{caption\_trans}}$ | 1 (50.0%) |" in out
    assert r"| $N_{\mathrm{speech}}$ | 1 (50.0%) |" in out
    assert r"| $N_{\mathrm{speech\_trans}}$ | 1 (50.0%) |" in out
    assert r"| $N_{\mathrm{music}}$ | 1 (50.0%) |" in out
    assert r"| $\tilde{x}_{\mathrm{views}}$ | 150 |" in out
    assert r"| $\mu_\mathrm{views}$ | 150.0 |" in out
    assert r"| $[\min-max]_{\mathrm{views}}$ | 100-200 |" in out

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
        s.add(User(id=1, username="alpha", parse_status="success", user_disqualified=0))
        s.add(Clip(id=11, user_id=1, disqualified=0))
        s.commit()

    from generators.dataset_summary_clips import clips_summary_to_markdown

    out = clips_summary_to_markdown(eng)

    assert r"| $\tilde{x}_{\mathrm{views}}$ | - |" in out


def test_render_clips_summary_returns_markdown_object():
    eng = _make_engine()
    with Session(eng) as s:
        s.add(User(id=1, username="alpha", parse_status="success", user_disqualified=0))
        s.add(Clip(id=11, user_id=1, disqualified=0))
        s.commit()

    from docs.quarto_helpers import render_clips_summary
    from generators.dataset_summary_clips import clips_summary_to_markdown

    rendered = render_clips_summary(eng=eng)

    assert isinstance(rendered, Markdown)
    assert rendered.data == clips_summary_to_markdown(eng)
