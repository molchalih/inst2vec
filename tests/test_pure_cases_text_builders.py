"""Tests for the speech-only / caption-only embedding text builders."""

from __future__ import annotations

from types import SimpleNamespace

from modules.embeddings.text import build_spoken_text, build_textual_text


def _clip(**kw):
    base = dict(
        caption_clean=None,
        caption_text=None,
        caption_language=None,
        caption_translation=None,
        is_speech_detected=None,
        speech_transcription=None,
        speech_language=None,
        speech_translation=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


# ── spoken (speech transcript only) ──────────────────────────────────────────


def test_spoken_english_uses_transcription():
    clip = _clip(
        is_speech_detected=True,
        speech_language="en",
        speech_transcription="hello world",
    )
    assert build_spoken_text(clip, None) == "hello world"


def test_spoken_non_english_uses_translation():
    clip = _clip(
        is_speech_detected=True,
        speech_language="fr",
        speech_transcription="bonjour",
        speech_translation="hello",
    )
    assert build_spoken_text(clip, None) == "hello"


def test_spoken_none_when_no_speech():
    clip = _clip(is_speech_detected=False, speech_transcription="ignored")
    assert build_spoken_text(clip, None) is None


def test_spoken_none_when_empty_transcript():
    clip = _clip(is_speech_detected=True, speech_language="en", speech_transcription="")
    assert build_spoken_text(clip, None) is None


def test_spoken_ignores_mir_row():
    clip = _clip(
        is_speech_detected=True, speech_language="en", speech_transcription="hi"
    )
    sentinel_mir = object()
    assert build_spoken_text(clip, sentinel_mir) == "hi"


# ── textual (caption only) ───────────────────────────────────────────────────


def test_textual_english_uses_caption_clean():
    clip = _clip(caption_language="en", caption_clean="a clean caption")
    assert build_textual_text(clip, None) == "a clean caption"


def test_textual_non_english_uses_translation():
    clip = _clip(
        caption_language="es",
        caption_clean="hola",
        caption_translation="hello",
    )
    assert build_textual_text(clip, None) == "hello"


def test_textual_falls_back_to_caption_text():
    clip = _clip(caption_language="en", caption_clean=None, caption_text="raw caption")
    assert build_textual_text(clip, None) == "raw caption"


def test_textual_none_when_empty():
    clip = _clip(caption_language="en", caption_clean="", caption_text="")
    assert build_textual_text(clip, None) is None
