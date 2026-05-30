"""Tests for the spoken / textual label clip-input adapters."""

from __future__ import annotations

from types import SimpleNamespace

from modules.labels.inputs import spoken_input, textual_input


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


def test_spoken_input_speech_only():
    clip = _clip(
        is_speech_detected=True, speech_language="en", speech_transcription="hi"
    )
    assert spoken_input(clip, None, None) == "hi"


def test_spoken_input_none_without_speech():
    clip = _clip(is_speech_detected=False, speech_transcription="ignored")
    assert spoken_input(clip, None, None) is None


def test_spoken_input_ignores_mir_and_visual():
    clip = _clip(
        is_speech_detected=True, speech_language="en", speech_transcription="hi"
    )
    assert spoken_input(clip, object(), {"x": 1}) == "hi"


def test_textual_input_caption_only():
    clip = _clip(caption_language="en", caption_clean="a caption")
    assert textual_input(clip, None, None) == "a caption"


def test_textual_input_none_when_empty():
    clip = _clip(caption_language="en", caption_clean="", caption_text="")
    assert textual_input(clip, None, None) is None


def test_textual_input_ignores_mir_and_visual():
    clip = _clip(caption_language="en", caption_clean="cap")
    assert textual_input(clip, object(), {"x": 1}) == "cap"
