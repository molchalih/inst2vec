"""Tests for modules/embeddings/text.py."""

from types import SimpleNamespace

import pytest

from modules.embeddings.text import (
    _is_non_english,
    build_audio_text,
    build_gemini_text,
    build_sandwich_text,
    verbalize_mir,
)


@pytest.mark.parametrize(
    "lang,expected",
    [
        (None, False),
        ("", False),
        ("en", False),
        ("EN", False),
        ("eng", False),
        ("en-US", False),
        ("English", False),
        ("ru", True),
        ("fr", True),
        ("DE", True),
    ],
)
def test_is_non_english(lang, expected):
    assert _is_non_english(lang) is expected


def _clip(**kwargs):
    defaults = dict(
        caption_text=None,
        caption_clean=None,
        caption_language=None,
        caption_translation=None,
        speech_transcription=None,
        speech_language=None,
        speech_translation=None,
        is_speech_detected=None,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _mir(**kwargs) -> SimpleNamespace:
    defaults = dict(
        is_music_detected=True,
        genre_labels=None,
        moodtheme_labels=None,
        instrument_labels=None,
        is_acoustic=None,
        is_electronic=None,
        is_instrumental=None,
        is_happy=None,
        is_sad=None,
        is_party=None,
        is_relaxed=None,
        is_aggressive=None,
        is_female_voice=None,
        is_bright_timbre=None,
        is_tonal=None,
        danceability=None,
        engagement=None,
        approachability=None,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


# ── verbalize_mir ─────────────────────────────────────────────────────────────


def test_verbalize_mir_happy_path_full_string():
    mir = _mir(
        genre_labels="Electronic House, Pop Synthpop, Rock Indie",
        moodtheme_labels="energetic, uplifting, party, summer",
        instrument_labels="synthesizer, drums, guitar",
        is_acoustic=False,
        is_electronic=True,
        is_instrumental=False,
        is_happy=True,
        is_sad=False,
        is_party=True,
        is_relaxed=False,
        is_aggressive=False,
        is_female_voice=True,
        is_bright_timbre=True,
        is_tonal=True,
        danceability=0.85,
        engagement=8.0,
        approachability=6.0,
    )
    out = verbalize_mir(mir)

    assert out.startswith("Music: Electronic House, Pop Synthpop —")
    assert "energetic, uplifting, party" in out
    assert "synthesizer, drums" in out
    assert "flags: electronic, happy, party, female vocal, bright timbre, tonal" in out
    assert "highly danceable" in out
    assert "highly engaging" in out
    assert "moderately approachable" in out


def test_verbalize_mir_empty_labels_section_omitted():
    mir = _mir(
        genre_labels="",
        moodtheme_labels="energetic, uplifting",
        instrument_labels=None,
        danceability=0.5,
        engagement=5.0,
        approachability=5.0,
    )
    out = verbalize_mir(mir)
    assert "energetic, uplifting" in out
    assert "Music: ," not in out
    assert "synthesizer" not in out


def test_verbalize_mir_no_flags_when_all_false():
    mir = _mir(
        genre_labels="Pop, Rock",
        is_acoustic=False,
        is_electronic=False,
        is_instrumental=False,
        is_happy=False,
        is_sad=False,
        is_party=False,
        is_relaxed=False,
        is_aggressive=False,
        is_female_voice=False,
        is_bright_timbre=False,
        is_tonal=False,
    )
    assert "flags:" not in verbalize_mir(mir)


def test_verbalize_mir_no_flags_when_all_none():
    assert "flags:" not in verbalize_mir(_mir(genre_labels="Pop"))


@pytest.mark.parametrize(
    "value,expected",
    [
        (0.90, "highly danceable"),
        (0.66, "highly danceable"),
        (0.65, "moderately danceable"),
        (0.33, "moderately danceable"),
        (0.32, "low danceability"),
        (0.00, "low danceability"),
    ],
)
def test_verbalize_mir_danceability_buckets(value, expected):
    assert expected in verbalize_mir(_mir(genre_labels="Pop", danceability=value))


@pytest.mark.parametrize(
    "value,expected",
    [
        (10.0, "highly engaging"),
        (7.0, "highly engaging"),
        (6.9, "moderately engaging"),
        (4.0, "moderately engaging"),
        (3.9, "low engagement"),
        (1.0, "low engagement"),
    ],
)
def test_verbalize_mir_engagement_buckets(value, expected):
    assert expected in verbalize_mir(_mir(genre_labels="Pop", engagement=value))


@pytest.mark.parametrize(
    "value,expected",
    [
        (10.0, "very approachable"),
        (7.0, "very approachable"),
        (6.9, "moderately approachable"),
        (4.0, "moderately approachable"),
        (3.9, "low approachability"),
        (1.0, "low approachability"),
    ],
)
def test_verbalize_mir_approachability_buckets(value, expected):
    assert expected in verbalize_mir(_mir(genre_labels="Pop", approachability=value))


def test_verbalize_mir_returns_empty_when_nothing_to_emit():
    assert verbalize_mir(_mir()) == ""


# ── build_sandwich_text ───────────────────────────────────────────────────────


def test_sandwich_caption_only_when_no_speech_no_music():
    clip = _clip(caption_clean="hello world", is_speech_detected=False)
    out = build_sandwich_text(clip, None)
    assert out == "hello world"


def test_sandwich_uses_caption_translation_for_non_english():
    clip = _clip(
        caption_clean="привет",
        caption_language="ru",
        caption_translation="hello",
        is_speech_detected=False,
    )
    out = build_sandwich_text(clip, None)
    assert out == "hello"


def test_sandwich_speech_block_only_when_is_speech_detected_true():
    for flag, expected_has_speech in [(True, True), (False, False), (None, False)]:
        clip = _clip(
            speech_transcription="this is speech",
            speech_language="en",
            is_speech_detected=flag,
        )
        out = build_sandwich_text(clip, None) or ""
        assert ("this is speech" in out) is expected_has_speech


def test_sandwich_music_block_only_when_is_music_detected_true():
    clip = _clip(caption_clean="cap", is_speech_detected=False)
    on = _mir(genre_labels="Pop", is_music_detected=True)
    off = _mir(genre_labels="Pop", is_music_detected=False)
    none = _mir(genre_labels="Pop", is_music_detected=None)
    assert "Music:" in build_sandwich_text(clip, on)
    out_off = build_sandwich_text(clip, off)
    out_none = build_sandwich_text(clip, none)
    assert out_off == "cap"
    assert out_none == "cap"
    # mir_row=None also excludes the music block.
    assert build_sandwich_text(clip, None) == "cap"


def test_sandwich_returns_none_when_every_block_empty():
    clip = _clip()
    assert build_sandwich_text(clip, None) is None
    assert build_sandwich_text(clip, _mir(is_music_detected=False)) is None


def test_sandwich_joins_caption_speech_music_with_pipe():
    clip = _clip(
        caption_clean="cap",
        is_speech_detected=True,
        speech_transcription="speech",
        speech_language="en",
    )
    mir = _mir(genre_labels="Pop", is_music_detected=True)
    out = build_sandwich_text(clip, mir)
    assert out is not None
    parts = out.split(" | ")
    assert parts[0] == "cap"
    assert parts[1] == "speech"
    assert parts[2].startswith("Music:")


# ── build_audio_text ──────────────────────────────────────────────────────────


def test_audio_returns_none_when_no_speech_no_music():
    assert build_audio_text(_clip(), None) is None
    assert build_audio_text(_clip(is_speech_detected=False), None) is None


def test_audio_speech_only_when_no_music():
    clip = _clip(
        is_speech_detected=True, speech_transcription="speech", speech_language="en"
    )
    out = build_audio_text(clip, None)
    assert out == "speech"


def test_audio_music_only_when_no_speech():
    mir = _mir(genre_labels="Pop", is_music_detected=True)
    out = build_audio_text(_clip(is_speech_detected=False), mir)
    assert out is not None and out.startswith("Music:")


def test_audio_speech_and_music_joined_with_pipe():
    clip = _clip(
        is_speech_detected=True, speech_transcription="speech", speech_language="en"
    )
    mir = _mir(genre_labels="Pop", is_music_detected=True)
    out = build_audio_text(clip, mir)
    assert out is not None
    parts = out.split(" | ")
    assert parts[0] == "speech"
    assert parts[1].startswith("Music:")


def test_audio_excludes_caption():
    """Captions are deliberately excluded from the audio case."""
    clip = _clip(caption_clean="caption text", is_speech_detected=False)
    assert build_audio_text(clip, None) is None


@pytest.mark.parametrize("is_music", [False, None])
def test_audio_excludes_music_when_not_detected(is_music):
    mir = _mir(genre_labels="Pop", is_music_detected=is_music)
    clip = _clip(is_speech_detected=False)
    assert build_audio_text(clip, mir) is None


# ── build_gemini_text (regression — no behavior change) ───────────────────────


def test_gemini_returns_none_when_no_caption_no_speech():
    assert build_gemini_text(_clip(), None) is None


def test_gemini_includes_caption_and_speech_when_present():
    clip = _clip(
        caption_clean="cap",
        is_speech_detected=True,
        speech_transcription="speech",
        speech_language="en",
    )
    out = build_gemini_text(clip, None) or ""
    assert "cap" in out and "speech" in out
    assert "Music:" not in out


def test_gemini_ignores_mir_row():
    """build_gemini_text must not verbalize music, regardless of mir_row."""
    clip = _clip(
        caption_clean="cap",
        is_speech_detected=False,
    )
    mir = _mir(genre_labels="Pop", is_music_detected=True)
    out = build_gemini_text(clip, mir) or ""
    assert "Music:" not in out
