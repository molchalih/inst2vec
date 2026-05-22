from types import SimpleNamespace

import pytest

from modules.embeddings.text import (
    _is_non_english,
    build_audio_text,
    build_gemini_text,
    build_sandwich_text,
    verbalize_music,
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


def _music(**kwargs):
    defaults = dict(
        artist="Test Artist",
        track="Test Track",
        energy=None,
        valence=None,
        acousticness=None,
        instrumentalness=None,
        danceability=None,
        speechiness=None,
        tempo=None,
        mode=None,
        key=None,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _clip(**kwargs):
    defaults = dict(
        caption_text=None,
        caption_clean=None,
        caption_language=None,
        caption_translation=None,
        speech_transcription=None,
        speech_language=None,
        speech_translation=None,
        is_speech_detected=True,
        music_id=None,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


# ── verbalize_music ───────────────────────────────────────────────────────────


def test_verbalize_music_prefix():
    m = _music(
        track="Laura Palmer's Theme",
        artist="Angelo Badalamenti",
        energy=0.01,
        valence=0.04,
        acousticness=0.84,
        instrumentalness=0.91,
        danceability=0.06,
        speechiness=0.03,
        tempo=68.0,
        mode=0,
        key=3,
    )
    result = verbalize_music(m)
    assert result.startswith('Music: "Laura Palmer\'s Theme" by Angelo Badalamenti — ')


def test_verbalize_music_low_energy_melancholic_acoustic_instrumental():
    m = _music(
        energy=0.01,
        valence=0.04,
        acousticness=0.84,
        instrumentalness=0.91,
        danceability=0.06,
        speechiness=0.03,
        tempo=68.0,
        mode=0,
        key=3,
    )
    result = verbalize_music(m)
    assert "low energy" in result
    assert "dark and melancholic" in result
    assert "acoustic" in result
    assert "instrumental" in result
    assert "not danceable" in result
    assert "very slow (68 BPM)" in result
    assert "D# minor" in result


def test_verbalize_music_high_energy_upbeat_electronic_vocal():
    m = _music(
        energy=0.90,
        valence=0.79,
        acousticness=0.01,
        instrumentalness=0.00,
        danceability=0.52,
        speechiness=0.04,
        tempo=117.0,
        mode=1,
        key=9,
    )
    result = verbalize_music(m)
    assert "very high energy" in result
    assert "very upbeat" in result
    assert "electronic" in result
    assert "vocal" in result
    assert "moderate tempo (117 BPM)" in result
    assert "A major" in result
    assert "danceable" not in result


def test_verbalize_music_no_key_mode():
    m = _music(
        energy=0.37,
        valence=0.18,
        acousticness=0.55,
        instrumentalness=0.84,
        danceability=0.24,
        speechiness=0.03,
        tempo=66.0,
        mode=None,
        key=None,
    )
    result = verbalize_music(m)
    assert "minor" not in result
    assert "major" not in result
    assert "instrumental" in result
    assert "very slow (66 BPM)" in result


def test_verbalize_music_speechy():
    m = _music(
        energy=0.50,
        valence=0.40,
        acousticness=0.30,
        instrumentalness=0.00,
        danceability=0.50,
        speechiness=0.70,
        tempo=95.0,
        mode=None,
        key=None,
    )
    result = verbalize_music(m)
    assert "spoken word" in result


def test_verbalize_music_rap():
    m = _music(
        energy=0.70,
        valence=0.50,
        acousticness=0.10,
        instrumentalness=0.00,
        danceability=0.80,
        speechiness=0.45,
        tempo=95.0,
        mode=None,
        key=None,
    )
    result = verbalize_music(m)
    assert "rap or speech-heavy" in result
    assert "highly danceable" in result


def test_verbalize_music_fast_tempo():
    m = _music(
        energy=0.85,
        valence=0.60,
        acousticness=0.05,
        instrumentalness=0.00,
        danceability=0.70,
        speechiness=0.05,
        tempo=175.0,
        mode=1,
        key=0,
    )
    result = verbalize_music(m)
    assert "fast (175 BPM)" in result
    assert "C major" in result


def test_verbalize_music_all_none_features():
    m = _music()
    result = verbalize_music(m)
    assert result == 'Music: "Test Track" by Test Artist — '


# ── build_sandwich_text ───────────────────────────────────────────────────────


def test_build_sandwich_text_english_caption_and_speech():
    clip = _clip(
        caption_text="Hello world",
        caption_language="en",
        speech_transcription="Welcome everyone",
        speech_language="en",
    )
    result = build_sandwich_text(clip, {})
    assert result == "Hello world | Welcome everyone"


def test_build_sandwich_text_uses_caption_translation_for_non_english():
    clip = _clip(
        caption_text="Привет мир",
        caption_language="ru",
        caption_translation="Hello world",
        speech_transcription=None,
    )
    result = build_sandwich_text(clip, {})
    assert result == "Hello world"
    assert "Привет" not in result


def test_build_sandwich_text_uses_speech_translation_for_non_english():
    clip = _clip(
        caption_text=None,
        speech_transcription="Bonjour le monde",
        speech_language="fr",
        speech_translation="Hello world",
    )
    result = build_sandwich_text(clip, {})
    assert result == "Hello world"


def test_build_sandwich_text_falls_back_to_raw_if_no_translation():
    clip = _clip(
        caption_text="Привет мир",
        caption_language="ru",
        caption_translation=None,
        speech_transcription=None,
    )
    result = build_sandwich_text(clip, {})
    assert result == "Привет мир"


def test_build_sandwich_text_appends_music():
    m = _music(
        track="Test Track",
        artist="Test Artist",
        energy=0.90,
        valence=0.79,
        acousticness=0.01,
        instrumentalness=0.00,
        danceability=0.52,
        speechiness=0.04,
        tempo=117.0,
        mode=1,
        key=9,
    )
    clip = _clip(caption_text="Great video", caption_language="en", music_id=42)
    result = build_sandwich_text(clip, {42: m})
    assert result is not None
    assert result.startswith("Great video | Music:")
    assert "Test Track" in result
    assert "Test Artist" in result


def test_build_sandwich_text_no_music_when_music_id_none():
    clip = _clip(caption_text="Great video", caption_language="en", music_id=None)
    result = build_sandwich_text(clip, {})
    assert result is not None
    assert "Music:" not in result


def test_build_sandwich_text_returns_none_when_no_text():
    clip = _clip(
        caption_text=None,
        caption_translation=None,
        speech_transcription=None,
        speech_translation=None,
        music_id=None,
    )
    assert build_sandwich_text(clip, {}) is None


def test_build_sandwich_text_skips_empty_strings():
    clip = _clip(
        caption_text="  ",
        caption_language="en",
        speech_transcription="  ",
        speech_language="en",
    )
    assert build_sandwich_text(clip, {}) is None


def test_build_sandwich_text_ignores_speech_when_not_detected():
    """Even if a transcript sits in the row, the sandwich text must not
    include it unless ``is_speech_detected is True``."""
    clip = _clip(
        caption_text="Great video",
        caption_language="en",
        speech_transcription="Thanks for watching!",
        speech_language="en",
        is_speech_detected=False,
    )
    result = build_sandwich_text(clip, {})
    assert result == "Great video"
    assert "Thanks for watching" not in (result or "")


def test_build_sandwich_text_ignores_speech_when_detection_pending():
    """``is_speech_detected is None`` (not yet processed) → speech ignored."""
    clip = _clip(
        caption_text=None,
        speech_transcription="some words",
        speech_language="en",
        is_speech_detected=None,
    )
    result = build_sandwich_text(clip, {})
    assert result is None


# ── build_audio_text ──────────────────────────────────────────────────────────


def test_build_audio_text_music_only():
    m = _music(
        track="Test Track",
        artist="Test Artist",
        energy=0.90,
        valence=0.79,
        acousticness=0.01,
        instrumentalness=0.00,
        danceability=0.52,
        speechiness=0.04,
        tempo=117.0,
        mode=1,
        key=9,
    )
    clip = _clip(music_id=42)
    result = build_audio_text(clip, {42: m})
    assert result is not None
    assert result.startswith('Music: "Test Track" by Test Artist')
    assert "caption" not in result.lower()


def test_build_audio_text_speech_only_english():
    clip = _clip(speech_transcription="Stay focused", speech_language="en")
    result = build_audio_text(clip, {})
    assert result == "Stay focused"


def test_build_audio_text_speech_only_uses_translation():
    clip = _clip(
        speech_transcription="Bonjour le monde",
        speech_language="fr",
        speech_translation="Hello world",
    )
    result = build_audio_text(clip, {})
    assert result == "Hello world"


def test_build_audio_text_speech_falls_back_to_raw_if_no_translation():
    clip = _clip(
        speech_transcription="Bonjour le monde",
        speech_language="fr",
        speech_translation=None,
    )
    result = build_audio_text(clip, {})
    assert result == "Bonjour le monde"


def test_build_audio_text_both_music_and_speech():
    m = _music(
        track="T",
        artist="A",
        energy=0.5,
        valence=0.5,
        acousticness=0.5,
        instrumentalness=0.0,
        danceability=0.5,
        speechiness=0.1,
        tempo=100.0,
        mode=1,
        key=0,
    )
    clip = _clip(speech_transcription="Let's go", speech_language="en", music_id=1)
    result = build_audio_text(clip, {1: m})
    assert result is not None
    parts = result.split(" | ")
    assert len(parts) == 2
    assert parts[0] == "Let's go"
    assert parts[1].startswith('Music: "T" by A')


def test_build_audio_text_no_caption_included():
    m = _music(
        track="T",
        artist="A",
        energy=0.5,
        valence=0.5,
        acousticness=0.5,
        instrumentalness=0.0,
        danceability=0.5,
        speechiness=0.1,
        tempo=100.0,
        mode=1,
        key=0,
    )
    clip = _clip(
        caption_text="Some caption",
        caption_language="en",
        speech_transcription="Hello",
        speech_language="en",
        music_id=1,
    )
    result = build_audio_text(clip, {1: m})
    assert result is not None
    assert "Some caption" not in result


def test_build_audio_text_returns_none_when_neither():
    clip = _clip()
    assert build_audio_text(clip, {}) is None


def test_build_audio_text_skips_empty_speech():
    clip = _clip(speech_transcription="  ", speech_language="en")
    assert build_audio_text(clip, {}) is None


def test_build_audio_text_ignores_missing_music_id():
    clip = _clip(speech_transcription="Hello", speech_language="en", music_id=99)
    result = build_audio_text(clip, {})
    assert result == "Hello"


def test_build_audio_text_speech_language_none_uses_raw():
    clip = _clip(
        speech_transcription="Hello world",
        speech_language=None,
        speech_translation="Something else",
    )
    result = build_audio_text(clip, {})
    assert result == "Hello world"


def test_build_audio_text_ignores_speech_when_not_detected():
    clip = _clip(
        caption_text=None,
        speech_transcription="Thanks for watching!",
        speech_language="en",
        is_speech_detected=False,
    )
    result = build_audio_text(clip, {})
    assert result is None  # no speech and no music → None


def test_build_audio_text_returns_only_music_when_speech_flagged_off():
    m = _music(track="X", artist="Y", energy=0.5)
    clip = _clip(
        caption_text=None,
        speech_transcription="Bye bye.",
        speech_language="en",
        is_speech_detected=False,
        music_id=7,
    )
    result = build_audio_text(clip, {7: m})
    assert result is not None
    assert result.startswith("Music:")
    assert "Bye bye" not in result


# ── build_gemini_text ─────────────────────────────────────────────────────────


def test_build_gemini_text_ignores_speech_when_not_detected():
    clip = _clip(
        caption_text="Caption stays",
        caption_language="en",
        speech_transcription="Subtitles by someone",
        speech_language="en",
        is_speech_detected=False,
    )
    result = build_gemini_text(clip, {})
    assert result == "Caption stays"
    assert "Subtitles" not in (result or "")
