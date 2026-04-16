import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import numpy as np
from types import SimpleNamespace
from modules.embeddings import verbalize_music, _build_text, _build_audio_text, _bytes_to_array, _aggregate_user_embeddings


# ── helpers ──────────────────────────────────────────────────────────────────

def _music(**kwargs):
    defaults = dict(
        artist="Test Artist", track="Test Track",
        energy=None, valence=None, acousticness=None,
        instrumentalness=None, danceability=None,
        speechiness=None, tempo=None, mode=None, key=None,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _clip(**kwargs):
    defaults = dict(
        caption_text=None, caption_language=None, caption_translation=None,
        speech_transcription=None, speech_language=None, speech_translation=None,
        music_id=None,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _make_blob(values: list[float]) -> bytes:
    return np.array(values, dtype=np.float32).tobytes()


# ── verbalize_music ───────────────────────────────────────────────────────────

def test_verbalize_music_prefix():
    m = _music(track="Laura Palmer's Theme", artist="Angelo Badalamenti",
               energy=0.01, valence=0.04, acousticness=0.84,
               instrumentalness=0.91, danceability=0.06,
               speechiness=0.03, tempo=68.0, mode=0, key=3)
    result = verbalize_music(m)
    assert result.startswith('Music: "Laura Palmer\'s Theme" by Angelo Badalamenti — ')


def test_verbalize_music_low_energy_melancholic_acoustic_instrumental():
    m = _music(energy=0.01, valence=0.04, acousticness=0.84,
               instrumentalness=0.91, danceability=0.06,
               speechiness=0.03, tempo=68.0, mode=0, key=3)
    result = verbalize_music(m)
    assert "low energy" in result
    assert "dark and melancholic" in result
    assert "acoustic" in result
    assert "instrumental" in result
    assert "not danceable" in result
    assert "very slow (68 BPM)" in result
    assert "D# minor" in result


def test_verbalize_music_high_energy_upbeat_electronic_vocal():
    m = _music(energy=0.90, valence=0.79, acousticness=0.01,
               instrumentalness=0.00, danceability=0.52,
               speechiness=0.04, tempo=117.0, mode=1, key=9)
    result = verbalize_music(m)
    assert "very high energy" in result
    assert "very upbeat" in result
    assert "electronic" in result
    assert "vocal" in result
    assert "moderate tempo (117 BPM)" in result
    assert "A major" in result
    assert "danceable" not in result


def test_verbalize_music_no_key_mode():
    m = _music(energy=0.37, valence=0.18, acousticness=0.55,
               instrumentalness=0.84, danceability=0.24,
               speechiness=0.03, tempo=66.0, mode=None, key=None)
    result = verbalize_music(m)
    assert "minor" not in result
    assert "major" not in result
    assert "instrumental" in result
    assert "very slow (66 BPM)" in result


def test_verbalize_music_speechy():
    m = _music(energy=0.50, valence=0.40, acousticness=0.30,
               instrumentalness=0.00, danceability=0.50,
               speechiness=0.70, tempo=95.0, mode=None, key=None)
    result = verbalize_music(m)
    assert "spoken word" in result


def test_verbalize_music_rap():
    m = _music(energy=0.70, valence=0.50, acousticness=0.10,
               instrumentalness=0.00, danceability=0.80,
               speechiness=0.45, tempo=95.0, mode=None, key=None)
    result = verbalize_music(m)
    assert "rap or speech-heavy" in result
    assert "highly danceable" in result


def test_verbalize_music_fast_tempo():
    m = _music(energy=0.85, valence=0.60, acousticness=0.05,
               instrumentalness=0.00, danceability=0.70,
               speechiness=0.05, tempo=175.0, mode=1, key=0)
    result = verbalize_music(m)
    assert "fast (175 BPM)" in result
    assert "C major" in result


def test_verbalize_music_all_none_features():
    m = _music()
    result = verbalize_music(m)
    assert result == 'Music: "Test Track" by Test Artist — '


# ── _build_text ───────────────────────────────────────────────────────────────

def test_build_text_english_caption_and_speech():
    clip = _clip(caption_text="Hello world", caption_language="en",
                 speech_transcription="Welcome everyone", speech_language="en")
    result = _build_text(clip, {})
    assert result == "Hello world | Welcome everyone"


def test_build_text_uses_caption_translation_for_non_english():
    clip = _clip(caption_text="Привет мир", caption_language="ru",
                 caption_translation="Hello world",
                 speech_transcription=None)
    result = _build_text(clip, {})
    assert result == "Hello world"
    assert "Привет" not in result


def test_build_text_uses_speech_translation_for_non_english():
    clip = _clip(caption_text=None,
                 speech_transcription="Bonjour le monde", speech_language="fr",
                 speech_translation="Hello world")
    result = _build_text(clip, {})
    assert result == "Hello world"


def test_build_text_falls_back_to_raw_if_no_translation():
    clip = _clip(caption_text="Привет мир", caption_language="ru",
                 caption_translation=None,
                 speech_transcription=None)
    result = _build_text(clip, {})
    assert result == "Привет мир"


def test_build_text_appends_music():
    m = _music(track="Test Track", artist="Test Artist",
               energy=0.90, valence=0.79, acousticness=0.01,
               instrumentalness=0.00, danceability=0.52,
               speechiness=0.04, tempo=117.0, mode=1, key=9)
    clip = _clip(caption_text="Great video", caption_language="en", music_id=42)
    result = _build_text(clip, {42: m})
    assert result.startswith("Great video | Music:")
    assert "Test Track" in result
    assert "Test Artist" in result


def test_build_text_no_music_when_music_id_none():
    clip = _clip(caption_text="Great video", caption_language="en", music_id=None)
    result = _build_text(clip, {})
    assert "Music:" not in result


def test_build_text_returns_none_when_no_text():
    clip = _clip(caption_text=None, caption_translation=None,
                 speech_transcription=None, speech_translation=None,
                 music_id=None)
    assert _build_text(clip, {}) is None


def test_build_text_skips_empty_strings():
    clip = _clip(caption_text="  ", caption_language="en",
                 speech_transcription="  ", speech_language="en")
    assert _build_text(clip, {}) is None


# ── _bytes_to_array ───────────────────────────────────────────────────────────

def test_bytes_to_array_roundtrip():
    arr = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    blob = arr.tobytes()
    result = _bytes_to_array(blob)
    np.testing.assert_array_almost_equal(result, arr)


def test_bytes_to_array_dtype_is_float32():
    arr = np.array([1.0, 2.0], dtype=np.float32)
    result = _bytes_to_array(arr.tobytes())
    assert result.dtype == np.float32


def test_bytes_to_array_returns_copy():
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    blob = arr.tobytes()
    result = _bytes_to_array(blob)
    result[0] = 99.0
    result2 = _bytes_to_array(blob)
    assert result2[0] == pytest.approx(1.0)


# ── _aggregate_user_embeddings ────────────────────────────────────────────────

def test_aggregate_single_clip_per_user():
    rows = [
        (_make_blob([1.0, 2.0, 3.0]), 101),
        (_make_blob([4.0, 5.0, 6.0]), 102),
    ]
    result = _aggregate_user_embeddings(rows)
    assert set(result.keys()) == {101, 102}
    np.testing.assert_array_almost_equal(_bytes_to_array(result[101]), [1.0, 2.0, 3.0])
    np.testing.assert_array_almost_equal(_bytes_to_array(result[102]), [4.0, 5.0, 6.0])


def test_aggregate_mean_of_multiple_clips():
    rows = [
        (_make_blob([1.0, 3.0]), 101),
        (_make_blob([3.0, 1.0]), 101),
        (_make_blob([0.0, 0.0]), 101),
    ]
    result = _aggregate_user_embeddings(rows)
    np.testing.assert_array_almost_equal(
        _bytes_to_array(result[101]), [4.0 / 3.0, 4.0 / 3.0]
    )


def test_aggregate_output_dtype_is_float32():
    rows = [(_make_blob([1.0, 2.0]), 101)]
    result = _aggregate_user_embeddings(rows)
    assert _bytes_to_array(result[101]).dtype == np.float32


def test_aggregate_empty_rows_returns_empty_dict():
    assert _aggregate_user_embeddings([]) == {}


def test_aggregate_mixed_users():
    rows = [
        (_make_blob([2.0, 4.0]), 1),
        (_make_blob([0.0, 0.0]), 1),
        (_make_blob([10.0, 10.0]), 2),
    ]
    result = _aggregate_user_embeddings(rows)
    assert set(result.keys()) == {1, 2}
    np.testing.assert_array_almost_equal(_bytes_to_array(result[1]), [1.0, 2.0])
    np.testing.assert_array_almost_equal(_bytes_to_array(result[2]), [10.0, 10.0])


# ── _build_audio_text ─────────────────────────────────────────────────────────

def test_build_audio_text_music_only():
    m = _music(track="Test Track", artist="Test Artist",
               energy=0.90, valence=0.79, acousticness=0.01,
               instrumentalness=0.00, danceability=0.52,
               speechiness=0.04, tempo=117.0, mode=1, key=9)
    clip = _clip(music_id=42)
    result = _build_audio_text(clip, {42: m})
    assert result is not None
    assert result.startswith('Music: "Test Track" by Test Artist')
    assert "caption" not in result.lower()


def test_build_audio_text_speech_only_english():
    clip = _clip(speech_transcription="Stay focused", speech_language="en")
    result = _build_audio_text(clip, {})
    assert result == "Stay focused"


def test_build_audio_text_speech_only_uses_translation():
    clip = _clip(speech_transcription="Bonjour le monde", speech_language="fr",
                 speech_translation="Hello world")
    result = _build_audio_text(clip, {})
    assert result == "Hello world"


def test_build_audio_text_speech_falls_back_to_raw_if_no_translation():
    clip = _clip(speech_transcription="Bonjour le monde", speech_language="fr",
                 speech_translation=None)
    result = _build_audio_text(clip, {})
    assert result == "Bonjour le monde"


def test_build_audio_text_both_music_and_speech():
    m = _music(track="T", artist="A", energy=0.5, valence=0.5,
               acousticness=0.5, instrumentalness=0.0,
               danceability=0.5, speechiness=0.1, tempo=100.0, mode=1, key=0)
    clip = _clip(speech_transcription="Let's go", speech_language="en", music_id=1)
    result = _build_audio_text(clip, {1: m})
    assert result is not None
    parts = result.split(" | ")
    assert len(parts) == 2
    assert parts[0] == "Let's go"
    assert parts[1].startswith('Music: "T" by A')


def test_build_audio_text_no_caption_included():
    """Captions must never appear in audio text even when present on the clip."""
    m = _music(track="T", artist="A", energy=0.5, valence=0.5,
               acousticness=0.5, instrumentalness=0.0,
               danceability=0.5, speechiness=0.1, tempo=100.0, mode=1, key=0)
    clip = _clip(caption_text="Some caption", caption_language="en",
                 speech_transcription="Hello", speech_language="en", music_id=1)
    result = _build_audio_text(clip, {1: m})
    assert "Some caption" not in result


def test_build_audio_text_returns_none_when_neither():
    clip = _clip()
    assert _build_audio_text(clip, {}) is None


def test_build_audio_text_skips_empty_speech():
    clip = _clip(speech_transcription="  ", speech_language="en")
    assert _build_audio_text(clip, {}) is None


def test_build_audio_text_ignores_missing_music_id():
    clip = _clip(speech_transcription="Hello", speech_language="en", music_id=99)
    result = _build_audio_text(clip, {})   # music_id 99 not in map
    assert result == "Hello"
