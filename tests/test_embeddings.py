import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from types import SimpleNamespace
from modules.embeddings import verbalize_music, _build_text


def _music(**kwargs):
    defaults = dict(
        artist="Test Artist", track="Test Track",
        energy=None, valence=None, acousticness=None,
        instrumentalness=None, danceability=None,
        speechiness=None, tempo=None, mode=None, key=None,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


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
    m = _music()  # all features are None
    result = verbalize_music(m)
    assert result == 'Music: "Test Track" by Test Artist — '


def _clip(**kwargs):
    defaults = dict(
        caption_text=None, caption_language=None, caption_translation=None,
        speech_transcription=None, speech_language=None, speech_translation=None,
        music_id=None,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


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
