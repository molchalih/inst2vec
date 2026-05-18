"""Rule-based text construction for embedding cases.

Caption/speech translation rule: use translation iff original language
is not in {"en", None} and a non-empty translation exists; otherwise
fall back to the original text. Music is verbalized into a compact
textual description.
"""

from __future__ import annotations

_KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _is_non_english(lang: str | None) -> bool:
    """True iff ``lang`` is present and not an English-family tag.

    Canonical English check across the codebase. SQL equivalent:
    ``func.lower(col).notlike("en%")``. Matches "en", "EN", "eng",
    "en-US", "English" all as English.
    """
    return bool(lang) and not lang.lower().startswith("en")


def verbalize_music(music) -> str:
    descriptors = []

    if music.energy is not None:
        if music.energy >= 0.80:
            descriptors.append("very high energy")
        elif music.energy >= 0.55:
            descriptors.append("high energy")
        elif music.energy >= 0.30:
            descriptors.append("moderate energy")
        else:
            descriptors.append("low energy")

    if music.valence is not None:
        if music.valence >= 0.75:
            descriptors.append("very upbeat")
        elif music.valence >= 0.50:
            descriptors.append("positive")
        elif music.valence >= 0.25:
            descriptors.append("bittersweet")
        else:
            descriptors.append("dark and melancholic")

    if music.acousticness is not None:
        if music.acousticness >= 0.75:
            descriptors.append("acoustic")
        elif music.acousticness <= 0.20:
            descriptors.append("electronic")

    if music.instrumentalness is not None:
        if music.instrumentalness >= 0.50:
            descriptors.append("instrumental")
        else:
            descriptors.append("vocal")

    if music.danceability is not None:
        if music.danceability >= 0.75:
            descriptors.append("highly danceable")
        elif music.danceability <= 0.25:
            descriptors.append("not danceable")

    if music.speechiness is not None and music.speechiness >= 0.33:
        if music.speechiness >= 0.66:
            descriptors.append("spoken word")
        else:
            descriptors.append("rap or speech-heavy")

    if music.tempo is not None:
        bpm = round(music.tempo)
        if music.tempo >= 150:
            descriptors.append(f"fast ({bpm} BPM)")
        elif music.tempo >= 110:
            descriptors.append(f"moderate tempo ({bpm} BPM)")
        elif music.tempo >= 75:
            descriptors.append(f"slow ({bpm} BPM)")
        else:
            descriptors.append(f"very slow ({bpm} BPM)")

    if music.mode is not None and music.key is not None and 0 <= int(music.key) <= 11:
        mode_str = "major" if music.mode == 1 else "minor"
        key_str = _KEY_NAMES[int(music.key)]
        descriptors.append(f"{key_str} {mode_str}")

    desc = ", ".join(descriptors)
    track = music.track or "Unknown Track"
    artist = music.artist or "Unknown Artist"
    return f'Music: "{track}" by {artist} — {desc}'


def build_sandwich_text(clip, music_map: dict) -> str | None:
    parts = []

    cap = (
        clip.caption_translation
        if _is_non_english(clip.caption_language)
        and clip.caption_translation
        and clip.caption_translation.strip()
        else (clip.caption_clean or clip.caption_text or "")
    )
    if cap.strip():
        parts.append(cap.strip())

    speech = (
        clip.speech_translation
        if _is_non_english(clip.speech_language)
        and clip.speech_translation
        and clip.speech_translation.strip()
        else (clip.speech_transcription or "")
    )
    if speech.strip():
        parts.append(speech.strip())

    if clip.music_id is not None and clip.music_id in music_map:
        parts.append(verbalize_music(music_map[clip.music_id]))

    return " | ".join(parts) if parts else None


def build_gemini_text(clip, _music_map: dict) -> str | None:
    """Caption + transcript for the gemini case.

    Uses translation when source language is non-English and a non-empty
    translation exists; otherwise the cleaned/original text. Music is
    NOT verbalized — the model gets the raw audio track separately.
    Returns ``None`` when both caption and transcript are empty.
    """
    cap = (
        clip.caption_translation
        if _is_non_english(clip.caption_language)
        and clip.caption_translation
        and clip.caption_translation.strip()
        else (clip.caption_clean or clip.caption_text or "")
    )
    speech = (
        clip.speech_translation
        if _is_non_english(clip.speech_language)
        and clip.speech_translation
        and clip.speech_translation.strip()
        else (clip.speech_transcription or "")
    )

    parts = []
    if cap and cap.strip():
        parts.append(cap.strip())
    if speech and speech.strip():
        parts.append(speech.strip())
    if not parts:
        return None
    return "\n\n---\n\n".join(parts)


def build_audio_text(clip, music_map: dict) -> str | None:
    # Order: speech first, music second — matches the audio embedding
    # instruction priority. Captions are deliberately excluded.
    parts = []

    speech = (
        clip.speech_translation
        if _is_non_english(clip.speech_language)
        and clip.speech_translation
        and clip.speech_translation.strip()
        else (clip.speech_transcription or "")
    )
    if speech.strip():
        parts.append(speech.strip())

    if clip.music_id is not None and clip.music_id in music_map:
        parts.append(verbalize_music(music_map[clip.music_id]))

    return " | ".join(parts) if parts else None
