"""Rule-based text construction for embedding cases.

Caption/speech translation rule: use translation iff original language
is not in {"en", None} and a non-empty translation exists; otherwise
fall back to the original text. Music is verbalized from an AudioMIR
row into a compact textual description via ``verbalize_mir``.
"""

from __future__ import annotations

from core.lang import is_english


def _is_non_english(lang: str | None) -> bool:
    """True iff ``lang`` is present and not an English-family tag."""
    return bool(lang) and not is_english(lang)


_MIR_FLAG_FIELDS: tuple[tuple[str, str], ...] = (
    ("is_acoustic", "acoustic"),
    ("is_electronic", "electronic"),
    ("is_instrumental", "instrumental"),
    ("is_happy", "happy"),
    ("is_sad", "sad"),
    ("is_party", "party"),
    ("is_relaxed", "relaxed"),
    ("is_aggressive", "aggressive"),
    ("is_female_voice", "female vocal"),
    ("is_bright_timbre", "bright timbre"),
    ("is_tonal", "tonal"),
)


def _topk_split(value: str | None, k: int) -> list[str]:
    """Split a comma-separated topk_csv-style string and keep the first ``k`` labels."""
    if not value:
        return []
    parts = [p.strip() for p in value.split(",")]
    parts = [p for p in parts if p]
    return parts[:k]


def _bucket_danceability(value: float | None) -> str | None:
    if value is None:
        return None
    if value >= 0.66:
        return "highly danceable"
    if value >= 0.33:
        return "moderately danceable"
    return "low danceability"


def _bucket_engagement(value: float | None) -> str | None:
    if value is None:
        return None
    if value >= 7.0:
        return "highly engaging"
    if value >= 4.0:
        return "moderately engaging"
    return "low engagement"


def _bucket_approachability(value: float | None) -> str | None:
    if value is None:
        return None
    if value >= 7.0:
        return "very approachable"
    if value >= 4.0:
        return "moderately approachable"
    return "low approachability"


def verbalize_mir(mir) -> str:
    """Compact, readable description of an AudioMIR row.

    Shape: ``Music: <top-2 genres> — <top-3 mood/themes>, <top-2 instruments>;
    flags: <true-only flags>; <danceability bucket>, <engagement bucket>,
    <approachability bucket>``.

    Empty sections are skipped. Returns ``""`` only when every section is
    empty — defensive; should not occur when ``mir.is_music_detected`` is
    True for a successful MIR row.
    """
    genres = _topk_split(mir.genre_labels, 2)
    moods = _topk_split(mir.moodtheme_labels, 3)
    instruments = _topk_split(mir.instrument_labels, 2)

    head_parts: list[str] = []
    if genres:
        head_parts.append(", ".join(genres))
    tail_parts: list[str] = []
    if moods:
        tail_parts.append(", ".join(moods))
    if instruments:
        tail_parts.append(", ".join(instruments))

    sections: list[str] = []
    if head_parts and tail_parts:
        sections.append(f"{head_parts[0]} — {', '.join(tail_parts)}")
    elif head_parts:
        sections.append(head_parts[0])
    elif tail_parts:
        sections.append(", ".join(tail_parts))

    flags = [label for attr, label in _MIR_FLAG_FIELDS if getattr(mir, attr) is True]
    if flags:
        sections.append("flags: " + ", ".join(flags))

    buckets: list[str] = []
    for fn, value in (
        (_bucket_danceability, mir.danceability),
        (_bucket_engagement, mir.engagement),
        (_bucket_approachability, mir.approachability),
    ):
        bucket = fn(value)
        if bucket is not None:
            buckets.append(bucket)
    if buckets:
        sections.append(", ".join(buckets))

    if not sections:
        return ""
    return "Music: " + "; ".join(sections)


def build_sandwich_text(clip, mir_row) -> str | None:
    """Caption + speech + MIR music text, joined by '` | `'.

    ``mir_row`` is the matching ``AudioMIR`` row for the clip (or ``None``).
    The music block is included iff ``mir_row is not None and
    mir_row.is_music_detected is True``. The speech block is included iff
    ``clip.is_speech_detected is True``.
    """
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

    if clip.is_speech_detected is True:
        speech = (
            clip.speech_translation
            if _is_non_english(clip.speech_language)
            and clip.speech_translation
            and clip.speech_translation.strip()
            else (clip.speech_transcription or "")
        )
        if speech.strip():
            parts.append(speech.strip())

    if mir_row is not None and mir_row.is_music_detected is True:
        music_text = verbalize_mir(mir_row)
        if music_text:
            parts.append(music_text)

    return " | ".join(parts) if parts else None


def build_gemini_text(clip, _mir_row) -> str | None:
    """Caption + transcript for the gemini case.

    Music is NOT verbalized — the model gets the raw audio track
    separately. ``_mir_row`` is accepted to keep the text_builder
    signature uniform with the other cases and is intentionally ignored.
    Returns ``None`` when both caption and transcript are empty.
    """
    cap = (
        clip.caption_translation
        if _is_non_english(clip.caption_language)
        and clip.caption_translation
        and clip.caption_translation.strip()
        else (clip.caption_clean or clip.caption_text or "")
    )
    speech = ""
    if clip.is_speech_detected is True:
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


def build_audio_text(clip, mir_row) -> str | None:
    """Speech + MIR music text, joined by '` | `'. Captions excluded.

    Order: speech first, music second — matches the audio embedding
    instruction priority.
    """
    parts = []

    if clip.is_speech_detected is True:
        speech = (
            clip.speech_translation
            if _is_non_english(clip.speech_language)
            and clip.speech_translation
            and clip.speech_translation.strip()
            else (clip.speech_transcription or "")
        )
        if speech.strip():
            parts.append(speech.strip())

    if mir_row is not None and mir_row.is_music_detected is True:
        music_text = verbalize_mir(mir_row)
        if music_text:
            parts.append(music_text)

    return " | ".join(parts) if parts else None


def build_spoken_text(clip, _mir_row) -> str | None:
    """Speech transcript ONLY (no music, no caption).

    Same translation rule as the speech half of ``build_sandwich_text``:
    use ``speech_translation`` when the source language is non-English and a
    non-empty translation exists, else ``speech_transcription``. ``_mir_row``
    is accepted to keep the text_builder signature uniform and is ignored.
    Returns ``None`` when no speech was detected or the transcript is empty.
    """
    if clip.is_speech_detected is not True:
        return None
    speech = (
        clip.speech_translation
        if _is_non_english(clip.speech_language)
        and clip.speech_translation
        and clip.speech_translation.strip()
        else (clip.speech_transcription or "")
    )
    speech = speech.strip()
    return speech if speech else None


def build_textual_text(clip, _mir_row) -> str | None:
    """Clip caption ONLY (no speech, no music).

    Same caption translation rule as ``build_sandwich_text``. ``_mir_row`` is
    accepted to keep the signature uniform and is ignored. Returns ``None``
    when the caption is empty.
    """
    cap = (
        clip.caption_translation
        if _is_non_english(clip.caption_language)
        and clip.caption_translation
        and clip.caption_translation.strip()
        else (clip.caption_clean or clip.caption_text or "")
    )
    cap = cap.strip()
    return cap if cap else None
