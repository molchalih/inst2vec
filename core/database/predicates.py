"""Reusable SQL clause-builder helpers for Clip queries.

Each helper returns a tuple of SQLAlchemy clause elements consumed by
`query.filter(*…)`. No engine, no session, no side effects.
"""

from sqlalchemy import or_
from sqlalchemy.sql import func

from core.database.models import Clip, ClipLabel
from core.lang import sql_is_english


def clip_used_in_analysis():
    """Canonical filter: clips that should drive downstream computation.

    Returns a tuple of clauses for `query.filter(*clip_used_in_analysis())`.
    """
    return (
        Clip.is_selected.is_(True),
        Clip.is_downloaded.is_(True),
    )


def clip_needs_speech_detection():
    """Clips eligible for Whisper transcription: selected, downloaded, unresolved."""
    return (
        *clip_used_in_analysis(),
        Clip.is_speech_detected.is_(None),
    )


def clip_has_detected_speech():
    """Clips that Whisper marked as containing meaningful speech."""
    return (
        *clip_used_in_analysis(),
        Clip.is_speech_detected.is_(True),
    )


def clip_needs_speech_translation():
    """Clips with detected non-English speech that still lack a translation."""
    return (
        *clip_has_detected_speech(),
        Clip.speech_transcription.is_not(None),
        Clip.speech_transcription != "",
        Clip.speech_language.is_not(None),
        Clip.speech_language != "",
        ~sql_is_english(Clip.speech_language),
        (Clip.speech_translation.is_(None)) | (Clip.speech_translation == ""),
    )


def has_raw_caption():
    """Clips that have any non-empty raw scraped caption."""
    return (
        Clip.caption_text.is_not(None),
        func.trim(Clip.caption_text) != "",
    )


def has_clean_caption():
    """Clips that already have a non-empty normalized caption."""
    return (
        Clip.caption_clean.is_not(None),
        func.trim(Clip.caption_clean) != "",
    )


def needs_caption_cleaning():
    """Selected, downloaded clips with raw text but no caption_clean yet."""
    return (
        *clip_used_in_analysis(),
        Clip.caption_text.is_not(None),
        func.trim(Clip.caption_text) != "",
        Clip.caption_clean.is_(None),
    )


def needs_caption_language_detection():
    """Selected, downloaded clips with caption_clean and no language tag."""
    return (
        *clip_used_in_analysis(),
        Clip.caption_clean.is_not(None),
        func.trim(Clip.caption_clean) != "",
        (Clip.caption_language.is_(None)) | (Clip.caption_language == ""),
    )


def needs_caption_translation():
    """Selected, downloaded clips with detected non-English clean caption and no translation.

    Rows sealed as ``"und"`` (Lingua returned no language) are terminally
    classified and excluded — there's no source language to translate from.
    """
    return (
        *clip_used_in_analysis(),
        Clip.caption_clean.is_not(None),
        func.trim(Clip.caption_clean) != "",
        Clip.caption_language.is_not(None),
        Clip.caption_language != "",
        Clip.caption_language != "und",
        ~sql_is_english(Clip.caption_language),
        (Clip.caption_translation.is_(None)) | (Clip.caption_translation == ""),
    )


def clip_needs_label():
    """Selected, downloaded clips with no successful label row yet.

    Caller must `.outerjoin(ClipLabel, ClipLabel.clip_id == Clip.id)` before
    applying these clauses.
    """
    return (
        *clip_used_in_analysis(),
        or_(ClipLabel.clip_id.is_(None), ClipLabel.status == "pending"),
    )


def clip_label_done():
    """Selected, downloaded clips with a successful ``ClipLabel`` row.

    Caller must join ``ClipLabel`` on ``ClipLabel.clip_id == Clip.id``.
    """
    return (*clip_used_in_analysis(), ClipLabel.status == "success")
