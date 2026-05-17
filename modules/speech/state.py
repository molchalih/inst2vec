"""Shared state constants and text helpers for the speech package."""

from __future__ import annotations

import re

from sqlalchemy.orm import Session

from core.database import Clip

SCOPE_CLASSIFY: str = "classify_speech"
SCOPE_TRANSLATE: str = "translate_speech"
SCOPE_CLEAN: str = "clean_speech"

STAGE_SPEECH: str = "speech"
SCOPE_SPEECH: str = "all"

# Substrings that indicate a hallucination / non-speech transcription.
# Clips whose translation/transcription contains one are reclassified as no-speech.
HALLUCINATION_MARKERS: list[str] = [
    "DimaTorzok",
]

_NON_LETTER_RE = re.compile(r"[^A-Za-zА-Яа-яÀ-ɏЀ-ӿ]+")
_REPEAT_MIN_TOKENS = 5  # at least N tokens of the same lowercased word in a row


def has_meaningful_speech_text(text: str | None, min_meaningful_chars: int) -> bool:
    """True iff text contains at least ``min_meaningful_chars`` letters after
    stripping punctuation, digits and whitespace."""
    if not text:
        return False
    cleaned = _NON_LETTER_RE.sub("", text.strip())
    return len(cleaned) >= min_meaningful_chars


def has_hallucination_marker(text: str | None) -> bool:
    """True iff text contains any known hallucination marker substring."""
    if not text:
        return False
    return any(marker in text for marker in HALLUCINATION_MARKERS)


def is_repeated_output(text: str | None) -> bool:
    """True iff text consists of ``_REPEAT_MIN_TOKENS`` or more consecutive
    identical (case-insensitive) tokens — a classic Whisper looping artifact."""
    if not text:
        return False
    tokens = [t for t in re.findall(r"\w+", text.lower()) if t]
    if len(tokens) < _REPEAT_MIN_TOKENS:
        return False
    run = 1
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i - 1]:
            run += 1
            if run >= _REPEAT_MIN_TOKENS:
                return True
        else:
            run = 1
    return False


def reset_speech_outputs(session: Session) -> None:
    """NULL all seven speech output columns on every clip.

    Resets the four content columns (is_speech_detected,
    speech_transcription, speech_language, speech_translation) and
    the three Whisper metric columns (speech_confidence,
    speech_avg_logprob, speech_compression_ratio) atomically.
    Called from process_speech() when the speech config fingerprint
    drifts. Resets all clips (not just currently-eligible ones) so
    clips that re-enter the selection pool on a later run can't carry
    stale derived fields produced under the previous config. The
    row-level idempotence inside classify/translate/clean refills the
    cleared columns for eligible clips on the next pass.
    """
    session.query(Clip).update(
        {
            Clip.is_speech_detected: None,
            Clip.speech_transcription: None,
            Clip.speech_language: None,
            Clip.speech_translation: None,
            Clip.speech_confidence: None,
            Clip.speech_avg_logprob: None,
            Clip.speech_compression_ratio: None,
        },
        synchronize_session=False,
    )
    session.commit()
