"""Shared state constants and text helpers for the speech package."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path

from sqlalchemy.orm import Session

from core.config import SpeechSettings
from core.database import Clip
from core.pipeline import Stage

SCOPE_CLASSIFY: str = "classify_speech"
SCOPE_TRANSLATE: str = "translate_speech"
SCOPE_CLEAN: str = "clean_speech"

STAGE_SPEECH: Stage = Stage.SPEECH
SCOPE_SPEECH: str = "all"

HALLUCINATIONS_DIR = Path(__file__).parent / "hallucinations"
_EXTRAS_FILENAME = "extras.txt"
_HALLUCINATION_MIN_PHRASE_CHARS = 5  # drop entries shorter than this from corpora
# Substring matching is reserved for phrases long enough to be unambiguous bot
# output. Common short outros like "Thank you." or "Thanks for watching!" can
# legitimately appear inside real transcripts, so they stay exact-match-only.
# Project-curated extras (extras.txt) bypass this floor — entries there are
# vetted as unique-to-hallucinations regardless of length.
_HALLUCINATION_SUBSTRING_MIN_CHARS = 40


def _load_hallucination_phrases(
    files: Iterable[Path], min_phrase_chars: int
) -> list[str]:
    """Read phrases from each ``files`` path.

    Skips blank lines and lines whose first non-whitespace character is
    ``#``. Trims surrounding whitespace, drops entries shorter than
    ``min_phrase_chars``, and deduplicates while preserving order.
    """
    seen: set[str] = set()
    out: list[str] = []
    for path in files:
        if not path.is_file():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            phrase = raw.strip()
            if not phrase or phrase.startswith("#"):
                continue
            if len(phrase) < min_phrase_chars:
                continue
            if phrase in seen:
                continue
            seen.add(phrase)
            out.append(phrase)
    return out


HALLUCINATION_PHRASES: list[str] = _load_hallucination_phrases(
    sorted(HALLUCINATIONS_DIR.glob("*.txt")),
    min_phrase_chars=_HALLUCINATION_MIN_PHRASE_CHARS,
)

_EXTRAS_PHRASES: list[str] = _load_hallucination_phrases(
    [HALLUCINATIONS_DIR / _EXTRAS_FILENAME],
    min_phrase_chars=_HALLUCINATION_MIN_PHRASE_CHARS,
)
_EXTRAS_PHRASE_SET: frozenset[str] = frozenset(_EXTRAS_PHRASES)

# Substring-eligible subset: long-enough generic phrases plus all
# project-curated extras. Used by both the Python matcher and the SQL
# filter so neither path substring-scans short common phrases.
_HALLUCINATION_SUBSTRING_PHRASES: list[str] = [
    p
    for p in HALLUCINATION_PHRASES
    if len(p) >= _HALLUCINATION_SUBSTRING_MIN_CHARS or p in _EXTRAS_PHRASE_SET
]

# Raw-phrase view used by SQL filters (e.g. clean_speech's case-sensitive
# Translation.contains() scan). Restricted to substring-safe phrases so the
# SQL pass cannot nuke a legitimate translation that merely embeds a short
# common corpus entry like "Yeah." or "Thank you.". Python-side callers
# should use has_hallucination_marker() instead.
HALLUCINATION_MARKERS: list[str] = _HALLUCINATION_SUBSTRING_PHRASES

# Cyrillic basic ranges (А-Я, а-я) are subsumed by the full block Ѐ-ӿ (U+0400-04FF).
_NON_LETTER_RE = re.compile(r"[^A-Za-zÀ-ɏЀ-ӿ]+")
_REPEAT_MIN_TOKENS = 5  # at least N tokens of the same lowercased word in a row


def has_meaningful_speech_text(text: str | None, min_meaningful_chars: int) -> bool:
    """True iff text contains at least ``min_meaningful_chars`` letters after
    stripping punctuation, digits and whitespace."""
    if not text:
        return False
    cleaned = _NON_LETTER_RE.sub("", text.strip())
    return len(cleaned) >= min_meaningful_chars


def is_too_short(text: str | None, min_chars: int) -> bool:
    """True iff ``text`` stripped of surrounding whitespace has fewer
    than ``min_chars`` characters.

    Counts punctuation and digits (unlike ``has_meaningful_speech_text``,
    which counts letters only). Catches sub-token Whisper outputs such
    as ``"you"`` or ``"."`` regardless of their composition.
    """
    if not text:
        return True
    return len(text.strip()) < min_chars


def has_low_letter_ratio(text: str | None, min_ratio: float) -> bool:
    """True iff the alphabetic-letter fraction of ``text`` is below
    ``min_ratio``.

    Letters use the same Latin/Cyrillic class as
    ``has_meaningful_speech_text``. Catches outputs dominated by
    punctuation, digits, or non-letter glyphs.
    """
    if not text:
        return True
    stripped = text.strip()
    if not stripped:
        return True
    letters = len(_NON_LETTER_RE.sub("", stripped))
    total_non_ws = sum(1 for ch in stripped if not ch.isspace())
    if total_non_ws == 0:
        return True
    return (letters / total_non_ws) < min_ratio


def _normalize_for_hallucination(text: str) -> str:
    """Lowercase, strip, collapse internal whitespace."""
    return re.sub(r"\s+", " ", text.strip().lower())


_NORMALIZED_HALLUCINATION_PHRASES: list[str] = [
    p for p in (_normalize_for_hallucination(raw) for raw in HALLUCINATION_PHRASES) if p
]
_NORMALIZED_SUBSTRING_PHRASES: list[str] = [
    p
    for p in (
        _normalize_for_hallucination(raw) for raw in _HALLUCINATION_SUBSTRING_PHRASES
    )
    if p
]


def has_hallucination_marker(text: str | None) -> bool:
    """True iff ``text`` matches a known hallucination phrase.

    Matching is case-insensitive and whitespace-normalized. Two modes:
    exact whole-text match against every corpus phrase (catches short,
    single-sentence outputs like ``"Bye."`` without firing on real
    speech that merely contains the word); substring match against the
    restricted substring-safe subset (long-enough generic phrases plus
    project-curated extras) so common everyday phrases embedded in
    longer real transcripts do not false-positive.
    """
    if not text:
        return False
    n = _normalize_for_hallucination(text)
    if not n:
        return False
    if any(n == p for p in _NORMALIZED_HALLUCINATION_PHRASES):
        return True
    return any(p in n for p in _NORMALIZED_SUBSTRING_PHRASES)


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


# Fields whose values can change speech outputs. Purely-operational
# knobs (commit_every, vad_ffmpeg_timeout_s) are intentionally excluded
# so a value change does not invalidate the stored fingerprint.
_SPEECH_CONFIG_FIELDS: tuple[str, ...] = (
    "whisper_model",
    "translate_model",
    "translate_target_lang",
    "translation_max_chars",
    "translate_max_new_tokens",
    "logprob_threshold",
    "compression_threshold",
    "min_meaningful_chars",
    "dirty_min_chars",
    "dirty_min_letter_ratio",
    "vad_enabled",
    "vad_sampling_rate",
    "vad_threshold",
    "vad_min_speech_ms",
    "vad_min_silence_ms",
    "vad_speech_pad_ms",
    "vad_min_total_speech_s",
)


def speech_config_payload(cfg: SpeechSettings) -> str:
    """Stable JSON of the SpeechSettings fields that affect speech outputs."""
    payload = {f: getattr(cfg, f) for f in _SPEECH_CONFIG_FIELDS}
    return json.dumps(payload, sort_keys=True, default=str)
