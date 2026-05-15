"""Whisper transcription stage: classify clips as speech / no-speech."""

from __future__ import annotations

import os

import whisper
from sqlalchemy import or_

from modules.console import log, progress
from modules.database import (
    Clip,
    clip_has_detected_speech,
    clip_needs_speech_detection,
    get_session,
)
from modules.speech.state import (
    HALLUCINATION_MARKERS,
    SCOPE_CLASSIFY,
    SCOPE_CLEAN,
    has_hallucination_marker,
    has_meaningful_speech_text,
    is_repeated_output,
)


def _transcribe(model, path: str) -> tuple[str, str, float, float, float]:
    """Return (text, language, speech_confidence, avg_logprob, compression_ratio).

    Whisper is called with ``condition_on_previous_text=False`` and ``beam_size=1``
    to suppress two common hallucination patterns: long-context drift and
    beam-search loop collapse.

    All float metrics are 0.0 when Whisper produces no segments.
    """
    result = model.transcribe(path, condition_on_previous_text=False, beam_size=1)
    text = (result.get("text") or "").strip()
    language = result.get("language") or ""
    segs = result.get("segments") or []
    if segs:
        mean_no_speech = sum(s.get("no_speech_prob", 0.0) for s in segs) / len(segs)
        confidence = max(0.0, min(1.0, 1.0 - mean_no_speech))
        avg_logprob = sum(s.get("avg_logprob", 0.0) for s in segs) / len(segs)
        compression_ratio = sum(s.get("compression_ratio", 0.0) for s in segs) / len(
            segs
        )
    else:
        confidence = avg_logprob = compression_ratio = 0.0
    return text, language, confidence, avg_logprob, compression_ratio


def classify_speech(
    video_dir: str,
    whisper_model: str,
    commit_every: int,
    logprob_threshold: float,
    compression_threshold: float,
    min_meaningful_chars: int,
) -> None:
    """Transcribe all unresolved clips with Whisper.

    Decision matrix per clip:
        meaningful AND clean   → is_speech_detected = True
        meaningful BUT dirty   → is_speech_detected = False
        not meaningful         → is_speech_detected = False
        missing file           → leave NULL (retryable)
        Whisper exception      → leave NULL (retryable)
    """
    session = get_session()
    clips = (
        session.query(Clip)
        .filter(*clip_needs_speech_detection())
        .order_by(Clip.id.desc())
        .all()
    )
    if not clips:
        session.close()
        return

    log(SCOPE_CLASSIFY, f"{len(clips)} clips to transcribe")
    model: whisper.Whisper | None = None
    detected = no_speech = missing = errored = 0

    with progress(len(clips), "Transcribing") as advance:
        for i, clip in enumerate(clips, 1):
            path = f"{video_dir}/{clip.id}.mp4"
            if not os.path.exists(path):
                missing += 1
                advance()
                continue

            if model is None:
                log(SCOPE_CLASSIFY, f"loading {whisper_model}…")
                model = whisper.load_model(whisper_model)
            try:
                text, language, conf, avg_logprob, compression_ratio = _transcribe(
                    model, path
                )
            except Exception:
                errored += 1
                advance(detail=f"{clip.id}: transcription error (left unresolved)")
                continue

            clip.speech_transcription = text
            clip.speech_language = language or None
            clip.speech_confidence = conf if text else None
            clip.speech_avg_logprob = avg_logprob if text else None
            clip.speech_compression_ratio = compression_ratio if text else None

            low_logprob = bool(text) and avg_logprob < logprob_threshold
            high_compression = bool(text) and compression_ratio > compression_threshold
            dirty = (
                low_logprob
                or high_compression
                or has_hallucination_marker(text)
                or is_repeated_output(text)
            )
            meaningful = has_meaningful_speech_text(text, min_meaningful_chars)

            if meaningful and not dirty:
                clip.is_speech_detected = True
                detected += 1
                preview = text[:60] + ("…" if len(text) > 60 else "")
                advance(detail=f'{clip.id}: "{preview}"')
            else:
                clip.is_speech_detected = False
                no_speech += 1
                advance()

            if i % commit_every == 0:
                session.commit()

    session.commit()
    session.close()
    parts = [f"{detected} with speech", f"{no_speech} silent"]
    if missing:
        parts.append(f"{missing} skipped (video not downloaded yet)")
    if errored:
        parts.append(f"{errored} skipped (transcription error)")
    log(SCOPE_CLASSIFY, f"done — {', '.join(parts)}", level="ok")


def clean_speech() -> None:
    """Reset is_speech_detected=False for clips whose translation matches a
    hallucination marker — a post-hoc safety net for cases the classifier
    let through."""
    session = get_session()
    filter_conditions = [
        Clip.speech_translation.contains(marker) for marker in HALLUCINATION_MARKERS
    ]
    clips = (
        session.query(Clip)
        .filter(
            *clip_has_detected_speech(),
            Clip.speech_translation.is_not(None),
            or_(*filter_conditions),
        )
        .order_by(Clip.id)
        .all()
    )
    if not clips:
        session.close()
        return

    for clip in clips:
        clip.is_speech_detected = False
        log(
            SCOPE_CLEAN,
            f"{clip.id}: marked is_speech_detected=False "
            f'(translation: "{(clip.speech_translation or "")[:60]}")',
        )

    session.commit()
    session.close()
    log(SCOPE_CLEAN, f"done — {len(clips)} clips cleared")
