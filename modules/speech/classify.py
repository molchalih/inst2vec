"""Whisper transcription stage: classify clips as speech / no-speech."""

from __future__ import annotations

import time
from pathlib import Path

import whisper
from sqlalchemy import or_

from core.console import log, progress
from core.database import (
    Clip,
    clip_has_detected_speech,
    clip_needs_speech_detection,
    get_session,
)
from core.ffmpeg import probe_audio_stream
from modules.speech.state import (
    HALLUCINATION_MARKERS,
    has_hallucination_marker,
    has_low_letter_ratio,
    has_meaningful_speech_text,
    is_repeated_output,
    is_too_short,
)
from modules.speech.vad import VadConfig, prepare_for_whisper

SCOPE = "whisper"
SCOPE_CLEAN = "whisper:clean"


def _transcribe(model, path: str) -> tuple[str, str, float, float, float]:
    """Return (text, language, speech_confidence, avg_logprob, compression_ratio).

    Whisper is called with ``condition_on_previous_text=False``, ``beam_size=1``
    and ``temperature=0`` to suppress long-context drift, beam-search loop
    collapse, and sampling-induced hallucinations.
    """
    result = model.transcribe(
        path,
        condition_on_previous_text=False,
        beam_size=1,
        temperature=0,
    )
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
    speech_audio_dir: str,
    whisper_model: str,
    commit_every: int,
    logprob_threshold: float,
    compression_threshold: float,
    min_meaningful_chars: int,
    dirty_min_chars: int,
    dirty_min_letter_ratio: float,
    vad_config: VadConfig,
) -> None:
    """Transcribe all unresolved clips with Whisper, gated by Silero VAD.

    Per-clip flow:
        No audio stream        → is_speech_detected = False (terminal)
        VAD enabled, no speech → is_speech_detected = False (Whisper skipped)
        VAD enabled, speech    → transcribe speech-only WAV
        VAD disabled           → transcribe the raw video as before
        Hallucinated/empty     → is_speech_detected = False
        Missing file           → leave NULL (retryable)
        ffprobe failure        → leave NULL (retryable)
        VAD/Whisper exception  → leave NULL (retryable)
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

    speech_out = Path(speech_audio_dir)
    speech_out.mkdir(parents=True, exist_ok=True)

    log(SCOPE, "SCAN", "clips", "ok", stats={"todo": len(clips)})
    model: whisper.Whisper | None = None
    detected = no_speech_vad = no_speech = missing = errored = 0
    t_stage = time.perf_counter()

    with progress(len(clips), "Transcribing") as advance:
        for i, clip in enumerate(clips, 1):
            video_path = Path(video_dir) / f"{clip.id}.mp4"
            if not video_path.exists():
                missing += 1
                log(
                    SCOPE,
                    "ASR",
                    f"clip_{clip.id}",
                    "ERR",
                    stats={"err": "video not downloaded yet"},
                )
                advance()
                continue

            probe = probe_audio_stream(str(video_path))
            if probe is None:
                errored += 1
                log(
                    SCOPE,
                    "ASR",
                    f"clip_{clip.id}",
                    "ERR",
                    stats={"err": "ffprobe failed"},
                )
                advance(detail=f"{clip.id}: ffprobe failed (left unresolved)")
                continue
            if probe is False:
                clip.is_speech_detected = False
                no_speech_vad += 1
                log(
                    SCOPE,
                    "ASR",
                    f"clip_{clip.id}",
                    "none",
                    stats={"reason": "no audio stream"},
                )
                advance(detail=f"{clip.id}: no audio stream")
                if i % commit_every == 0:
                    session.commit()
                continue

            t0 = time.perf_counter()
            try:
                vad = prepare_for_whisper(video_path, speech_out, vad_config)
            except Exception as exc:
                errored += 1
                log(
                    SCOPE,
                    "ASR",
                    f"clip_{clip.id}",
                    "ERR",
                    stats={
                        "time": time.perf_counter() - t0,
                        "err": f"VAD error: {exc}",
                    },
                )
                advance(detail=f"{clip.id}: VAD error (left unresolved)")
                continue

            if not vad.is_speech_detected:
                clip.is_speech_detected = False
                no_speech_vad += 1
                log(
                    SCOPE,
                    "ASR",
                    f"clip_{clip.id}",
                    "none",
                    stats={"time": time.perf_counter() - t0, "src": "vad"},
                )
                advance(detail=f"{clip.id}: VAD silent")
                if i % commit_every == 0:
                    session.commit()
                continue

            assert (
                vad.speech_audio_path is not None
            )  # guaranteed by VadResult invariant
            if model is None:
                t_load = time.perf_counter()
                model = whisper.load_model(whisper_model)
                log(
                    SCOPE,
                    "LOAD",
                    whisper_model,
                    "ok",
                    stats={"time": time.perf_counter() - t_load},
                )
            try:
                text, language, conf, avg_logprob, compression_ratio = _transcribe(
                    model, str(vad.speech_audio_path)
                )
            except Exception as exc:
                errored += 1
                log(
                    SCOPE,
                    "ASR",
                    f"clip_{clip.id}",
                    "ERR",
                    stats={
                        "time": time.perf_counter() - t0,
                        "err": f"transcription error: {exc}",
                    },
                )
                advance(detail=f"{clip.id}: transcription error (left unresolved)")
                continue

            clip.speech_transcription = text
            clip.speech_language = language or None
            clip.speech_confidence = conf if text else None
            clip.speech_avg_logprob = avg_logprob if text else None
            clip.speech_compression_ratio = compression_ratio if text else None

            low_logprob = bool(text) and avg_logprob < logprob_threshold
            high_compression = bool(text) and compression_ratio > compression_threshold
            too_short = is_too_short(text, min_chars=dirty_min_chars)
            low_letter_ratio = has_low_letter_ratio(
                text, min_ratio=dirty_min_letter_ratio
            )
            dirty = (
                low_logprob
                or high_compression
                or too_short
                or low_letter_ratio
                or has_hallucination_marker(text)
                or is_repeated_output(text)
            )
            meaningful = has_meaningful_speech_text(text, min_meaningful_chars)

            if meaningful and not dirty:
                clip.is_speech_detected = True
                detected += 1
                preview = text[:60] + ("…" if len(text) > 60 else "")
                log(
                    SCOPE,
                    "ASR",
                    f"clip_{clip.id}",
                    language or "ok",
                    stats={
                        "time": time.perf_counter() - t0,
                        "conf": round(conf, 2),
                    },
                )
                advance(detail=f'{clip.id}: "{preview}"')
            else:
                clip.is_speech_detected = False
                no_speech += 1
                log(
                    SCOPE,
                    "ASR",
                    f"clip_{clip.id}",
                    "none",
                    stats={"time": time.perf_counter() - t0, "src": "whisper"},
                )
                advance()

            if i % commit_every == 0:
                session.commit()

    session.commit()
    session.close()
    log(
        SCOPE,
        "SEAL",
        "transcribe",
        "ok",
        stats={
            "speech": detected,
            "silent_vad": no_speech_vad,
            "silent_whisper": no_speech,
            "missing": missing,
            "err": errored,
            "time": time.perf_counter() - t_stage,
        },
    )


def clean_speech() -> None:
    """Null the seven speech columns for clips whose translation matches a
    hallucination marker — a post-hoc safety net for cases the classifier
    let through.

    Nulls all seven columns (matching reset_speech_outputs) so
    downstream text builders see the row as empty and the embedding
    dependency hash invalidates any sealed poisoned embeddings on the
    next run.
    """
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

    t_stage = time.perf_counter()
    for clip in clips:
        original = (clip.speech_translation or "")[:60]
        clip.is_speech_detected = False
        clip.speech_transcription = None
        clip.speech_language = None
        clip.speech_translation = None
        clip.speech_confidence = None
        clip.speech_avg_logprob = None
        clip.speech_compression_ratio = None
        log(
            SCOPE_CLEAN,
            "CLEAN",
            f"clip_{clip.id}",
            "ok",
            stats={"preview": original[:32]},
        )

    session.commit()
    session.close()
    log(
        SCOPE_CLEAN,
        "SEAL",
        "clean",
        "ok",
        stats={"cleared": len(clips), "time": time.perf_counter() - t_stage},
    )
