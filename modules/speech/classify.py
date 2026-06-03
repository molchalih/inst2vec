"""Whisper transcription stage: classify clips as speech / no-speech."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch
from faster_whisper import WhisperModel
from sqlalchemy import or_

from core.console import progress
from core.database import (
    Clip,
    clip_has_detected_speech,
    clip_needs_speech_detection,
    get_session,
)
from core.ffmpeg import probe_audio_stream
from core.log import event, item, scope
from modules.speech.state import (
    HALLUCINATION_MARKERS,
    has_hallucination_marker,
    has_low_letter_ratio,
    has_meaningful_speech_text,
    is_repeated_output,
    is_too_short,
)
from modules.speech.vad import VadConfig, prepare_for_whisper


def _transcribe(model: WhisperModel, path: str) -> tuple[str, str, float, float, float]:
    """Return (text, language, speech_confidence, avg_logprob, compression_ratio).

    Whisper is called with ``condition_on_previous_text=False``, ``beam_size=1``
    and ``temperature=0`` to suppress long-context drift, beam-search loop
    collapse, and sampling-induced hallucinations. Backed by CTranslate2 via
    ``faster-whisper``; ``transcribe`` returns a (generator, info) pair so the
    segment iterator is drained eagerly here.
    """
    segments_iter, info = model.transcribe(
        path,
        condition_on_previous_text=False,
        beam_size=1,
        temperature=0,
    )
    segs = list(segments_iter)
    text = " ".join((s.text or "").strip() for s in segs).strip()
    language = info.language or ""
    if segs:
        mean_no_speech = sum(s.no_speech_prob for s in segs) / len(segs)
        confidence = max(0.0, min(1.0, 1.0 - mean_no_speech))
        avg_logprob = sum(s.avg_logprob for s in segs) / len(segs)
        compression_ratio = sum(s.compression_ratio for s in segs) / len(segs)
    else:
        confidence = avg_logprob = compression_ratio = 0.0
    return text, language, confidence, avg_logprob, compression_ratio


@dataclass
class _SpeechCounters:
    detected: int = 0
    no_speech_vad: int = 0
    no_speech: int = 0
    missing: int = 0
    errored: int = 0


def _preflight_clip(
    clip,
    video_path: Path,
    speech_out: Path,
    vad_config: VadConfig,
    counters: _SpeechCounters,
    advance,
):
    """Run the missing/ffprobe/no-audio/VAD gates before transcription.

    Returns the ``VadResult`` to transcribe, or ``None`` when the clip is
    resolved here (counter bumped, event + progress emitted). A ``True``
    sentinel marks the silent/no-audio cases that the caller must
    ``commit_every``-flush; ``False`` marks the leave-NULL retryable cases.
    """
    if not video_path.exists():
        counters.missing += 1
        event(
            "EXTRACT",
            f"clip_{clip.id}",
            result="ERR",
            stats={"err": "video not downloaded yet"},
        )
        advance()
        return False

    probe = probe_audio_stream(str(video_path))
    if probe is None:
        counters.errored += 1
        event(
            "EXTRACT",
            f"clip_{clip.id}",
            result="ERR",
            stats={"err": "ffprobe failed"},
        )
        advance(detail=f"{clip.id}: ffprobe failed (left unresolved)")
        return False
    if probe is False:
        clip.is_speech_detected = False
        counters.no_speech_vad += 1
        event("SKIP", f"clip_{clip.id}", stats={"reason": "no_audio_stream"})
        advance(detail=f"{clip.id}: no audio stream")
        return True

    t0 = time.perf_counter()
    try:
        vad = prepare_for_whisper(video_path, speech_out, vad_config)
    except Exception as exc:
        counters.errored += 1
        event(
            "EXTRACT",
            f"clip_{clip.id}",
            result="ERR",
            stats={"time": time.perf_counter() - t0, "err": f"VAD error: {exc}"},
        )
        advance(detail=f"{clip.id}: VAD error (left unresolved)")
        return False

    if not vad.is_speech_detected:
        clip.is_speech_detected = False
        counters.no_speech_vad += 1
        event(
            "SKIP",
            f"clip_{clip.id}",
            stats={"time": time.perf_counter() - t0, "reason": "vad_silent"},
        )
        advance(detail=f"{clip.id}: VAD silent")
        return True

    return vad


def _transcribe_and_classify(
    clip,
    model: WhisperModel,
    speech_audio_path: str,
    counters: _SpeechCounters,
    advance,
    *,
    logprob_threshold: float,
    compression_threshold: float,
    min_meaningful_chars: int,
    dirty_min_chars: int,
    dirty_min_letter_ratio: float,
) -> None:
    """Transcribe one clip, persist columns, and classify speech vs. noise."""
    with item("EXTRACT", f"clip_{clip.id}") as t:
        text, language, conf, avg_logprob, compression_ratio = _transcribe(
            model, speech_audio_path
        )

        clip.speech_transcription = text
        clip.speech_language = language or None
        clip.speech_confidence = conf if text else None
        clip.speech_avg_logprob = avg_logprob if text else None
        clip.speech_compression_ratio = compression_ratio if text else None

        dirty = _is_dirty_transcription(
            text,
            avg_logprob,
            compression_ratio,
            logprob_threshold=logprob_threshold,
            compression_threshold=compression_threshold,
            dirty_min_chars=dirty_min_chars,
            dirty_min_letter_ratio=dirty_min_letter_ratio,
        )
        meaningful = has_meaningful_speech_text(text, min_meaningful_chars)

        if meaningful and not dirty:
            clip.is_speech_detected = True
            counters.detected += 1
            preview = text[:60] + ("…" if len(text) > 60 else "")
            t.stats(lang=language, conf=round(conf, 2))
            advance(detail=f'{clip.id}: "{preview}"')
        else:
            clip.is_speech_detected = False
            counters.no_speech += 1
            t.stats(lang=language, speech=False)
            advance()

    if t.failed:
        counters.errored += 1


def _load_whisper_model(whisper_model: str) -> WhisperModel:
    """Load the faster-whisper model, picking the device/compute type.

    CTranslate2 backend; ``compute_type="float16"`` on GPU, ``int8`` on CPU
    dev (CT2 has no fp16 CPU kernels). Same model-name strings as the
    openai-whisper CLI, so the ``whisper_model`` config value remains the
    source of truth.
    """
    t_load = time.perf_counter()
    if torch.cuda.is_available():
        device, compute_type = "cuda", "float16"
    else:
        device, compute_type = "cpu", "int8"
    model = WhisperModel(whisper_model, device=device, compute_type=compute_type)
    event("LOAD", whisper_model, stats={"time": time.perf_counter() - t_load})
    return model


def _is_dirty_transcription(
    text: str,
    avg_logprob: float,
    compression_ratio: float,
    *,
    logprob_threshold: float,
    compression_threshold: float,
    dirty_min_chars: int,
    dirty_min_letter_ratio: float,
) -> bool:
    """True when a transcript looks like noise/hallucination and must be rejected."""
    low_logprob = bool(text) and avg_logprob < logprob_threshold
    high_compression = bool(text) and compression_ratio > compression_threshold
    too_short = is_too_short(text, min_chars=dirty_min_chars)
    low_letter_ratio = has_low_letter_ratio(text, min_ratio=dirty_min_letter_ratio)
    return (
        low_logprob
        or high_compression
        or too_short
        or low_letter_ratio
        or has_hallucination_marker(text)
        or is_repeated_output(text)
    )


@scope("speech")
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
) -> tuple[int, int]:
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
        return 0, 0

    speech_out = Path(speech_audio_dir)
    speech_out.mkdir(parents=True, exist_ok=True)

    event("SCAN", "clips", stats={"todo": len(clips)})
    model: WhisperModel | None = None
    counters = _SpeechCounters()

    with progress(len(clips), "Transcribing") as advance:
        for i, clip in enumerate(clips, 1):
            video_path = Path(video_dir) / f"{clip.id}.mp4"
            outcome = _preflight_clip(
                clip, video_path, speech_out, vad_config, counters, advance
            )
            if outcome is False:
                continue
            if outcome is True:
                if i % commit_every == 0:
                    session.commit()
                continue

            vad = outcome
            assert (
                vad.speech_audio_path is not None
            )  # guaranteed by VadResult invariant
            if model is None:
                model = _load_whisper_model(whisper_model)

            _transcribe_and_classify(
                clip,
                model,
                str(vad.speech_audio_path),
                counters,
                advance,
                logprob_threshold=logprob_threshold,
                compression_threshold=compression_threshold,
                min_meaningful_chars=min_meaningful_chars,
                dirty_min_chars=dirty_min_chars,
                dirty_min_letter_ratio=dirty_min_letter_ratio,
            )

            if i % commit_every == 0:
                session.commit()

    session.commit()
    session.close()
    event(
        "SCAN",
        "transcribe",
        stats={
            "speech": counters.detected,
            "silent_vad": counters.no_speech_vad,
            "silent_whisper": counters.no_speech,
            "missing": counters.missing,
            "err": counters.errored,
        },
    )
    return counters.detected, counters.errored


@scope("speech:clean")
def clean_speech() -> int:
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
        return 0

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
        event("CLEAN", f"clip_{clip.id}", stats={"preview": original[:32]})

    session.commit()
    session.close()
    event(
        "SCAN",
        "clean",
        stats={"cleared": len(clips), "time": time.perf_counter() - t_stage},
    )
    return len(clips)
