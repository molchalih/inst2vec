"""Speech pipeline: VAD pre-gate + classify (Whisper) + translate (TranslateGemma).

The top-level entry ``process_speech`` wraps the three row-level stages in
a single config-only fingerprint gate: on config drift it nulls every
speech output column on eligible clips, then lets the row-level
idempotence in classify/translate/clean repopulate them. On a match it
skips the reset and lets the existing row-level loops pick up any
partial work from a previous run.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from core import fingerprint as fp
from core.config import Secrets, Settings, SpeechSettings
from core.database import get_engine
from core.log import StageResult, scope, stage
from modules.speech.classify import classify_speech, clean_speech
from modules.speech.state import (
    SCOPE_SPEECH,
    STAGE_SPEECH,
    reset_speech_outputs,
    speech_config_payload,
)
from modules.speech.translate import translate_speech
from modules.speech.vad import VadConfig, VadResult, prepare_for_whisper

__all__ = [
    "VadConfig",
    "VadResult",
    "classify_speech",
    "clean_speech",
    "prepare_for_whisper",
    "run",
    "translate_speech",
]


@scope("speech")
def process_speech(
    cfg: SpeechSettings,
    *,
    video_dir: str,
    speech_audio_dir: str,
) -> tuple[int, int, int]:
    """Fingerprint-gated entry point for the speech pipeline."""
    current = fp.Fingerprint(
        data=fp.hash_text(""),
        config=fp.hash_text(speech_config_payload(cfg)),
        dependency=fp.hash_text(""),
    )

    with Session(get_engine()) as session:
        fp.gate(
            session,
            STAGE_SPEECH,
            SCOPE_SPEECH,
            current,
            reset_speech_outputs,
            log_scope="speech",
            drift_msg="resetting outputs",
        )
        session.commit()

    transcribed, failed = classify_speech(
        video_dir=video_dir,
        speech_audio_dir=speech_audio_dir,
        whisper_model=cfg.whisper_model,
        commit_every=cfg.commit_every,
        logprob_threshold=cfg.logprob_threshold,
        compression_threshold=cfg.compression_threshold,
        min_meaningful_chars=cfg.min_meaningful_chars,
        dirty_min_chars=cfg.dirty_min_chars,
        dirty_min_letter_ratio=cfg.dirty_min_letter_ratio,
        vad_config=VadConfig(
            enabled=cfg.vad_enabled,
            sampling_rate=cfg.vad_sampling_rate,
            threshold=cfg.vad_threshold,
            min_speech_ms=cfg.vad_min_speech_ms,
            min_silence_ms=cfg.vad_min_silence_ms,
            speech_pad_ms=cfg.vad_speech_pad_ms,
            min_total_speech_s=cfg.vad_min_total_speech_s,
            ffmpeg_timeout_s=cfg.vad_ffmpeg_timeout_s,
        ),
    )
    translate_speech(
        commit_every=cfg.commit_every,
        translate_model=cfg.translate_model,
        translate_target_lang=cfg.translate_target_lang,
        translation_max_chars=cfg.translation_max_chars,
        translate_max_new_tokens=cfg.translate_max_new_tokens,
        translate_batch_size=cfg.translate_batch_size,
    )
    cleaned = clean_speech()

    with Session(get_engine()) as session:
        fp.mark_complete(session, STAGE_SPEECH, SCOPE_SPEECH, current)
        session.commit()

    return transcribed, failed, cleaned


@stage("speech")
def run(settings: Settings, secrets: Secrets) -> StageResult:
    """Speech transcription + translation + post-clean."""
    transcribed, failed, cleaned = process_speech(
        settings.speech,
        video_dir=settings.paths.video_dir,
        speech_audio_dir=settings.paths.speech_audio_dir,
    )
    return StageResult(transcribed=transcribed, failed=failed, cleaned=cleaned)
