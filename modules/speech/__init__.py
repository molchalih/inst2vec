"""Speech pipeline: VAD pre-gate + classify (Whisper) + translate (TranslateGemma).

The top-level entry ``process_speech`` wraps the three row-level stages in
a single config-only fingerprint gate: on config drift it nulls every
speech output column on eligible clips, then lets the row-level
idempotence in classify/translate/clean repopulate them. On a match it
skips the reset and lets the existing row-level loops pick up any
partial work from a previous run.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any

from sqlalchemy.orm import Session

from core import fingerprint as fp
from core.config import SpeechSettings
from core.console import log
from core.database import StageState, get_engine
from modules.speech.classify import classify_speech, clean_speech
from modules.speech.state import (
    SCOPE_SPEECH,
    STAGE_SPEECH,
    reset_speech_outputs,
)
from modules.speech.translate import translate_speech
from modules.speech.vad import VadConfig, VadResult, prepare_for_whisper

__all__ = [
    "VadConfig",
    "VadResult",
    "classify_speech",
    "clean_speech",
    "prepare_for_whisper",
    "process_speech",
    "translate_speech",
]


def _config_payload(cfg: Any) -> str:
    """Stable JSON of the speech settings used in the config hash."""
    if hasattr(cfg, "model_dump"):
        payload = cfg.model_dump()
    elif is_dataclass(cfg):
        payload = asdict(cfg)
    else:
        payload = dict(cfg.__dict__)
    return json.dumps(payload, sort_keys=True, default=str)


def process_speech(
    cfg: SpeechSettings,
    *,
    video_dir: str,
    speech_audio_dir: str,
) -> None:
    """Fingerprint-gated entry point for the speech pipeline."""
    current = fp.Fingerprint(
        data=fp.hash_text(""),
        config=fp.hash_text(_config_payload(cfg)),
        dependency=fp.hash_text(""),
    )

    with Session(get_engine()) as session:
        stored = session.get(StageState, (STAGE_SPEECH, SCOPE_SPEECH))
        if stored is None:
            log("speech", "no prior state — sealing on completion")
        elif fp.is_stale(session, STAGE_SPEECH, SCOPE_SPEECH, current):
            diff = fp.describe_diff(session, STAGE_SPEECH, SCOPE_SPEECH, current)
            log("speech", f"config drift ({diff}) — resetting outputs")
            reset_speech_outputs(session)
        else:
            log("speech", "fingerprint match — skipping reset")

    classify_speech(
        video_dir=video_dir,
        speech_audio_dir=speech_audio_dir,
        whisper_model=cfg.whisper_model,
        commit_every=cfg.commit_every,
        logprob_threshold=cfg.logprob_threshold,
        compression_threshold=cfg.compression_threshold,
        min_meaningful_chars=cfg.min_meaningful_chars,
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
    )
    clean_speech()

    with Session(get_engine()) as session:
        fp.mark_complete(session, STAGE_SPEECH, SCOPE_SPEECH, current)
        session.commit()
