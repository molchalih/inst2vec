"""CLI wrapper. Real logic lives in modules/speech/retry.py."""

from __future__ import annotations

import os

from core.config import load_runtime_config
from core.database import init_db
from modules.speech import VadConfig
from modules.speech.retry import retry_failed_detection


def main() -> None:
    settings, secrets = load_runtime_config()
    init_db(secrets.database_url, secrets.identity_db_url)
    os.makedirs(settings.paths.video_dir, exist_ok=True)
    os.makedirs(settings.paths.speech_audio_dir, exist_ok=True)
    retry_failed_detection(
        video_dir=settings.paths.video_dir,
        speech_audio_dir=settings.paths.speech_audio_dir,
        whisper_model=settings.speech.whisper_model,
        commit_every=settings.speech.commit_every,
        logprob_threshold=settings.speech.logprob_threshold,
        compression_threshold=settings.speech.compression_threshold,
        min_meaningful_chars=settings.speech.min_meaningful_chars,
        vad_config=VadConfig(
            enabled=settings.speech.vad_enabled,
            sampling_rate=settings.speech.vad_sampling_rate,
            threshold=settings.speech.vad_threshold,
            min_speech_ms=settings.speech.vad_min_speech_ms,
            min_silence_ms=settings.speech.vad_min_silence_ms,
            speech_pad_ms=settings.speech.vad_speech_pad_ms,
            min_total_speech_s=settings.speech.vad_min_total_speech_s,
        ),
    )


if __name__ == "__main__":
    main()
