"""Manual recovery: re-attempt clips with is_speech_detected IS NULL.

Single-pass call to public ``classify_speech``. Identical decision logic to
the pipeline run, but operator-controlled (no aggressive retry loop, no
duplicated Whisper logic).

Usage:
    uv run python scripts/retry_failed_speech_detection.py
"""

from __future__ import annotations

import os

from modules.config import load_runtime_config
from modules.console import log
from modules.database import (
    Clip,
    clip_needs_speech_detection,
    get_session,
    init_db,
)
from modules.speech import classify_speech

SCOPE = "retry-speech"


def retry_failed_speech_detection(
    video_dir: str,
    whisper_model: str,
    commit_every: int,
    logprob_threshold: float,
    compression_threshold: float,
    min_meaningful_chars: int,
) -> None:
    session = get_session()
    try:
        unresolved = session.query(Clip).filter(*clip_needs_speech_detection()).count()
    finally:
        session.close()

    if not unresolved:
        log(SCOPE, "no unresolved clips to retry")
        return

    log(SCOPE, f"retrying {unresolved} unresolved clips")
    classify_speech(
        video_dir=video_dir,
        whisper_model=whisper_model,
        commit_every=commit_every,
        logprob_threshold=logprob_threshold,
        compression_threshold=compression_threshold,
        min_meaningful_chars=min_meaningful_chars,
    )


if __name__ == "__main__":
    settings, secrets = load_runtime_config()
    init_db(secrets.database_url, secrets.identity_db_url)
    os.makedirs(settings.paths.video_dir, exist_ok=True)
    retry_failed_speech_detection(
        video_dir=settings.paths.video_dir,
        whisper_model=settings.speech.whisper_model,
        commit_every=settings.speech.commit_every,
        logprob_threshold=settings.speech.logprob_threshold,
        compression_threshold=settings.speech.compression_threshold,
        min_meaningful_chars=settings.speech.min_meaningful_chars,
    )
