"""Manual recovery for failed Speech rows.

Functions here are also invoked by ``scripts/retry_failed_speech_detection.py``
CLI wrapper. Tests import from this module directly.
"""

from __future__ import annotations

from core.console import log
from core.database import Clip, clip_needs_speech_detection, get_session
from modules.speech import VadConfig, classify_speech

SCOPE = "retry-speech"


def retry_failed_detection(
    video_dir: str,
    speech_audio_dir: str,
    whisper_model: str,
    commit_every: int,
    logprob_threshold: float,
    compression_threshold: float,
    min_meaningful_chars: int,
    vad_config: VadConfig,
) -> None:
    """Re-attempt clips with is_speech_detected IS NULL.

    Single-pass call to classify_speech. Identical decision logic to the
    pipeline run, but operator-controlled.
    """
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
        speech_audio_dir=speech_audio_dir,
        whisper_model=whisper_model,
        commit_every=commit_every,
        logprob_threshold=logprob_threshold,
        compression_threshold=compression_threshold,
        min_meaningful_chars=min_meaningful_chars,
        vad_config=vad_config,
    )
