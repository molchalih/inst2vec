"""VAD (voice-activity-detection) helper stub.

This module is intentionally minimal — its purpose is to fix the public
API shape so a real VAD backend (e.g. silero-vad, webrtcvad) can be
dropped in without touching ``classify.py``.

Behavior today:
    enabled=False  → returns the input path unchanged
    enabled=True   → returns None if input is missing, else input path
                     (no actual trimming happens yet)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VadConfig:
    enabled: bool = False
    min_speech_seconds: float = 0.5
    aggressiveness: int = 2  # 0..3 — webrtcvad-style scale


def prepare_for_whisper(
    video: Path,
    out_dir: Path,  # reserved for future trimmed-audio output
    config: VadConfig,
) -> Path | None:
    """Return a path Whisper should transcribe, or None to skip the clip.

    Returns the input ``video`` unchanged when VAD is disabled. When enabled,
    today's implementation only filters out missing inputs; a future patch
    will produce a trimmed audio file inside ``out_dir``.
    """
    if not video.exists():
        return None
    if not config.enabled:
        return video
    # Future: invoke real VAD, write a trimmed audio file to ``out_dir``,
    # return that path or None if no speech segments survived.
    return video
