"""ffmpeg helper: extract a short stereo audio sample from a video."""

from __future__ import annotations

from pathlib import Path

from core.config import MusicSettings
from core.ffmpeg import run_ffmpeg


def extract_audio_sample(
    video: Path,
    out_dir: Path,
    music: MusicSettings,
) -> Path | None:
    """Run ffmpeg to extract a stereo audio sample. WAV preferred, MP3 as size fallback.

    Returns the output path on success, or None on ffmpeg failure / timeout / oversize.
    """
    sample_max_bytes = int(music.manual_features_max_mb * 1024 * 1024)
    base = [
        "ffmpeg",
        "-y",
        "-i",
        str(video),
        "-vn",
        "-t",
        str(music.manual_features_max_seconds),
        "-ac",
        "2",
        "-ar",
        str(music.manual_features_sample_rate),
    ]

    wav = out_dir / f"{video.stem}.wav"
    if (
        run_ffmpeg(
            [*base, "-c:a", "pcm_s16le", str(wav)],
            timeout=music.ffmpeg_timeout_seconds,
        )
        and wav.exists()
        and wav.stat().st_size <= sample_max_bytes
    ):
        return wav

    mp3 = out_dir / f"{video.stem}.mp3"
    if (
        run_ffmpeg(
            [*base, "-b:a", music.manual_features_mp3_bitrate, str(mp3)],
            timeout=music.ffmpeg_timeout_seconds,
        )
        and mp3.exists()
        and mp3.stat().st_size <= sample_max_bytes
    ):
        return mp3

    return None
