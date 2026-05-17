"""Silero-VAD pre-gate for the speech pipeline.

Pipeline position:
    Inside ``classify_speech``, before any Whisper invocation.

Responsibilities:
    1. Normalize the input media (video or audio) to mono 16 kHz PCM-WAV via
       ffmpeg.
    2. Run Silero VAD on the normalized waveform to detect speech segments.
    3. If speech survives the duration gate, concatenate the (padded) speech
       segments and write a speech-only WAV to ``out_dir/<stem>.wav``.
    4. Return a ``VadResult`` describing the decision.

This module is DB-free and ffmpeg-isolated; callers handle persistence and
retries. Failure to run ffmpeg raises ``RuntimeError`` so the caller can leave
the row NULL for retry (mirrors Whisper's exception handling in classify.py).
"""

from __future__ import annotations

import contextlib
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Imported via attribute so tests can monkeypatch the module reference.
import silero_vad as _silero  # type: ignore[import-not-found]

from core.ffmpeg import run_ffmpeg as _run_ffmpeg


@dataclass(frozen=True)
class VadConfig:
    enabled: bool = True
    sampling_rate: int = 16000
    threshold: float = 0.5
    min_speech_ms: int = 250
    min_silence_ms: int = 100
    speech_pad_ms: int = 150
    min_total_speech_s: float = 0.5
    ffmpeg_timeout_s: int = 60


@dataclass(frozen=True)
class VadSegment:
    start_sample: int
    end_sample: int


@dataclass(frozen=True)
class VadResult:
    is_speech_detected: bool
    speech_audio_path: Path | None
    segments: list[VadSegment] = field(default_factory=list)
    total_speech_seconds: float = 0.0


_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = _silero.load_silero_vad()
    return _MODEL


def prepare_for_whisper(
    media_path: Path,
    out_dir: Path,
    config: VadConfig,
) -> VadResult:
    """Run the VAD pre-gate against ``media_path``.

    When VAD is disabled, returns a pass-through ``VadResult`` whose
    ``speech_audio_path`` is the input itself — Whisper sees the original file.

    When enabled:
        * No speech / total speech below ``min_total_speech_s`` →
          ``is_speech_detected=False``, no file written.
        * Speech detected → speech-only WAV written to
          ``out_dir/<media_stem>.wav``; ``is_speech_detected=True``.
    """
    if not config.enabled:
        return VadResult(is_speech_detected=True, speech_audio_path=media_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    samples = _load_mono_16k(
        media_path, out_dir, config.sampling_rate, config.ffmpeg_timeout_s
    )

    timestamps = _silero.get_speech_timestamps(
        _to_tensor(samples),
        _get_model(),
        sampling_rate=config.sampling_rate,
        threshold=config.threshold,
        min_speech_duration_ms=config.min_speech_ms,
        min_silence_duration_ms=config.min_silence_ms,
        speech_pad_ms=config.speech_pad_ms,
        return_seconds=False,
    )
    segments = [
        VadSegment(start_sample=int(ts["start"]), end_sample=int(ts["end"]))
        for ts in timestamps
    ]
    total_samples = sum(s.end_sample - s.start_sample for s in segments)
    total_speech_s = total_samples / float(config.sampling_rate)

    if total_speech_s < config.min_total_speech_s:
        return VadResult(
            is_speech_detected=False,
            speech_audio_path=None,
            segments=segments,
            total_speech_seconds=total_speech_s,
        )

    speech_only = np.concatenate(
        [samples[s.start_sample : s.end_sample] for s in segments]
    )
    out_path = out_dir / f"{media_path.stem}.wav"
    _write_wav(out_path, speech_only, config.sampling_rate)
    return VadResult(
        is_speech_detected=True,
        speech_audio_path=out_path,
        segments=segments,
        total_speech_seconds=total_speech_s,
    )


def _load_mono_16k(
    media_path: Path,
    out_dir: Path,
    sampling_rate: int,
    ffmpeg_timeout_s: int,
) -> np.ndarray:
    """ffmpeg -> int16 PCM WAV -> float32 numpy array in [-1, 1]."""
    tmp = out_dir / f"{media_path.stem}.vad.wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(media_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sampling_rate),
        "-c:a",
        "pcm_s16le",
        str(tmp),
    ]
    if not _run_ffmpeg(cmd, timeout=ffmpeg_timeout_s):
        raise RuntimeError(f"ffmpeg failed to normalize {media_path}")
    try:
        with wave.open(str(tmp), "rb") as wf:
            frames = wf.readframes(wf.getnframes())
        return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink()


def _to_tensor(samples: np.ndarray):
    import torch

    return torch.from_numpy(samples).contiguous()


def _write_wav(path: Path, samples: np.ndarray, sampling_rate: int) -> None:
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sampling_rate)
        wf.writeframes(pcm.tobytes())
