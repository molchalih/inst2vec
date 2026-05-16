"""Embedding case specifications.

``embedding_case`` is the stable identity of a complete embedding recipe
(modality + provider + model family/version + prompt + input-building
logic). It is the only idempotence boundary the embeddings package
recognizes — completion is derived from rows in ClipEmbedding /
UserEmbedding keyed by ``(clip_id|user_id, embedding_case)``.

Note: ``"audio"`` is currently an audio-SEMANTIC TEXT embedding (built
from speech + verbalized music), not a raw-waveform embedding. The name
is kept for thesis/pipeline simplicity.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from modules.embeddings.providers import LocalQwenProvider, Provider
from modules.embeddings.text import build_audio_text, build_sandwich_text

AUDIO_INSTRUCTION = (
    "Represent the audio character of this video: its musical mood, energy, "
    "and any spoken content."
)


@dataclass(frozen=True)
class EmbeddingCaseSpec:
    name: str
    text_builder: Callable[[object, dict], str | None] | None
    requires_video: bool
    requires_text: bool
    instruction: str | None
    provider_factory: Callable[[object], Provider]
    payload_builder: Callable[
        [object, str | None, str | None, float | None, int | None], dict
    ]
    apply_video_token_fallback: bool


# ── provider factories ───────────────────────────────────────────────────────


def _local_qwen_video_factory(settings) -> Provider:
    return LocalQwenProvider(
        model_path=settings.paths.model_path,
        max_length=settings.embeddings.embed_max_length,
        max_frames=settings.embeddings.adaptive_max_frames,
        fps=settings.embeddings.adaptive_default_fps,
    )


def _local_qwen_text_factory(settings) -> Provider:
    return LocalQwenProvider(
        model_path=settings.paths.model_path,
        max_length=settings.embeddings.embed_max_length,
    )


# ── payload builders ─────────────────────────────────────────────────────────


def _video_payload(clip, text, video_path, fps, max_frames) -> dict:
    return {"video": video_path, "fps": fps, "max_frames": max_frames}


def _sandwich_payload(clip, text, video_path, fps, max_frames) -> dict:
    return {
        "video": video_path,
        "fps": fps,
        "max_frames": max_frames,
        "text": text,
    }


def _audio_payload(clip, text, video_path, fps, max_frames) -> dict:
    return {"text": text, "instruction": AUDIO_INSTRUCTION}


# ── registry ─────────────────────────────────────────────────────────────────


VIDEO_CASE = EmbeddingCaseSpec(
    name="video",
    text_builder=None,
    requires_video=True,
    requires_text=False,
    instruction=None,
    provider_factory=_local_qwen_video_factory,
    payload_builder=_video_payload,
    apply_video_token_fallback=True,
)

SANDWICH_CASE = EmbeddingCaseSpec(
    name="sandwich",
    text_builder=build_sandwich_text,
    requires_video=True,
    requires_text=True,
    instruction=None,
    provider_factory=_local_qwen_video_factory,
    payload_builder=_sandwich_payload,
    apply_video_token_fallback=True,
)

AUDIO_CASE = EmbeddingCaseSpec(
    name="audio",
    text_builder=build_audio_text,
    requires_video=False,
    requires_text=True,
    instruction=AUDIO_INSTRUCTION,
    provider_factory=_local_qwen_text_factory,
    payload_builder=_audio_payload,
    apply_video_token_fallback=False,
)


CASE_REGISTRY: dict[str, EmbeddingCaseSpec] = {
    "video": VIDEO_CASE,
    "sandwich": SANDWICH_CASE,
    "audio": AUDIO_CASE,
}

DEFAULT_CASES: tuple[str, ...] = ("video", "sandwich", "audio")
