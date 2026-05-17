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

import os as _os
from collections.abc import Callable
from dataclasses import dataclass

from modules.embeddings.providers import (
    LocalQwenProvider,
    Provider,
    RemoteQwenProvider,
)
from modules.embeddings.text import build_audio_text, build_sandwich_text
from modules.storage import get_object_store

AUDIO_INSTRUCTION = (
    "Represent the audio character of this video: its musical mood, energy, "
    "and any spoken content."
)


@dataclass(frozen=True)
class EmbeddingCaseSpec:
    name: str
    text_builder: Callable[[object, dict], str | None] | None
    requires_video: bool
    provider_factory: Callable[[object, object], Provider]
    payload_builder: Callable[
        [object, str | None, str | None, float | None, int | None], dict
    ]
    apply_video_token_fallback: bool


# ── provider factories ───────────────────────────────────────────────────────


def _qwen_video_factory(settings, secrets) -> Provider:
    if settings.embeddings.provider == "remote":
        return RemoteQwenProvider(
            url=secrets.embedder_remote_url,
            token=secrets.embedder_token,
            storage=get_object_store(settings, secrets),
            timeout_s=settings.embeddings.request_timeout_s,
            max_retries=settings.embeddings.max_retries,
        )
    return LocalQwenProvider(
        model_path=settings.paths.model_path,
        max_length=settings.embeddings.embed_max_length,
        max_frames=settings.embeddings.adaptive_max_frames,
        fps=settings.embeddings.adaptive_default_fps,
    )


def _qwen_text_factory(settings, secrets) -> Provider:
    if settings.embeddings.provider == "remote":
        return RemoteQwenProvider(
            url=secrets.embedder_remote_url,
            token=secrets.embedder_token,
            storage=get_object_store(settings, secrets),
            timeout_s=settings.embeddings.request_timeout_s,
            max_retries=settings.embeddings.max_retries,
        )
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
    provider_factory=_qwen_video_factory,
    payload_builder=_video_payload,
    apply_video_token_fallback=True,
)

SANDWICH_CASE = EmbeddingCaseSpec(
    name="sandwich",
    text_builder=build_sandwich_text,
    requires_video=True,
    provider_factory=_qwen_video_factory,
    payload_builder=_sandwich_payload,
    apply_video_token_fallback=True,
)

AUDIO_CASE = EmbeddingCaseSpec(
    name="audio",
    text_builder=build_audio_text,
    requires_video=False,
    provider_factory=_qwen_text_factory,
    payload_builder=_audio_payload,
    apply_video_token_fallback=False,
)


CASE_REGISTRY: dict[str, EmbeddingCaseSpec] = {
    "video": VIDEO_CASE,
    "sandwich": SANDWICH_CASE,
    "audio": AUDIO_CASE,
}

DEFAULT_CASES: tuple[str, ...] = ("video", "sandwich", "audio")


# Recipe versions for text builders. Bump the value when the corresponding
# build_*_text logic changes semantics so existing rows are invalidated.
TEXT_RECIPE_VERSIONS: dict[str, str] = {
    "video": "none",
    "sandwich": "sandwich_v1",
    "audio": "audio_v1",
}


def case_config_identity(spec: EmbeddingCaseSpec, settings) -> str:
    """Stable identity string for a case's recipe + relevant settings.

    Fed into ``fingerprint.hash_text`` to produce the case's config_hash.
    Co-located with the spec definitions so changing a case's identity
    inputs lives next to the case itself.
    """
    parts = [
        f"case={spec.name}",
        f"provider={_legacy_provider_name(spec.provider_factory)}",
        f"model={_os.path.basename(settings.paths.model_path)}",
        f"max_len={settings.embeddings.embed_max_length}",
        f"max_frames={settings.embeddings.adaptive_max_frames}",
        f"fps={settings.embeddings.adaptive_default_fps}",
        f"token_fallback={spec.apply_video_token_fallback}",
        f"text_recipe={TEXT_RECIPE_VERSIONS.get(spec.name, 'unknown')}",
    ]
    if spec.name == "audio":
        parts.append(f"instruction={AUDIO_INSTRUCTION}")
    return "|".join(parts)


def _legacy_provider_name(factory) -> str:
    """Map the new (post-remote-switch) factory names back to their
    pre-switch identities so existing clip embeddings stay valid.
    The names changed from `_local_qwen_*_factory` to `_qwen_*_factory`
    when the remote path was added, but the local-provider config that
    each factory produces is bit-for-bit identical, so the config
    fingerprint must not change.
    """
    name = getattr(factory, "__name__", repr(factory))
    return {
        "_qwen_video_factory": "_local_qwen_video_factory",
        "_qwen_text_factory": "_local_qwen_text_factory",
    }.get(name, name)
