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
from dataclasses import dataclass, field

from core.storage import get_object_store
from modules.embeddings.providers import (
    LocalQwenProvider,
    Provider,
    RemoteQwenProvider,
)
from modules.embeddings.text import (
    build_audio_text,
    build_gemini_text,
    build_sandwich_text,
)

# audio_path is only consumed by the gemini case; every other builder
# accepts it and ignores it so the runner can pass it uniformly without
# branching by case name.

AUDIO_INSTRUCTION = (
    "Represent the audio character of this video: its musical mood, energy, "
    "and any spoken content."
)


@dataclass(frozen=True)
class EmbeddingSecrets:
    """Secret bag threaded through provider factories.

    Local providers ignore this; remote providers (Gemini, the
    self-hosted Qwen GPU pod) read the credentials they need. Every
    field defaults to ``""``/``None`` so ``EmbeddingSecrets()`` is a
    valid no-arg call for tests / pipelines that don't use remote
    providers.
    """

    gemini_api_key: str | None = None
    embedder_remote_url: str = ""
    embedder_token: str = ""
    object_store_endpoint: str = ""
    object_store_access_key: str = ""
    object_store_secret_key: str = ""


@dataclass(frozen=True)
class EmbeddingCaseSpec:
    name: str
    text_builder: Callable[[object, dict], str | None] | None
    requires_video: bool
    provider_factory: Callable[[object, object], Provider]
    payload_builder: Callable[
        [object, str | None, str | None, str | None, float | None, int | None], dict
    ]
    apply_video_token_fallback: bool
    # Names of Clip / Music columns (or synthetic ``_video_file_stat`` /
    # ``_audio_file_stat`` sentinels) whose values feed the case's text +
    # payload builders. Drives ``dependency_rows_for_case`` so adding a new
    # case never needs to touch state.py.
    dependency_columns: tuple[str, ...]
    # Bump when the corresponding text/payload-builder logic changes
    # semantics so existing rows are invalidated via case_config_identity.
    recipe_version: str
    # ``settings.embeddings`` bool attrs that all must be truthy for the
    # case to appear in ``default_cases``. Empty tuple means always-on.
    requires: tuple[str, ...] = field(default_factory=tuple)


# ── provider factories ───────────────────────────────────────────────────────


def _require_remote_config(settings, secrets) -> None:
    """Fail fast when [embeddings].provider = "remote" but the required
    URL / token / object-store settings are still at their empty defaults.

    Raised once at factory-time so the operator sees one actionable error
    instead of every clip embedding failing generically inside the runner.
    """
    missing: list[str] = []
    if not secrets.embedder_remote_url:
        missing.append("EMBEDDER_REMOTE_URL")
    if not secrets.embedder_token:
        missing.append("EMBEDDER_TOKEN")
    if not settings.storage.bucket:
        missing.append("[storage].bucket")
    if not secrets.object_store_access_key:
        missing.append("OBJECT_STORE_ACCESS_KEY")
    if not secrets.object_store_secret_key:
        missing.append("OBJECT_STORE_SECRET_KEY")
    if missing:
        raise RuntimeError("embeddings.provider=remote requires: " + ", ".join(missing))


def qwen_provider(settings, secrets, *, with_frames: bool) -> Provider:
    if settings.embeddings.provider == "remote":
        _require_remote_config(settings, secrets)
        return RemoteQwenProvider(
            url=secrets.embedder_remote_url,
            token=secrets.embedder_token,
            storage=get_object_store(settings, secrets),
            timeout_s=settings.embeddings.request_timeout_s,
            max_retries=settings.embeddings.max_retries,
        )
    if with_frames:
        return LocalQwenProvider(
            model_path=settings.paths.model_path,
            max_length=settings.embeddings.embed_max_length,
            max_frames=settings.embeddings.adaptive_max_frames,
            fps=settings.embeddings.adaptive_default_fps,
        )
    return LocalQwenProvider(
        model_path=settings.paths.model_path,
        max_length=settings.embeddings.embed_max_length,
    )


def _make_qwen_factory(*, with_frames: bool, name: str):
    def factory(settings, secrets):
        return qwen_provider(settings, secrets, with_frames=with_frames)

    factory.__name__ = name
    return factory


_QWEN_VIDEO = _make_qwen_factory(with_frames=True, name="qwen_provider_video")
_QWEN_TEXT = _make_qwen_factory(with_frames=False, name="qwen_provider_text")


# ── payload builders ─────────────────────────────────────────────────────────


def _video_payload(clip, text, video_path, audio_path, fps, max_frames) -> dict:
    return {"video": video_path, "fps": fps, "max_frames": max_frames}


def _sandwich_payload(clip, text, video_path, audio_path, fps, max_frames) -> dict:
    return {
        "video": video_path,
        "fps": fps,
        "max_frames": max_frames,
        "text": text,
    }


def _audio_payload(clip, text, video_path, audio_path, fps, max_frames) -> dict:
    return {"text": text, "instruction": AUDIO_INSTRUCTION}


# ── registry ─────────────────────────────────────────────────────────────────


VIDEO_CASE = EmbeddingCaseSpec(
    name="video",
    text_builder=None,
    requires_video=True,
    provider_factory=_QWEN_VIDEO,
    payload_builder=_video_payload,
    apply_video_token_fallback=True,
    dependency_columns=("_video_file_stat",),
    recipe_version="none",
)

SANDWICH_CASE = EmbeddingCaseSpec(
    name="sandwich",
    text_builder=build_sandwich_text,
    requires_video=True,
    provider_factory=_QWEN_VIDEO,
    payload_builder=_sandwich_payload,
    apply_video_token_fallback=True,
    dependency_columns=(
        "caption_clean",
        "caption_language",
        "caption_translation",
        "speech_transcription",
        "speech_language",
        "speech_translation",
        "music_id",
    ),
    recipe_version="sandwich_v1",
)

AUDIO_CASE = EmbeddingCaseSpec(
    name="audio",
    text_builder=build_audio_text,
    requires_video=False,
    provider_factory=_QWEN_TEXT,
    payload_builder=_audio_payload,
    apply_video_token_fallback=False,
    dependency_columns=(
        "speech_transcription",
        "speech_language",
        "speech_translation",
    ),
    recipe_version="audio_v1",
)


def _gemini_factory(settings, secrets) -> Provider:
    from modules.embeddings.gemini import GeminiMultimodalProvider

    if secrets is None or getattr(secrets, "gemini_api_key", None) is None:
        raise RuntimeError(
            "gemini provider requires secrets.gemini_api_key; "
            "set GEMINI_API_KEY and embeddings.gemini_enabled=true"
        )
    return GeminiMultimodalProvider(
        api_key=secrets.gemini_api_key,
        model=settings.embeddings.gemini_model,
        output_dim=settings.embeddings.gemini_output_dim,
        max_video_seconds=settings.embeddings.gemini_max_video_seconds,
        max_audio_seconds=settings.embeddings.gemini_max_audio_seconds,
        request_timeout_s=settings.embeddings.gemini_request_timeout_s,
        max_retries=settings.embeddings.gemini_max_retries,
    )


def _gemini_payload(clip, text, video_path, audio_path, fps, max_frames) -> dict:
    if audio_path is None:
        raise ValueError(
            "gemini payload requires audio_path; the runner must compute it "
            "from settings.paths.audio_dir, not reload config.toml"
        )
    return {"video_path": video_path, "audio_path": audio_path, "text": text}


GEMINI_CASE = EmbeddingCaseSpec(
    name="gemini",
    text_builder=build_gemini_text,
    requires_video=True,
    provider_factory=_gemini_factory,
    payload_builder=_gemini_payload,
    apply_video_token_fallback=False,
    dependency_columns=("_video_file_stat", "_audio_file_stat"),
    recipe_version="gemini_v1",
    requires=("gemini_enabled",),
)


CASE_REGISTRY: dict[str, EmbeddingCaseSpec] = {
    "video": VIDEO_CASE,
    "sandwich": SANDWICH_CASE,
    "audio": AUDIO_CASE,
    "gemini": GEMINI_CASE,
}


def default_cases(settings) -> tuple[str, ...]:
    """Return the set of cases whose ``requires`` gates all evaluate truthy.

    A case with ``requires=()`` is always returned; a case with
    ``requires=("gemini_enabled",)`` is returned only when
    ``settings.embeddings.gemini_enabled`` is truthy.
    """
    cases: list[str] = []
    for name, spec in CASE_REGISTRY.items():
        if all(getattr(settings.embeddings, flag, False) for flag in spec.requires):
            cases.append(name)
    return tuple(cases)


def case_config_identity(spec: EmbeddingCaseSpec, settings) -> str:
    """Stable identity string for a case's recipe + relevant settings.

    Fed into ``fingerprint.hash_text`` to produce the case's config_hash.
    Co-located with the spec definitions so changing a case's identity
    inputs lives next to the case itself.
    """
    parts = [
        f"case={spec.name}",
        f"provider={getattr(spec.provider_factory, '__name__', repr(spec.provider_factory))}",
        f"model={_os.path.basename(settings.paths.model_path)}",
        f"max_len={settings.embeddings.embed_max_length}",
        f"max_frames={settings.embeddings.adaptive_max_frames}",
        f"fps={settings.embeddings.adaptive_default_fps}",
        f"token_fallback={spec.apply_video_token_fallback}",
        f"text_recipe={spec.recipe_version}",
    ]
    if spec.name == "audio":
        parts.append(f"instruction={AUDIO_INSTRUCTION}")
    if spec.name == "gemini":
        # The common ``model=`` field above is the Qwen path and does not
        # distinguish Gemini model versions; include the real Gemini model
        # so operators changing gemini_model invalidate the config hash.
        parts.append(f"gemini_model={settings.embeddings.gemini_model}")
        parts.append(f"output_dim={settings.embeddings.gemini_output_dim}")
        parts.append(f"audio_bitrate={settings.audio_extraction.audio_bitrate_kbps}")
        parts.append(f"audio_sample_rate={settings.audio_extraction.audio_sample_rate_hz}")
        parts.append(f"max_video_s={settings.embeddings.gemini_max_video_seconds}")
        parts.append(f"max_audio_s={settings.embeddings.gemini_max_audio_seconds}")
    return "|".join(parts)
