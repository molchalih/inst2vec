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
from functools import partial

from modules.embeddings.providers import (
    LocalQwenProvider,
    Provider,
    ProviderRouter,
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
    """Secret bag threaded through the embedding stage.

    The Gemini provider reads ``gemini_api_key``; local providers ignore it.
    ``embedder_token`` gates the coordinator + pods. The remaining fields feed
    the RunPod pod fleet: ``runpod_api_key`` + ``coordinator_public_host`` gate
    auto-deploy (``fleet_enabled``) and ``huggingface_token`` is forwarded into
    each pod's env. Every field defaults to ``""``/``None`` so
    ``EmbeddingSecrets()`` is a valid no-arg call for tests / pipelines that
    don't need credentials.
    """

    gemini_api_key: str | None = None
    embedder_token: str = ""
    runpod_api_key: str = ""
    coordinator_public_host: str = ""
    huggingface_token: str = ""


@dataclass(frozen=True)
class EmbeddingCaseSpec:
    name: str
    text_builder: Callable[[object, object], str | None] | None
    requires_video: bool
    provider_factory: Callable[[object, object], Provider]
    payload_builder: Callable[
        [object, str | None, str | None, str | None, float | None, int | None], dict
    ]
    apply_video_token_fallback: bool
    # Names of Clip columns (or synthetic ``_video_file_stat`` /
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
    # When False, the embedder pod excludes this case from SERVED_CASES.
    # Default True so new cases must opt out of remote serving explicitly.
    served_remotely: bool = True
    # Human-readable name shown in the frontend case picker.
    display_label: str = ""
    # When False, the visualization stage still writes DB rows for this
    # case, but the JSON exporter excludes it from manifest.json + the
    # runs/ subtree. Flip to True to expose the case without recomputing.
    expose_to_viewer: bool = True
    # Cases sharing a non-empty backbone are served by ONE provider instance
    # in build_provider_router (e.g. all "qwen" cases share one model). Empty
    # = standalone provider via provider_factory. NOT part of
    # case_config_identity, so flipping it never invalidates stored rows.
    backbone: str = ""


# ── provider factories ───────────────────────────────────────────────────────


def qwen_provider(settings, secrets, *, with_frames: bool) -> Provider:
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
_QWEN_FACTORIES = {_QWEN_VIDEO, _QWEN_TEXT}


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
    display_label="Visual",
    backbone="qwen",
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
        "is_speech_detected",
        "speech_transcription",
        "speech_language",
        "speech_translation",
        "_audio_mir_row",
    ),
    recipe_version="sandwich_v3",
    display_label="Visual + Music",
    backbone="qwen",
)

AUDIO_CASE = EmbeddingCaseSpec(
    name="audio",
    text_builder=build_audio_text,
    requires_video=False,
    provider_factory=_QWEN_TEXT,
    payload_builder=_audio_payload,
    apply_video_token_fallback=False,
    dependency_columns=(
        "is_speech_detected",
        "speech_transcription",
        "speech_language",
        "speech_translation",
        "_audio_mir_row",
    ),
    recipe_version="audio_v3",
    display_label="Speech",
    backbone="qwen",
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
    # build_gemini_text feeds caption + speech text into the payload; the
    # file-stat sentinels alone don't notice translations / transcripts
    # changing, so include the same Clip columns the builder reads.
    dependency_columns=(
        "_video_file_stat",
        "_audio_file_stat",
        "caption_clean",
        "caption_language",
        "caption_translation",
        "is_speech_detected",
        "speech_transcription",
        "speech_language",
        "speech_translation",
    ),
    recipe_version="gemini_v2",
    requires=("gemini_enabled",),
    served_remotely=False,
    display_label="Gemini",
    expose_to_viewer=False,
)


def _maest_factory(settings, secrets) -> Provider:
    from pathlib import Path

    from modules.embeddings.maest import MaestProvider

    mir = settings.mir
    checkpoint = Path(mir.model_dir) / mir.maest_checkpoint
    return MaestProvider(
        checkpoint_path=checkpoint,
        input_op=mir.maest_input,
        sample_rate=mir.inference_sample_rate,
        min_samples=int(mir.maest_patch_seconds * mir.inference_sample_rate),
    )


def _maest_payload(clip, text, video_path, audio_path, fps, max_frames) -> dict:
    if audio_path is None:
        raise ValueError(
            "maest payload requires audio_path; the runner must compute it "
            "from settings.paths.audio_dir"
        )
    return {"audio_path": audio_path}


MAEST_CASE = EmbeddingCaseSpec(
    name="maest",
    text_builder=None,
    requires_video=False,
    provider_factory=_maest_factory,
    payload_builder=_maest_payload,
    apply_video_token_fallback=False,
    dependency_columns=("_audio_file_stat",),
    recipe_version="maest_v1",
    requires=(),
    served_remotely=False,
    display_label="Music (MAEST)",
)


CASE_REGISTRY: dict[str, EmbeddingCaseSpec] = {
    "video": VIDEO_CASE,
    "sandwich": SANDWICH_CASE,
    "audio": AUDIO_CASE,
    "gemini": GEMINI_CASE,
    "maest": MAEST_CASE,
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
    from pathlib import Path

    from modules.mir.checkpoints import read_sidecar_sha256

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
        parts.append(
            f"audio_sample_rate={settings.audio_extraction.audio_sample_rate_hz}"
        )
        parts.append(f"max_video_s={settings.embeddings.gemini_max_video_seconds}")
        parts.append(f"max_audio_s={settings.embeddings.gemini_max_audio_seconds}")
    if spec.name == "maest":
        mir = settings.mir
        pb_path = Path(mir.model_dir) / mir.maest_checkpoint
        min_samples = int(mir.maest_patch_seconds * mir.inference_sample_rate)
        parts.append(f"maest_checkpoint={mir.maest_checkpoint}")
        parts.append(f"input_op={mir.maest_input}")
        parts.append("output_op=StatefulPartitionedCall:7")
        parts.append("aggregation=concat_cls_dist_mean_v1")
        parts.append("patch_reduction=mean")
        parts.append(f"input_sample_rate={mir.inference_sample_rate}")
        parts.append(f"min_samples={min_samples}")
        parts.append(f"checkpoint_sha256={read_sidecar_sha256(pb_path)}")
    return "|".join(parts)


def remote_served_cases() -> tuple[str, ...]:
    """Case names a pod may be handed (served_remotely=True)."""
    return tuple(n for n, s in CASE_REGISTRY.items() if s.served_remotely)


def build_provider_router(settings, secrets, cases) -> ProviderRouter:
    """Build a ProviderRouter over ``cases``.

    Cases with backbone="qwen" whose ``provider_factory`` is one of the
    standard ``_QWEN_VIDEO``/``_QWEN_TEXT`` factories share ONE Qwen instance
    (with_frames=True). A backbone="qwen" case with an overridden
    ``provider_factory`` falls through to its own instance via that factory,
    which lets tests inject a stub without touching the shared model. The
    audio case is text-only, and a text-only payload is independent of the
    model's frame settings (see vendor format_model_input), so the shared
    instance yields the same vector the old text-config provider did — no
    stored-row recompute, hence audio's config_identity is intentionally left
    unchanged. Everything else builds via its own provider_factory. All
    builders are deferred (the router instantiates on first use).
    """
    instances: dict[str, Provider] = {}

    def _qwen() -> Provider:
        if "qwen" not in instances:
            instances["qwen"] = qwen_provider(settings, secrets, with_frames=True)
        return instances["qwen"]

    factories: dict[str, Callable[[], Provider]] = {}
    for name in cases:
        spec = CASE_REGISTRY[name]
        if spec.backbone == "qwen" and spec.provider_factory in _QWEN_FACTORIES:
            factories[name] = _qwen
        else:
            factories[name] = partial(spec.provider_factory, settings, secrets)
    return ProviderRouter(factories)
