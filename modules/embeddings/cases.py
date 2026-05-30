"""Embedding case specifications.

``embedding_case`` is the stable identity of a complete embedding recipe
(modality + provider + model family/version + prompt + input-building
logic). It is the only idempotence boundary the embeddings package
recognizes — completion is derived from rows in ClipEmbedding /
UserEmbedding keyed by ``(clip_id|user_id, embedding_case)``.

Note: ``"auditory"`` is a raw-waveform MAEST acoustic embedding;
``"spoken"`` is a text embedding over the speech transcript only and
``"textual"`` over the caption only.
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
    build_sandwich_text,
    build_spoken_text,
    build_textual_text,
)

# Every payload builder accepts ``audio_path`` and ignores it unless its case
# reads audio (auditory), so the runner can pass it uniformly without
# branching by case name.

SPOKEN_INSTRUCTION = (
    "Represent the spoken content of this video: what is said and its topics."
)

TEXTUAL_INSTRUCTION = (
    "Represent the written caption of this post: its topics, tone, and framing."
)


@dataclass(frozen=True)
class EmbeddingCaseSpec:
    name: str
    text_builder: Callable[[object, object], str | None] | None
    requires_video: bool
    provider_factory: Callable[[object], Provider]
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


def qwen_provider(settings, *, with_frames: bool) -> Provider:
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
    def factory(settings):
        return qwen_provider(settings, with_frames=with_frames)

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


def _spoken_payload(clip, text, video_path, audio_path, fps, max_frames) -> dict:
    return {"text": text, "instruction": SPOKEN_INSTRUCTION}


def _textual_payload(clip, text, video_path, audio_path, fps, max_frames) -> dict:
    return {"text": text, "instruction": TEXTUAL_INSTRUCTION}


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
    display_label="Combined",
    backbone="qwen",
)

SPOKEN_CASE = EmbeddingCaseSpec(
    name="spoken",
    text_builder=build_spoken_text,
    requires_video=False,
    provider_factory=_QWEN_TEXT,
    payload_builder=_spoken_payload,
    apply_video_token_fallback=False,
    dependency_columns=(
        "is_speech_detected",
        "speech_transcription",
        "speech_language",
        "speech_translation",
    ),
    recipe_version="spoken_v1",
    display_label="Spoken",
    backbone="qwen",
)

TEXTUAL_CASE = EmbeddingCaseSpec(
    name="textual",
    text_builder=build_textual_text,
    requires_video=False,
    provider_factory=_QWEN_TEXT,
    payload_builder=_textual_payload,
    apply_video_token_fallback=False,
    dependency_columns=(
        "caption_clean",
        "caption_text",
        "caption_language",
        "caption_translation",
    ),
    recipe_version="textual_v1",
    display_label="Textual",
    backbone="qwen",
)


def _maest_factory(settings) -> Provider:
    from pathlib import Path

    from modules.embeddings.maest import MaestProvider

    mir = settings.mir
    onnx_path = Path(mir.model_dir) / mir.maest_onnx_checkpoint
    return MaestProvider(
        onnx_path=onnx_path,
        sample_rate=mir.inference_sample_rate,
    )


def _maest_payload(clip, text, video_path, audio_path, fps, max_frames) -> dict:
    if audio_path is None:
        raise ValueError(
            "maest payload requires audio_path; the runner must compute it "
            "from settings.paths.audio_dir"
        )
    return {"audio_path": audio_path}


AUDITORY_CASE = EmbeddingCaseSpec(
    name="auditory",
    text_builder=None,
    requires_video=False,
    provider_factory=_maest_factory,
    payload_builder=_maest_payload,
    apply_video_token_fallback=False,
    dependency_columns=("_audio_file_stat",),
    recipe_version="maest_v1",
    requires=(),
    display_label="Auditory",
)


CASE_REGISTRY: dict[str, EmbeddingCaseSpec] = {
    "video": VIDEO_CASE,
    "sandwich": SANDWICH_CASE,
    "auditory": AUDITORY_CASE,
    "spoken": SPOKEN_CASE,
    "textual": TEXTUAL_CASE,
}


def default_cases(settings) -> tuple[str, ...]:
    """Return the set of cases whose ``requires`` gates all evaluate truthy.

    A case with ``requires=()`` is always returned; a case gated on a
    ``settings.embeddings`` flag is returned only when that flag is truthy.
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
    if spec.name == "spoken":
        parts.append(f"instruction={SPOKEN_INSTRUCTION}")
    if spec.name == "textual":
        parts.append(f"instruction={TEXTUAL_INSTRUCTION}")
    if spec.name == "auditory":
        mir = settings.mir
        pb_path = Path(mir.model_dir) / mir.maest_checkpoint
        # backend=onnx is the deliberate migration marker: it flips the
        # config_hash so the runner wipes + re-extracts every maest row once.
        # The .pb sidecar sha256 stays the content anchor — the .onnx is the
        # numerically-equivalent conversion of that same .pb and carries no
        # sidecar (excluded from the MIR fingerprint).
        parts.append("backend=onnx")
        parts.append(f"maest_onnx_checkpoint={mir.maest_onnx_checkpoint}")
        # layer_4_tokens is the ONNX name for the .pb's StatefulPartitionedCall:7
        # (confirmed by parity); pb_equiv documents that the produced vector is
        # the same tensor the prior .pb-backed rows used.
        parts.append("output_op=layer_4_tokens")
        parts.append("pb_equiv=StatefulPartitionedCall:7")
        parts.append("aggregation=concat_cls_dist_mean_v1")
        parts.append("patch_reduction=mean")
        parts.append("patch_frames=1876")
        parts.append("patch_hop=1875")
        parts.append(f"input_sample_rate={mir.inference_sample_rate}")
        parts.append(f"checkpoint_sha256={read_sidecar_sha256(pb_path)}")
    return "|".join(parts)


def build_provider_router(settings, cases) -> ProviderRouter:
    """Build a ProviderRouter over ``cases``.

    Cases with backbone="qwen" whose ``provider_factory`` is one of the
    standard ``_QWEN_VIDEO``/``_QWEN_TEXT`` factories share ONE Qwen instance
    (with_frames=True). A backbone="qwen" case with an overridden
    ``provider_factory`` falls through to its own instance via that factory,
    which lets tests inject a stub without touching the shared model. The
    text-only cases (spoken/textual) feed a text-only payload that is
    independent of the model's frame settings (see vendor format_model_input),
    so the shared instance yields the same vector a standalone text-config
    provider would. Everything else builds via its own provider_factory. All
    builders are deferred (the router instantiates on first use).
    """
    instances: dict[str, Provider] = {}

    def _qwen() -> Provider:
        if "qwen" not in instances:
            instances["qwen"] = qwen_provider(settings, with_frames=True)
        return instances["qwen"]

    factories: dict[str, Callable[[], Provider]] = {}
    for name in cases:
        spec = CASE_REGISTRY[name]
        if spec.backbone == "qwen" and spec.provider_factory in _QWEN_FACTORIES:
            factories[name] = _qwen
        else:
            factories[name] = partial(spec.provider_factory, settings)
    return ProviderRouter(factories)
