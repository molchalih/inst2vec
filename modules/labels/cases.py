"""Per-case spec registry for the labels module.

Mirrors ``modules/embeddings/cases.py`` in shape: one frozen dataclass per
case declaring its modality, stage-1 input adapter, prompt keys, required
schema keys, and the upstream pipeline stages whose drift must invalidate
its stage-1 fingerprint. Case-specific behaviour in the labels pipeline
flows exclusively through this registry — no ``if case == "..."`` branches
in ``clip_pass.py`` / ``cluster_pass.py`` (SPEC §14.8).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from core.pipeline import Stage
from modules.captions.state import SCOPE_CAPTIONS
from modules.labels.inputs import (
    audio_input,
    maest_input,
    sandwich_input,
    video_input,
)
from modules.mir.state import SCOPE_MIR
from modules.speech.state import SCOPE_SPEECH

# Stage-1 input adapter contract. Receives ``(clip, mir_row, visual_payload)``
# — ``visual_payload`` is the ``ClipLabel(label_case="video")`` payload dict
# for sandwich/gemini (and ``None`` for every other case). Returns the
# textual input the generator will see, or ``None`` to signal the runner to
# mark ``status="failed"`` with the case-specific error string.
ClipInputBuilder = Callable[[object, object, dict | None], str | None]


@dataclass(frozen=True)
class LabelCaseSpec:
    name: str
    modality: str  # "visual" | "audio" | "music" | "multimodal"
    clip_input: ClipInputBuilder
    # Routes ``LabelsGenerator.run(video_path, prompt)`` (True, video case
    # only) versus ``LabelsGenerator.run_text(prompt, ...)`` (every other
    # case). The text-only branch already exists on the loaded Qwen
    # instance — no new dependency.
    clip_uses_video: bool
    clip_required_keys: frozenset[str]
    cluster_required_keys: frozenset[str]
    clip_prompt_key: str
    cluster_prompt_key: str
    # Upstream pipeline (stage, scope) pairs whose StageState row drift must
    # invalidate this case's stage-1 fingerprint. Scope matters: each upstream
    # stage seals its own scope key (e.g. SCOPE_SPEECH="all"), and looking up
    # a non-matching scope silently returns the empty-row hash — which would
    # make upstream drift go undetected.
    stage1_dependency_stages: tuple[tuple[Stage, str], ...]
    # Explicit role-key triplet so ``modules.labels.validation`` and
    # ``modules.visualization.export`` stop sniffing keys via ``startswith``.
    observable_key: str
    sentence_key: str
    repertoire_key: str
    # ── declarative ergonomics added 2026-05-28 ─────────────────────────────
    # Other label cases whose ``ClipLabel.payload`` this case consumes
    # via its ``clip_input`` adapter. Drives:
    #   * ``_data_hash_for_text`` / cluster pass per-clip input lookup;
    #   * dependency-hash composition (fold ``stage_dependency_hash(LABELS, dep)``);
    #   * topo-ordering of cases in ``pipeline.run``.
    # Each entry must reference a key present in ``REGISTRY``.
    consumes_label_cases: tuple[str, ...] = ()
    # Error string written to ``ClipLabel.error`` + ``bump_failure`` when this
    # case's ``clip_input`` adapter returns ``None``. ``None`` here means the
    # adapter never returns ``None`` (e.g. the video case, which is short-
    # circuited before the adapter runs).
    none_input_error: str | None = None
    # Whether this case runs the per-clip stage-1 pass (Qwen3-VL writes
    # ``ClipLabel`` rows). The visual case (video) must, because it is the
    # only place frames are reduced to text — the cluster pass has no other
    # access to frame data. The text rephrasing cases (sandwich, audio,
    # maest, gemini) skip stage 1 and consume their raw signals (captions,
    # speech, MIR, plus the video ClipLabel for sandwich/gemini) directly
    # in the cluster pass. Stage-1-skipped cases keep ``clip_input`` as the
    # adapter producing one cluster-pass per-clip text block.
    runs_clip_pass: bool = True


_VIDEO_CLIP_KEYS: frozenset[str] = frozenset(
    {
        "observable_visual_tags",
        "aesthetic_tags",
        "community_signalling_tags",
        "one_sentence_visual_reading",
    }
)

_AUDIO_CLIP_KEYS: frozenset[str] = frozenset(
    {
        "observable_audio_tags",
        "aesthetic_tags",
        "community_signalling_tags",
        "one_sentence_audio_reading",
    }
)

_MAEST_CLIP_KEYS: frozenset[str] = frozenset(
    {
        "observable_music_tags",
        "aesthetic_tags",
        "community_signalling_tags",
        "one_sentence_music_reading",
    }
)

_SANDWICH_CLIP_KEYS: frozenset[str] = frozenset(
    {
        "observable_multimodal_tags",
        "aesthetic_tags",
        "community_signalling_tags",
        "one_sentence_multimodal_reading",
    }
)

_COMMON_CLUSTER_KEYS: frozenset[str] = frozenset(
    {
        "cluster_label",
        "cluster_summary",
        "dominant_aesthetic_logic",
        "taste_signalling",
        "visibility_orientation",
        "internal_variations",
        "boundary_notes",
        "tool_tags",
    }
)


def _cluster_keys(repertoire_key: str) -> frozenset[str]:
    return _COMMON_CLUSTER_KEYS | {repertoire_key}


VIDEO_CASE = LabelCaseSpec(
    name="video",
    modality="visual",
    clip_input=video_input,
    clip_uses_video=True,
    clip_required_keys=_VIDEO_CLIP_KEYS,
    cluster_required_keys=_cluster_keys("dominant_visual_repertoire"),
    clip_prompt_key="video",
    cluster_prompt_key="video",
    stage1_dependency_stages=(),
    observable_key="observable_visual_tags",
    sentence_key="one_sentence_visual_reading",
    repertoire_key="dominant_visual_repertoire",
    consumes_label_cases=(),
    none_input_error=None,
)


_SPEECH_DEP: tuple[Stage, str] = (Stage.SPEECH, SCOPE_SPEECH)
_CAPTIONS_DEP: tuple[Stage, str] = (Stage.CAPTIONS, SCOPE_CAPTIONS)
_MIR_DEP: tuple[Stage, str] = (Stage.MIR, SCOPE_MIR)

AUDIO_CASE = LabelCaseSpec(
    name="audio",
    modality="audio",
    clip_input=audio_input,
    clip_uses_video=False,
    clip_required_keys=_AUDIO_CLIP_KEYS,
    cluster_required_keys=_cluster_keys("dominant_audio_repertoire"),
    clip_prompt_key="audio",
    cluster_prompt_key="audio",
    stage1_dependency_stages=(_SPEECH_DEP, _MIR_DEP),
    observable_key="observable_audio_tags",
    sentence_key="one_sentence_audio_reading",
    repertoire_key="dominant_audio_repertoire",
    consumes_label_cases=(),
    none_input_error="no_input",
    runs_clip_pass=False,
)

SANDWICH_CASE = LabelCaseSpec(
    name="sandwich",
    modality="multimodal",
    clip_input=sandwich_input,
    clip_uses_video=False,
    clip_required_keys=_SANDWICH_CLIP_KEYS,
    cluster_required_keys=_cluster_keys("dominant_multimodal_repertoire"),
    clip_prompt_key="sandwich",
    cluster_prompt_key="sandwich",
    stage1_dependency_stages=(_CAPTIONS_DEP, _SPEECH_DEP, _MIR_DEP),
    observable_key="observable_multimodal_tags",
    sentence_key="one_sentence_multimodal_reading",
    repertoire_key="dominant_multimodal_repertoire",
    consumes_label_cases=("video",),
    none_input_error="missing_video_label",
    runs_clip_pass=False,
)

MAEST_CASE = LabelCaseSpec(
    name="maest",
    modality="music",
    clip_input=maest_input,
    clip_uses_video=False,
    clip_required_keys=_MAEST_CLIP_KEYS,
    cluster_required_keys=_cluster_keys("dominant_music_repertoire"),
    clip_prompt_key="maest",
    cluster_prompt_key="maest",
    stage1_dependency_stages=(_MIR_DEP,),
    observable_key="observable_music_tags",
    sentence_key="one_sentence_music_reading",
    repertoire_key="dominant_music_repertoire",
    consumes_label_cases=(),
    none_input_error="no_music",
    runs_clip_pass=False,
)

GEMINI_CASE = LabelCaseSpec(
    name="gemini",
    modality="multimodal",
    clip_input=sandwich_input,
    clip_uses_video=False,
    clip_required_keys=_SANDWICH_CLIP_KEYS,
    cluster_required_keys=_cluster_keys("dominant_multimodal_repertoire"),
    clip_prompt_key="gemini",
    cluster_prompt_key="gemini",
    stage1_dependency_stages=(_CAPTIONS_DEP, _SPEECH_DEP, _MIR_DEP),
    observable_key="observable_multimodal_tags",
    sentence_key="one_sentence_multimodal_reading",
    repertoire_key="dominant_multimodal_repertoire",
    consumes_label_cases=("video",),
    none_input_error="missing_video_label",
    runs_clip_pass=False,
)


REGISTRY: dict[str, LabelCaseSpec] = {
    "video": VIDEO_CASE,
    "sandwich": SANDWICH_CASE,
    "audio": AUDIO_CASE,
    "gemini": GEMINI_CASE,
    "maest": MAEST_CASE,
}
