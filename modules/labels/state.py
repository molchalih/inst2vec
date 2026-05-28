"""Shared state constants and helpers for the labels module."""

from __future__ import annotations

from core.config import LabelsSettings
from core.fingerprint import stable_subset_payload

# Stage 1 (per-clip). One scope per case so each modality wipes / regenerates
# independently (mirrors how ``cluster_scope_for`` partitions stage 2).
STAGE_LABELS: str = "labels"


def clip_scope_for(case: str) -> str:
    return case


# Stage 2 (per-cluster). One scope per embedding_case so a single case
# can wipe + regenerate independently.
STAGE_CLUSTER_LABELS: str = "cluster_labels"


def cluster_scope_for(case: str) -> str:
    return case


# Fields hashed into the clip-pass fingerprint's ``config`` slot. Drift
# wipes the affected case's ``clip_labels`` rows. The per-case prompt is
# appended separately (see ``clip_labels_config_payload``) so a change to
# one case's prompt cannot drift another case's fingerprint. ``parallelism``
# is deliberately excluded — it is a runtime concurrency knob, not an
# output-affecting parameter.
_LABELS_CONFIG_FIELDS: tuple[str, ...] = (
    "model_path",
    "frame_count",
    "max_new_tokens",
    "generation_seed",
    "min_tags_per_kind",
    "max_tags_per_kind",
    "min_tag_chars",
    "max_tag_chars",
    "min_sentence_chars",
    "max_sentence_chars",
)

# Fields hashed into the cluster-pass fingerprint's ``config`` slot. A
# drift wipes the affected case's ``cluster_labels`` rows.
# ``cluster_max_attempts`` and ``parallelism`` are retry / runtime knobs
# and are excluded. The per-case cluster prompt is appended separately
# (see ``cluster_labels_config_payload``).
_CLUSTER_LABELS_CONFIG_FIELDS: tuple[str, ...] = (
    "model_path",
    "generation_seed",
    "cluster_max_new_tokens",
    "cluster_sample_token_budget",
    "cluster_max_clips_per_cluster",
    "cluster_max_clips_per_user",
    "cluster_min_tags",
    "cluster_max_tags",
    "cluster_min_sentence_chars",
    "cluster_max_sentence_chars",
    "min_tag_chars",
    "max_tag_chars",
)


def clip_labels_config_payload(labels: LabelsSettings, case: str) -> str:
    """Stable JSON string used as the clip-pass config-hash input for ``case``.

    Combines the case-agnostic generator knobs with the per-case stage-1
    prompt body so each case's fingerprint is isolated from prompt edits
    in other cases.
    """
    base = stable_subset_payload(labels, _LABELS_CONFIG_FIELDS)
    prompt = labels.case_prompts.get(case, "")
    return f"{base}|case_prompts.{case}={prompt}"


def cluster_labels_config_payload(labels: LabelsSettings, case: str) -> str:
    """Stable JSON string used as the cluster-pass config-hash input for ``case``."""
    base = stable_subset_payload(labels, _CLUSTER_LABELS_CONFIG_FIELDS)
    prompt = labels.cluster_case_prompts.get(case, "")
    return f"{base}|cluster_case_prompts.{case}={prompt}"
