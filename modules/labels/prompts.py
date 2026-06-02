"""Per-case prompt selection for the labels module.

Stage-1 prompts live under ``settings.labels.case_prompts`` and stage-2
cluster prompts under ``settings.labels.cluster_case_prompts``. Each edit
drifts the affected case's fingerprint and triggers a per-case wipe (see
``modules.labels.state.clip_labels_config_payload`` /
``cluster_labels_config_payload``).
"""

from __future__ import annotations

from core.config import LabelsSettings


def prompt_for(labels: LabelsSettings, *, case: str) -> str:
    """Verbatim clip prompt body for ``case``, stripped of surrounding whitespace.

    Raises ``ValueError`` when the case has no entry under
    ``[labels.case_prompts]`` — surfaces the SPEC §5.6 contract explicitly
    at first use rather than at config load time, so the visual-only flow
    continues to work without a config rewrite.
    """
    try:
        body = labels.case_prompts[case]
    except KeyError as exc:
        raise ValueError(f"missing labels.case_prompts.{case}") from exc
    return body.strip()


def prompt_for_cluster(labels: LabelsSettings, *, case: str) -> str:
    """Verbatim cluster prompt body for ``case``, stripped of surrounding whitespace."""
    try:
        body = labels.cluster_case_prompts[case]
    except KeyError as exc:
        raise ValueError(f"missing labels.cluster_case_prompts.{case}") from exc
    return body.strip()


def naming_prompt_for(labels: LabelsSettings, *, case: str) -> str:
    """Lead-in instructions for the global cluster-label naming pass.

    Case-agnostic: the naming pass coordinates labels across all clusters of a
    case from their summaries, so the same instruction body applies to every
    modality. ``case`` is accepted for call-site symmetry with the other prompt
    selectors and to leave room for a future per-case override.
    """
    return labels.cluster_naming_prompt.strip()
