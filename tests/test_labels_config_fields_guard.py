"""Guard: every output-affecting LabelsSettings field drifts a fingerprint.

If you add a knob to ``LabelsSettings`` that changes generated labels, add it to
``_LABELS_CONFIG_FIELDS`` or ``_CLUSTER_LABELS_CONFIG_FIELDS`` in
``modules/labels/state.py``. If it is a pure runtime / structural / prompt knob,
add it to ``RUNTIME_ALLOWLIST`` below with a reason. This test fails loudly on any
unclassified field so a new output-affecting setting cannot silently skip the
fingerprint and serve stale labels.
"""

from __future__ import annotations

from core.config import LabelsSettings
from modules.labels.state import (
    _CLUSTER_LABELS_CONFIG_FIELDS,
    _LABELS_CONFIG_FIELDS,
)

# Knobs that intentionally do NOT belong in a fingerprint config tuple.
RUNTIME_ALLOWLIST = {
    # Concurrency / throughput / retry — do not change output content.
    "parallelism",
    "batch_size",
    "max_attempts",
    "cluster_max_attempts",
    # Prompts are folded in per-case (clip_labels_config_payload /
    # cluster_labels_config_payload), not via the static tuples.
    "case_prompts",
    "cluster_case_prompts",
    # vLLM engine knobs — affect how the 30B is served, not the generated text
    # (decoding stays grammar-constrained + seeded).
    "cluster_gpu_memory_utilization",
    "cluster_max_model_len",
    "cluster_enforce_eager",
    # Clip-pass vLLM engine knobs — same rationale for the 8B clip tagger.
    "clip_gpu_memory_utilization",
    "clip_max_model_len",
    "clip_enforce_eager",
}


def test_every_labels_field_is_classified():
    fields = set(LabelsSettings.model_fields)
    fingerprinted = set(_LABELS_CONFIG_FIELDS) | set(_CLUSTER_LABELS_CONFIG_FIELDS)
    unclassified = fields - fingerprinted - RUNTIME_ALLOWLIST
    assert not unclassified, (
        f"Unclassified LabelsSettings fields: {sorted(unclassified)}. "
        "Add each to a config-fields tuple in modules/labels/state.py "
        "(if it changes generated labels) or to RUNTIME_ALLOWLIST (with a reason)."
    )
