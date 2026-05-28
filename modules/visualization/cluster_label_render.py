"""Translate ClusterLabel rows into the frontend ``label`` JSON block.

Owns the cluster soft-warning code → human-readable string map and the
shape of the per-cluster ``label`` payload. The block is case-agnostic
at the render boundary: the case-specific ``dominant_*_repertoire`` key
is flattened to ``repertoire`` and a ``modality`` string is stamped on
so the frontend can pick a section heading without knowing about cases.
"""

from __future__ import annotations

from core.database import ClusterLabel
from modules.labels.cases import LabelCaseSpec

_CLUSTER_WARNING_LABELS: dict[str, str] = {
    "SC1": "tag_count_out_of_range",
    "SC2": "tag_length_out_of_range",
    "SC3": "duplicate_tag_within_kind",
    "SC4": "ungrounded_tag_reference",
    "SC5": "invalid_confidence",
    "SC6": "sentence_length_out_of_range",
    "SC7": "invalid_tool_tags",
}


def cluster_warning_label(code: str) -> str:
    return _CLUSTER_WARNING_LABELS.get(code, code)


def display_label_for(row: ClusterLabel | None, fallback: str) -> str:
    """Use the generated cluster_label when stage-2 succeeded; else placeholder."""
    if row is None or row.status != "success":
        return fallback
    payload = row.payload or {}
    return payload.get("cluster_label") or fallback


def _repertoire_key(spec: LabelCaseSpec) -> str:
    """Locate the unique ``dominant_*_repertoire`` key in the cluster schema."""
    for key in spec.cluster_required_keys:
        if key.startswith("dominant_") and key.endswith("_repertoire"):
            return key
    # No repertoire key declared — fall back to the visual key name so the
    # render still works on legacy payloads.
    return "dominant_visual_repertoire"


def render_label_block(row: ClusterLabel | None, spec: LabelCaseSpec) -> dict | None:
    """Build the per-cluster ``label`` block; return None to omit it entirely.

    The case-specific ``dominant_*_repertoire`` key is flattened to
    ``repertoire`` and ``dominant_aesthetic_logic`` to ``aesthetic_logic``
    so the frontend schema is one shape regardless of case. ``modality``
    is stamped from the spec for section-heading wiring.
    """
    if row is None or row.status != "success":
        return None
    p = row.payload or {}
    rep_key = _repertoire_key(spec)
    return {
        "label": p.get("cluster_label", ""),
        "summary": p.get("cluster_summary", ""),
        "modality": spec.modality,
        "repertoire": p.get(rep_key, []),
        "aesthetic_logic": p.get("dominant_aesthetic_logic", []),
        "taste_signalling": p.get("taste_signalling", {}),
        "visibility_orientation": p.get("visibility_orientation", {}),
        "internal_variations": p.get("internal_variations", []),
        "boundary_notes": p.get("boundary_notes", ""),
        "tool_tags": p.get("tool_tags", []),
        "validation": row.validation or "ok",
        "warnings": [cluster_warning_label(c) for c in (row.warnings or [])],
    }
