"""Per-case JSON Schema for cluster-label grammar-constrained decoding.

Single source of truth mirroring ``validation._cluster_shapes_ok`` and the
cluster soft-fail bounds. Fed to the xgrammar logits processor in
``core.vendor.qwen3_text``. Tag-quality rules the grammar cannot express
(connector-word squash, non-ASCII) are enforced post-parse as HC5/HC6 in
``modules.labels.validation``.
"""

from __future__ import annotations

from core.config import LabelsSettings
from modules.labels.cases import LabelCaseSpec
from modules.labels.validation import CLUSTER_LABEL_MAX_CHARS

_RECURRENCE = ["dominant", "frequent", "occasional"]
_CONFIDENCE = ["high", "medium", "low"]


def _tag_schema(labels: LabelsSettings) -> dict:
    # At most ``cluster_tag_max_words`` space-separated lowercase/digit words
    # (hyphenated compounds allowed), no leading/trailing space.
    #
    # The grammar is deliberately LOOSER than the HC5 post-parse rule: it bounds
    # word count, length and charset, but cannot forbid connector words
    # (with/for/and/of/...) mid-tag. Forbidding a whole word from being a
    # connector needs a negative lookahead, which xgrammar (0.2.1) rejects at
    # compile time ("Lookahead is not supported yet"), and a lookahead-free
    # complement regex would be unmaintainable. So the connector-word blacklist
    # lives in HC5 (validation._cluster_tag_quality_codes); a violation triggers
    # a seed-escalating retry. Do NOT loosen HC5 to match this looser pattern.
    extra_words = max(labels.cluster_tag_max_words - 1, 0)
    pattern = (
        r"^[a-z0-9]+(-[a-z0-9]+)*"
        rf"( [a-z0-9]+(-[a-z0-9]+)*){{0,{extra_words}}}$"
    )
    return {
        "type": "string",
        "maxLength": labels.cluster_tag_max_chars,
        "pattern": pattern,
    }


def _desc_schema(max_chars: int) -> dict:
    return {"type": "string", "maxLength": max_chars}


def clip_schema(spec: LabelCaseSpec, labels: LabelsSettings) -> dict:
    """Per-case JSON Schema for clip-label grammar-constrained decoding.

    Encodes ONLY the hard ``validation._shapes_ok`` rules: exact key set,
    object shapes, value types, and the ``confidence`` enum. The soft count
    / length bounds (S1-S6) are deliberately left out so they remain
    post-parse warnings — ``ClipLabel.warnings`` feeds the cluster-pass clip
    ranking and must stay meaningful.
    """
    tag = {"type": "string"}
    observable_item = {
        "type": "object",
        "additionalProperties": False,
        "required": ["tag", "evidence"],
        "properties": {"tag": tag, "evidence": {"type": "string"}},
    }
    grounded_item = {
        "type": "object",
        "additionalProperties": False,
        "required": ["tag", "grounded_in", "confidence"],
        "properties": {
            "tag": tag,
            "grounded_in": {"type": "array", "items": tag},
            "confidence": {"type": "string", "enum": list(_CONFIDENCE)},
        },
    }
    props = {
        spec.observable_key: {"type": "array", "items": observable_item},
        "aesthetic_tags": {"type": "array", "items": grounded_item},
        "community_signalling_tags": {"type": "array", "items": grounded_item},
        spec.sentence_key: {"type": "string"},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(spec.clip_required_keys),
        "properties": props,
    }


def cluster_schema(spec: LabelCaseSpec, labels: LabelsSettings) -> dict:
    tag = _tag_schema(labels)
    desc = _desc_schema(labels.cluster_max_sentence_chars)
    array_bounds = {
        "minItems": labels.cluster_min_tags,
        "maxItems": labels.cluster_max_tags,
    }

    rep_item = {
        "type": "object",
        "additionalProperties": False,
        "required": ["tag", "description", "recurrence"],
        "properties": {
            "tag": tag,
            "description": desc,
            "recurrence": {"type": "string", "enum": _RECURRENCE},
        },
    }
    aes_item = {
        "type": "object",
        "additionalProperties": False,
        "required": ["tag", "grounded_in", "description"],
        "properties": {
            "tag": tag,
            "grounded_in": {"type": "array", "items": tag},
            "description": desc,
        },
    }
    confidence_obj = {
        "type": "object",
        "additionalProperties": False,
        "required": ["label", "description", "confidence"],
        "properties": {
            "label": _desc_schema(labels.cluster_max_sentence_chars),
            "description": desc,
            "confidence": {"type": "string", "enum": _CONFIDENCE},
        },
    }
    variation_item = {
        "type": "object",
        "additionalProperties": False,
        "required": ["variation", "description"],
        "properties": {
            "variation": _desc_schema(labels.cluster_max_sentence_chars),
            "description": desc,
        },
    }
    props = {
        "cluster_label": {"type": "string", "maxLength": CLUSTER_LABEL_MAX_CHARS},
        "cluster_summary": _desc_schema(labels.cluster_summary_max_chars),
        spec.repertoire_key: {"type": "array", "items": rep_item, **array_bounds},
        "dominant_aesthetic_logic": {
            "type": "array",
            "items": aes_item,
            **array_bounds,
        },
        "taste_signalling": confidence_obj,
        "visibility_orientation": confidence_obj,
        "internal_variations": {"type": "array", "items": variation_item},
        "boundary_notes": desc,
        "tool_tags": {"type": "array", "items": tag, "minItems": 1},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(spec.cluster_required_keys),
        "properties": props,
    }
