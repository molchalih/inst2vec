"""Pure validation rules over the raw model output.

Returns ``(payload | None, status, warnings)`` where:
- ``status`` ∈ ``{"ok", "warn", "failed"}``.
- Hard fails (H1/H2/H3) return ``payload=None, status="failed"`` and the
  first triggered rule code in ``warnings``.
- Soft fails (S1..S8 except S5) return ``payload`` parsed, ``status="warn"``,
  and the list of triggered rule codes (sorted).
- Clean output returns ``status="ok"`` with ``warnings=[]``.

The validator is case-aware: the required-key set and the "role" keys
(observable-tag list, one-sentence reading, repertoire entry) are looked
up from ``modules.labels.cases.REGISTRY[case]`` so per-modality schemas
share the same rule codes without case-specific branches.
"""

from __future__ import annotations

import json
import re
from typing import Any

from core.config import LabelsSettings
from modules.labels.cases import REGISTRY, LabelCaseSpec

_HASHTAGLIKE = re.compile(r"^[#@]?\w+$")
_CONFIDENCES: frozenset[str] = frozenset({"high", "medium", "low"})


def clip_role_keys(spec: LabelCaseSpec) -> tuple[str, str]:
    """Public alias for the (observable_*_tags, one_sentence_*_reading) pair."""
    return spec.observable_key, spec.sentence_key


_clip_role_keys = clip_role_keys  # internal alias; keep call sites stable


def _cluster_repertoire_key(spec: LabelCaseSpec) -> str:
    return spec.repertoire_key


def validate(
    raw: str, labels: LabelsSettings, *, case: str
) -> tuple[dict | None, str, list[str]]:
    spec = REGISTRY[case]
    parsed = _parse(raw)
    if parsed is None:
        return None, "failed", ["H1"]
    if not isinstance(parsed, dict) or set(parsed.keys()) != set(
        spec.clip_required_keys
    ):
        return None, "failed", ["H2"]
    if not _shapes_ok(parsed, spec):
        return None, "failed", ["H3"]

    warnings = sorted(_soft_fail_codes(parsed, labels, spec))
    if warnings:
        return parsed, "warn", warnings
    return parsed, "ok", []


def _parse(raw: str) -> Any:
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return None


def _shapes_ok(parsed: dict, spec: LabelCaseSpec) -> bool:
    observable_key, sentence_key = _clip_role_keys(spec)
    obs = parsed[observable_key]
    aes = parsed["aesthetic_tags"]
    com = parsed["community_signalling_tags"]
    sentence = parsed[sentence_key]
    if (
        not isinstance(obs, list)
        or not isinstance(aes, list)
        or not isinstance(com, list)
    ):
        return False
    if not isinstance(sentence, str):
        return False
    for entry in obs:
        if not isinstance(entry, dict):
            return False
        if not isinstance(entry.get("tag"), str):
            return False
        if not isinstance(entry.get("evidence"), str):
            return False
    for block in (aes, com):
        for entry in block:
            if not isinstance(entry, dict):
                return False
            if not isinstance(entry.get("tag"), str):
                return False
            grounded = entry.get("grounded_in")
            if not isinstance(grounded, list) or not all(
                isinstance(g, str) for g in grounded
            ):
                return False
            if not isinstance(entry.get("confidence"), str):
                return False
    return True


def _soft_fail_codes(
    parsed: dict, labels: LabelsSettings, spec: LabelCaseSpec
) -> set[str]:
    codes: set[str] = set()
    observable_key, sentence_key = _clip_role_keys(spec)
    obs = parsed[observable_key]
    aes = parsed["aesthetic_tags"]
    com = parsed["community_signalling_tags"]
    sentence = parsed[sentence_key]

    # S1: count bounds per kind
    for block in (obs, aes, com):
        n = len(block)
        if n < labels.min_tags_per_kind or n > labels.max_tags_per_kind:
            codes.add("S1")
            break

    # S2: tag length bounds
    for block in (obs, aes, com):
        for entry in block:
            t = entry["tag"].strip()
            if len(t) < labels.min_tag_chars or len(t) > labels.max_tag_chars:
                codes.add("S2")
                break
        if "S2" in codes:
            break

    # S3: duplicate tags within a kind (case-insensitive, whitespace-normalised)
    for block in (obs, aes, com):
        norm = [_norm(e["tag"]) for e in block]
        if len(set(norm)) != len(norm):
            codes.add("S3")
            break

    # S4: hashtag-like tag (#x, @x, or single \w+ token)
    for block in (obs, aes, com):
        for entry in block:
            t = entry["tag"].strip()
            if t.startswith("#") or t.startswith("@") or _HASHTAGLIKE.match(t):
                codes.add("S4")
                break
        if "S4" in codes:
            break

    # S6: confidence enum
    for block in (aes, com):
        for entry in block:
            if entry["confidence"] not in _CONFIDENCES:
                codes.add("S6")
                break
        if "S6" in codes:
            break

    # S7: grounded_in references must hit an earlier section's tags
    observable_norms = {_norm(e["tag"]) for e in obs}
    aesthetic_norms = {_norm(e["tag"]) for e in aes}
    for entry in aes:
        for g in entry["grounded_in"]:
            if _norm(g) not in observable_norms:
                codes.add("S7")
                break
        if "S7" in codes:
            break
    if "S7" not in codes:
        prior = observable_norms | aesthetic_norms
        for entry in com:
            for g in entry["grounded_in"]:
                if _norm(g) not in prior:
                    codes.add("S7")
                    break
            if "S7" in codes:
                break

    # S8: sentence length bounds
    n = len(sentence.strip())
    if n < labels.min_sentence_chars or n > labels.max_sentence_chars:
        codes.add("S8")

    return codes


def _norm(s: str) -> str:
    return " ".join(s.strip().lower().split())


# ---------------------------------------------------------------------------
# Cluster validator
# ---------------------------------------------------------------------------

_RECURRENCES: frozenset[str] = frozenset({"dominant", "frequent", "occasional"})


def validate_cluster(
    raw: str, labels: LabelsSettings, *, case: str
) -> tuple[dict | None, str, list[str]]:
    spec = REGISTRY[case]
    parsed = _parse(raw)
    if parsed is None:
        return None, "failed", ["HC1"]
    if not isinstance(parsed, dict) or set(parsed.keys()) != set(
        spec.cluster_required_keys
    ):
        return None, "failed", ["HC2"]
    if not _cluster_shapes_ok(parsed, spec):
        return None, "failed", ["HC3"]

    warnings = sorted(_cluster_soft_fail_codes(parsed, labels, spec))
    if warnings:
        return parsed, "warn", warnings
    return parsed, "ok", []


def _cluster_shapes_ok(parsed: dict, spec: LabelCaseSpec) -> bool:
    repertoire_key = _cluster_repertoire_key(spec)
    if not isinstance(parsed["cluster_label"], str):
        return False
    if not isinstance(parsed["cluster_summary"], str):
        return False
    if not isinstance(parsed["boundary_notes"], str):
        return False
    if not isinstance(parsed["tool_tags"], list) or not all(
        isinstance(t, str) for t in parsed["tool_tags"]
    ):
        return False
    rep = parsed[repertoire_key]
    if not isinstance(rep, list):
        return False
    for entry in rep:
        if not isinstance(entry, dict):
            return False
        if not isinstance(entry.get("tag"), str):
            return False
        if not isinstance(entry.get("description"), str):
            return False
        if entry.get("recurrence") not in _RECURRENCES:
            return False
    aes = parsed["dominant_aesthetic_logic"]
    if not isinstance(aes, list):
        return False
    for entry in aes:
        if not isinstance(entry, dict):
            return False
        if not isinstance(entry.get("tag"), str):
            return False
        g = entry.get("grounded_in")
        if not isinstance(g, list) or not all(isinstance(x, str) for x in g):
            return False
        if not isinstance(entry.get("description"), str):
            return False
    for key in ("taste_signalling", "visibility_orientation"):
        block = parsed[key]
        if not isinstance(block, dict):
            return False
        if not isinstance(block.get("label"), str):
            return False
        if not isinstance(block.get("description"), str):
            return False
        if not isinstance(block.get("confidence"), str):
            return False
    var = parsed["internal_variations"]
    if not isinstance(var, list):
        return False
    for entry in var:
        if not isinstance(entry, dict):
            return False
        if not isinstance(entry.get("variation"), str):
            return False
        if not isinstance(entry.get("description"), str):
            return False
    return True


def _cluster_soft_fail_codes(
    parsed: dict, labels: LabelsSettings, spec: LabelCaseSpec
) -> set[str]:
    codes: set[str] = set()
    repertoire_key = _cluster_repertoire_key(spec)
    rep = parsed[repertoire_key]
    aes = parsed["dominant_aesthetic_logic"]
    tool_tags = parsed["tool_tags"]

    # SC1: count bounds for repertoire and aesthetic logic
    for block in (rep, aes):
        n = len(block)
        if n < labels.cluster_min_tags or n > labels.cluster_max_tags:
            codes.add("SC1")
            break

    # SC2: tag length bounds — reusing existing min/max_tag_chars
    for block in (rep, aes):
        for entry in block:
            t = entry["tag"].strip()
            if len(t) < labels.min_tag_chars or len(t) > labels.max_tag_chars:
                codes.add("SC2")
                break
        if "SC2" in codes:
            break

    # SC3: duplicate tags within a block (case/whitespace-normalised)
    for block in (rep, aes):
        norm = [_norm(e["tag"]) for e in block]
        if len(set(norm)) != len(norm):
            codes.add("SC3")
            break

    # SC4: aesthetic_logic.grounded_in must reference a repertoire tag
    rep_norms = {_norm(e["tag"]) for e in rep}
    for entry in aes:
        for g in entry["grounded_in"]:
            if _norm(g) not in rep_norms:
                codes.add("SC4")
                break
        if "SC4" in codes:
            break

    # SC5: confidence enum on taste_signalling and visibility_orientation
    for key in ("taste_signalling", "visibility_orientation"):
        if parsed[key]["confidence"] not in _CONFIDENCES:
            codes.add("SC5")
            break

    # SC6: sentence-shaped fields length bounds
    sentence_fields: list[str] = [parsed["cluster_summary"], parsed["boundary_notes"]]
    for entry in rep + aes:
        sentence_fields.append(entry["description"])
    for key in ("taste_signalling", "visibility_orientation"):
        sentence_fields.append(parsed[key]["description"])
    for entry in parsed["internal_variations"]:
        sentence_fields.append(entry["description"])
    for s in sentence_fields:
        n = len(s.strip())
        if (
            n < labels.cluster_min_sentence_chars
            or n > labels.cluster_max_sentence_chars
        ):
            codes.add("SC6")
            break

    # SC7: tool_tags non-empty and none may carry a hashtag/at-sign prefix
    if not tool_tags:
        codes.add("SC7")
    else:
        for t in tool_tags:
            s = t.strip()
            if s.startswith("#") or s.startswith("@"):
                codes.add("SC7")
                break

    return codes


# Public surface — pure validators, no module-level mutable state.
__all__ = ["clip_role_keys", "validate", "validate_cluster"]
