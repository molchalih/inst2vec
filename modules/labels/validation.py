"""Pure validation rules over the raw model output.

Returns ``(payload | None, status, warnings)`` where:
- ``status`` ∈ ``{"ok", "warn", "failed"}``.
- Hard fails (H1/H2/H3) return ``payload=None, status="failed"`` and the
  first triggered rule code in ``warnings``. The cluster validator uses codes
  HC1–HC6 (HC5 = tag squash, HC6 = non-ASCII/emoji).
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
from collections.abc import Collection
from typing import Any

from core.config import LabelsSettings
from modules.labels.cases import REGISTRY, LabelCaseSpec

_HASHTAGLIKE = re.compile(r"^[#@]?\w+$")
_CONFIDENCES: frozenset[str] = frozenset({"high", "medium", "low"})
_NON_ASCII = re.compile(r"[^\x00-\x7f]")
_CONNECTOR_WORDS: frozenset[str] = frozenset(
    {"with", "for", "and", "of", "to", "in", "on", "the", "a", "an", "&"}
)


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


_CURLY_QUOTE_MAP = str.maketrans(
    {
        "“": '"',  # LEFT DOUBLE QUOTATION MARK
        "”": '"',  # RIGHT DOUBLE QUOTATION MARK
        "‘": "'",  # LEFT SINGLE QUOTATION MARK
        "’": "'",  # RIGHT SINGLE QUOTATION MARK
    }
)


def _parse(raw: str) -> Any:
    """Strict-first, repair-fallback JSON parsing.

    Qwen3-VL output is mostly-valid JSON with small recurring defects
    (curly quotes, trailing commas, missing object braces before
    array close, occasional markdown fence wrap). The pipeline tries:
    1. strict ``json.loads`` on curly-quote-normalised text;
    2. bracket-balanced version of the same text (handles
       "object opened, never closed before ``]``" which is the most
       common Qwen3-VL syntax bug and one ``json-repair`` cannot
       recover cleanly — it drops every key after the missing brace);
    3. ``json-repair`` as a final fallback.
    """
    normalised = raw.translate(_CURLY_QUOTE_MAP)
    try:
        return _strip_dict_keys(json.loads(normalised))
    except (ValueError, TypeError):
        pass
    balanced = _balance_brackets(normalised)
    if balanced != normalised:
        try:
            return _strip_dict_keys(json.loads(balanced))
        except (ValueError, TypeError):
            pass
    try:
        from json_repair import repair_json

        repaired = repair_json(balanced, return_objects=True)
    except Exception:
        return None
    # ``json-repair`` will happily turn pure garbage (e.g. ``"not json"``)
    # into an empty string or empty dict. Treat those as parse failures
    # rather than letting them slip through to the key-shape check —
    # otherwise HC1 (non-JSON) silently becomes HC2 (wrong keys).
    if isinstance(repaired, dict) and repaired:
        return _strip_dict_keys(repaired)
    if isinstance(repaired, list) and repaired:
        return _strip_dict_keys(repaired)
    return None


def _balance_brackets(s: str) -> str:
    """Insert missing ``}`` before ``]`` when an object is left unclosed.

    The model sometimes emits arrays like
    ``[{ "k": "v"}, { "k": "v"\\n ]`` — last entry's ``}`` is absent.
    Walk the string respecting JSON string literals; when a ``]`` is
    seen with one or more ``{`` still on top of the bracket stack,
    inject the missing ``}``\\ s first. Also flushes any unclosed
    brackets at end-of-string. Leaves valid input unchanged.
    """
    out: list[str] = []
    stack: list[str] = []
    in_str = False
    escape = False
    for ch in s:
        if in_str:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
        elif ch in "[{":
            stack.append(ch)
            out.append(ch)
        elif ch == "}":
            if stack and stack[-1] == "{":
                stack.pop()
            out.append(ch)
        elif ch == "]":
            while stack and stack[-1] == "{":
                out.append("}")
                stack.pop()
            if stack and stack[-1] == "[":
                stack.pop()
            out.append(ch)
        else:
            out.append(ch)
    while stack:
        out.append("}" if stack.pop() == "{" else "]")
    return "".join(out)


def _strip_dict_keys(obj: Any) -> Any:
    """Recursively strip whitespace from string keys.

    Qwen3-VL occasionally emits keys like ``" description"`` (leading
    space) — syntactically valid JSON but semantically wrong; the
    shape validator's ``entry.get("description")`` then misses the
    field. Normalising at parse time is safer than per-callsite
    `.get(key.strip(), entry.get(" " + key))` fallback hacks.
    """
    if isinstance(obj, dict):
        return {
            (k.strip() if isinstance(k, str) else k): _strip_dict_keys(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_strip_dict_keys(x) for x in obj]
    return obj


def format_failure_error(code: str, raw: str) -> str:
    """Diagnostic ``error`` payload for terminal H/HC failures.

    The validator throws away the raw model output once it has assigned
    a rule code; we re-attach the raw text so the DB row tells us *why*
    the model failed (truncation, markdown fence, preamble, mid-string
    syntax error, wrong nested type) without keeping the full transcript
    around. Cluster prompts are now capped at <~3.3k chars per emission;
    a 5000-char threshold captures any well-behaved failure in full.
    """
    n = len(raw)
    full_body_threshold = 5000
    if n <= full_body_threshold:
        body = raw.replace("\n", "\\n")
        return f"{code} len={n} body={body!r}"
    head = raw[:1500].replace("\n", "\\n")
    tail = raw[-1500:].replace("\n", "\\n")
    return f"{code} len={n} head={head!r} tail={tail!r}"


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


def _fuzzy_normalize_keys(parsed: dict, required: Collection[str]) -> dict:
    """Map model-typo'd top-level keys to the closest required key.

    Greedy decoding deterministically produces near-misses like
    ``visibility_orienation`` (missing ``t``), ``internal_varyations``
    (extra ``y``), ``dominant_aesthetic_logical`` (grammar-correction),
    ``taste_signaling`` (US spelling) — all unambiguously the intended
    field. ``difflib.get_close_matches`` with a 0.85 cutoff matches
    these without conflating distinct fields (the required key set
    has pairwise similarity well below 0.85).
    """
    import difflib

    required_set = set(required)
    out: dict = {}
    for k, v in parsed.items():
        if k in required_set:
            out[k] = v
            continue
        matches = difflib.get_close_matches(k, required, n=1, cutoff=0.85)
        out[matches[0] if matches else k] = v
    return out


_CLUSTER_LABEL_MAX_CHARS = 40


def _cluster_tag_quality_codes(
    parsed: dict, labels: LabelsSettings, spec: LabelCaseSpec
) -> list[str]:
    """Hard-fail codes for tag quality the grammar can't express.

    HC6 (non-ASCII / emoji) takes priority over HC5 (oversize squash: more than
    ``cluster_tag_max_words`` words). Only the WORD count is a hard fail because
    vLLM's structured decoding enforces the tag pattern's word cap, so HC5 is a
    true (rarely-firing) backstop. It does NOT hard-fail on character length:
    vLLM does NOT enforce the schema's ``maxLength``, so a char hard-fail just
    forces doomed retries on legitimate longer phrases; over-long tags are a
    SOFT warn (SC2), and connector words a SOFT warn (SC9), in
    ``_cluster_soft_fail_codes``. Returns ``[]`` when all tags are clean.
    """
    repertoire_key = _cluster_repertoire_key(spec)
    tags: list[str] = [e["tag"] for e in parsed[repertoire_key]]
    tags += [e["tag"] for e in parsed["dominant_aesthetic_logic"]]
    tags += parsed["tool_tags"]
    for t in tags:
        if _NON_ASCII.search(t):
            return ["HC6"]
    for t in tags:
        if len(t.strip().split()) > labels.cluster_tag_max_words:
            return ["HC5"]
    return []


def validate_cluster(
    raw: str, labels: LabelsSettings, *, case: str
) -> tuple[dict | None, str, list[str]]:
    spec = REGISTRY[case]
    parsed = _parse(raw)
    if parsed is None:
        return None, "failed", ["HC1"]
    if isinstance(parsed, dict):
        parsed = _fuzzy_normalize_keys(parsed, spec.cluster_required_keys)
    if not isinstance(parsed, dict) or set(parsed.keys()) != set(
        spec.cluster_required_keys
    ):
        return None, "failed", ["HC2"]
    if not _cluster_shapes_ok(parsed, spec):
        return None, "failed", ["HC3"]
    # HC4 — cluster_label length cap. Hard fail so the retry budget kicks
    # in (sampled retries with a fresh seed can produce a shorter label).
    label = parsed["cluster_label"]
    if isinstance(label, str) and len(label) > _CLUSTER_LABEL_MAX_CHARS:
        return None, "failed", ["HC4"]

    quality = _cluster_tag_quality_codes(parsed, labels, spec)
    if quality:
        return None, "failed", quality

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

    # SC2: tag length bounds. The grammar caps every tag at
    # ``cluster_tag_max_chars`` via the schema's ``maxLength``, but vLLM does
    # NOT enforce ``maxLength`` (see ``_cluster_tag_quality_codes``), so an
    # over-long tag must be flagged here. tool_tags share the same
    # ``_tag_schema`` bound, so they are checked alongside rep/aes.
    cluster_tags = [e["tag"].strip() for block in (rep, aes) for e in block]
    cluster_tags += [t.strip() for t in tool_tags]
    for t in cluster_tags:
        if len(t) < labels.min_tag_chars or len(t) > labels.cluster_tag_max_chars:
            codes.add("SC2")
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

    # SC8: cluster_summary target band (soft — informational, does not gate)
    # SC6 is the hard sentence ceiling; SC8 is the tighter summary target band.
    summary_len = len(parsed["cluster_summary"].strip())
    if (
        summary_len < labels.cluster_summary_target_min
        or summary_len > labels.cluster_summary_target_max
    ):
        codes.add("SC8")

    # SC9: connector word in a tag (soft — was the HC5 hard fail; the grammar's
    # word/char cap already blocks real squashes, so a connector in a short
    # descriptive phrase is flagged, not rejected).
    for block in (rep, aes):
        if any({w.lower() for w in e["tag"].split()} & _CONNECTOR_WORDS for e in block):
            codes.add("SC9")
            break
    if "SC9" not in codes and any(
        {w.lower() for w in t.split()} & _CONNECTOR_WORDS for t in tool_tags
    ):
        codes.add("SC9")

    return codes


# Public surface — pure validators, no module-level mutable state.
__all__ = ["clip_role_keys", "format_failure_error", "validate", "validate_cluster"]
