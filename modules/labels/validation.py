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
            in_str, escape = _consume_string_char(ch, out, escape)
            continue
        in_str = _consume_structural_char(ch, out, stack)
    while stack:
        out.append("}" if stack.pop() == "{" else "]")
    return "".join(out)


def _consume_string_char(ch: str, out: list[str], escape: bool) -> tuple[bool, bool]:
    """Append a char inside a JSON string literal; return ``(in_str, escape)``."""
    out.append(ch)
    if escape:
        return True, False
    if ch == "\\":
        return True, True
    if ch == '"':
        return False, False
    return True, False


def _consume_structural_char(ch: str, out: list[str], stack: list[str]) -> bool:
    """Append a char outside a string literal; return whether a string opened."""
    if ch == '"':
        out.append(ch)
        return True
    if ch in "[{":
        stack.append(ch)
    elif ch == "}":
        if stack and stack[-1] == "{":
            stack.pop()
    elif ch == "]":
        while stack and stack[-1] == "{":
            out.append("}")
            stack.pop()
        if stack and stack[-1] == "[":
            stack.pop()
    out.append(ch)
    return False


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


def _observable_entry_ok(entry: Any) -> bool:
    """A clip observable-tag entry needs string ``tag`` and ``evidence``."""
    return (
        isinstance(entry, dict)
        and isinstance(entry.get("tag"), str)
        and isinstance(entry.get("evidence"), str)
    )


def _grounded_entry_ok(entry: Any) -> bool:
    """A clip aesthetic/community entry: string ``tag``/``confidence`` and a
    ``grounded_in`` list of strings."""
    if not isinstance(entry, dict):
        return False
    if not isinstance(entry.get("tag"), str):
        return False
    grounded = entry.get("grounded_in")
    if not isinstance(grounded, list) or not all(isinstance(g, str) for g in grounded):
        return False
    return isinstance(entry.get("confidence"), str)


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
    if not all(_observable_entry_ok(entry) for entry in obs):
        return False
    return all(_grounded_entry_ok(entry) for block in (aes, com) for entry in block)


def _bad_count(block: list, labels: LabelsSettings) -> bool:
    return (
        len(block) < labels.min_tags_per_kind or len(block) > labels.max_tags_per_kind
    )


def _bad_tag_len(tag: str, labels: LabelsSettings) -> bool:
    t = tag.strip()
    return len(t) < labels.min_tag_chars or len(t) > labels.max_tag_chars


def _has_duplicate_tags(block: list) -> bool:
    norm = [_norm(e["tag"]) for e in block]
    return len(set(norm)) != len(norm)


def _is_hashtaglike(tag: str) -> bool:
    t = tag.strip()
    return t.startswith("#") or t.startswith("@") or bool(_HASHTAGLIKE.match(t))


def _grounded_unresolved(block: list, valid: set[str]) -> bool:
    """True if any entry in ``block`` grounds in a tag not present in ``valid``."""
    return any(_norm(g) not in valid for entry in block for g in entry["grounded_in"])


def _soft_fail_codes(
    parsed: dict, labels: LabelsSettings, spec: LabelCaseSpec
) -> set[str]:
    codes: set[str] = set()
    observable_key, sentence_key = _clip_role_keys(spec)
    obs = parsed[observable_key]
    aes = parsed["aesthetic_tags"]
    com = parsed["community_signalling_tags"]
    sentence = parsed[sentence_key]
    all_blocks = (obs, aes, com)

    # S1: count bounds per kind
    if any(_bad_count(block, labels) for block in all_blocks):
        codes.add("S1")

    # S2: tag length bounds
    if any(_bad_tag_len(e["tag"], labels) for block in all_blocks for e in block):
        codes.add("S2")

    # S3: duplicate tags within a kind (case-insensitive, whitespace-normalised)
    if any(_has_duplicate_tags(block) for block in all_blocks):
        codes.add("S3")

    # S4: hashtag-like tag (#x, @x, or single \w+ token)
    if any(_is_hashtaglike(e["tag"]) for block in all_blocks for e in block):
        codes.add("S4")

    # S6: confidence enum
    if any(e["confidence"] not in _CONFIDENCES for block in (aes, com) for e in block):
        codes.add("S6")

    # S7: grounded_in references must hit an earlier section's tags
    observable_norms = {_norm(e["tag"]) for e in obs}
    aesthetic_norms = {_norm(e["tag"]) for e in aes}
    if _grounded_unresolved(aes, observable_norms) or _grounded_unresolved(
        com, observable_norms | aesthetic_norms
    ):
        codes.add("S7")

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


def _is_str_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(x, str) for x in value)


def _repertoire_entry_ok(entry: Any) -> bool:
    return (
        isinstance(entry, dict)
        and isinstance(entry.get("tag"), str)
        and isinstance(entry.get("description"), str)
        and entry.get("recurrence") in _RECURRENCES
    )


def _cluster_aesthetic_entry_ok(entry: Any) -> bool:
    return (
        isinstance(entry, dict)
        and isinstance(entry.get("tag"), str)
        and _is_str_list(entry.get("grounded_in"))
        and isinstance(entry.get("description"), str)
    )


def _signal_block_ok(block: Any) -> bool:
    return (
        isinstance(block, dict)
        and isinstance(block.get("label"), str)
        and isinstance(block.get("description"), str)
        and isinstance(block.get("confidence"), str)
    )


def _variation_entry_ok(entry: Any) -> bool:
    return (
        isinstance(entry, dict)
        and isinstance(entry.get("variation"), str)
        and isinstance(entry.get("description"), str)
    )


def _cluster_shapes_ok(parsed: dict, spec: LabelCaseSpec) -> bool:
    repertoire_key = _cluster_repertoire_key(spec)
    if not all(
        isinstance(parsed[k], str)
        for k in ("cluster_label", "cluster_summary", "boundary_notes")
    ):
        return False
    if not _is_str_list(parsed["tool_tags"]):
        return False
    rep = parsed[repertoire_key]
    if not isinstance(rep, list) or not all(_repertoire_entry_ok(e) for e in rep):
        return False
    aes = parsed["dominant_aesthetic_logic"]
    if not isinstance(aes, list) or not all(
        _cluster_aesthetic_entry_ok(e) for e in aes
    ):
        return False
    if not all(
        _signal_block_ok(parsed[key])
        for key in ("taste_signalling", "visibility_orientation")
    ):
        return False
    var = parsed["internal_variations"]
    return isinstance(var, list) and all(_variation_entry_ok(e) for e in var)


def _cluster_sentence_fields(parsed: dict, rep: list, aes: list) -> list[str]:
    """All sentence-shaped fields whose length SC6 bounds."""
    fields: list[str] = [parsed["cluster_summary"], parsed["boundary_notes"]]
    fields += [entry["description"] for entry in rep + aes]
    fields += [
        parsed[key]["description"]
        for key in ("taste_signalling", "visibility_orientation")
    ]
    fields += [entry["description"] for entry in parsed["internal_variations"]]
    return fields


def _has_connector_word(tag: str) -> bool:
    return bool({w.lower() for w in tag.split()} & _CONNECTOR_WORDS)


def _cluster_tool_tags_invalid(tool_tags: list) -> bool:
    """SC7: tool_tags must be non-empty and free of #/@ prefixes."""
    return not tool_tags or any(
        t.strip().startswith("#") or t.strip().startswith("@") for t in tool_tags
    )


def _cluster_tags_have_connector(rep: list, aes: list, tool_tags: list) -> bool:
    """SC9: a connector word in any repertoire/aesthetic/tool tag."""
    rep_aes_tags = (e["tag"] for block in (rep, aes) for e in block)
    return any(_has_connector_word(t) for t in rep_aes_tags) or any(
        _has_connector_word(t) for t in tool_tags
    )


def _cluster_soft_fail_codes(
    parsed: dict, labels: LabelsSettings, spec: LabelCaseSpec
) -> set[str]:
    codes: set[str] = set()
    repertoire_key = _cluster_repertoire_key(spec)
    rep = parsed[repertoire_key]
    aes = parsed["dominant_aesthetic_logic"]
    tool_tags = parsed["tool_tags"]

    # SC1: count bounds for repertoire and aesthetic logic
    if any(
        len(block) < labels.cluster_min_tags or len(block) > labels.cluster_max_tags
        for block in (rep, aes)
    ):
        codes.add("SC1")

    # SC2: tag length bounds. The grammar caps every tag at
    # ``cluster_tag_max_chars`` via the schema's ``maxLength``, but vLLM does
    # NOT enforce ``maxLength`` (see ``_cluster_tag_quality_codes``), so an
    # over-long tag must be flagged here. tool_tags share the same
    # ``_tag_schema`` bound, so they are checked alongside rep/aes.
    cluster_tags = [e["tag"].strip() for block in (rep, aes) for e in block]
    cluster_tags += [t.strip() for t in tool_tags]
    if any(
        len(t) < labels.min_tag_chars or len(t) > labels.cluster_tag_max_chars
        for t in cluster_tags
    ):
        codes.add("SC2")

    # SC3: duplicate tags within a block (case/whitespace-normalised)
    if any(_has_duplicate_tags(block) for block in (rep, aes)):
        codes.add("SC3")

    # SC4: aesthetic_logic.grounded_in must reference a repertoire tag
    rep_norms = {_norm(e["tag"]) for e in rep}
    if _grounded_unresolved(aes, rep_norms):
        codes.add("SC4")

    # SC5: confidence enum on taste_signalling and visibility_orientation
    if any(
        parsed[key]["confidence"] not in _CONFIDENCES
        for key in ("taste_signalling", "visibility_orientation")
    ):
        codes.add("SC5")

    # SC6: sentence-shaped fields length bounds
    if any(
        len(s.strip()) < labels.cluster_min_sentence_chars
        or len(s.strip()) > labels.cluster_max_sentence_chars
        for s in _cluster_sentence_fields(parsed, rep, aes)
    ):
        codes.add("SC6")

    # SC7: tool_tags non-empty and none may carry a hashtag/at-sign prefix
    if _cluster_tool_tags_invalid(tool_tags):
        codes.add("SC7")

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
    if _cluster_tags_have_connector(rep, aes, tool_tags):
        codes.add("SC9")

    return codes


# Public surface — pure validators, no module-level mutable state.
__all__ = ["clip_role_keys", "format_failure_error", "validate", "validate_cluster"]
