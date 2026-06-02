"""Global cluster-label naming pass.

Per-cluster labels are generated in isolation (``cluster_pass``): each cluster's
prompt never sees its siblings, so semantically-close clusters — acutely the
music / ``auditory`` case — independently reach for the same head words
("ambient", "cinematic", "electronic", ...). The old exact-string dedup could
only see *byte-identical* collisions, so near-duplicates ("Cinematic Ambient
Tension" vs "Cinematic Ambient Fusion" vs "Ambient Cinematic Intimacy") sailed
through, and the few exact hits it caught were "fixed" with a deterministic
suffix that truncated mid-word and tacked on the dominant genre tag or the
cluster id ("Cinematic Ambient F — electronic ambient", "... (17)").

This pass replaces that machinery. It runs ONCE per case after per-cluster
generation and shows the model EVERY cluster's summary plus its distinguishing
elements in a single prompt — the only context in which a model can
deliberately spread its vocabulary across the whole set. The result is scored
for lexical overlap (no significant word may recur across labels) and length
consistency (Title Case, 2-3 words, ``<= CLUSTER_LABEL_MAX_CHARS``); residual
overlap drives a bounded feedback-regeneration loop. A deterministic
exact-uniqueness guard is the final, rarely-needed backstop and never truncates
mid-word.

Pure helpers (no DB / GPU) are unit-tested directly; ``assign_distinct_labels``
is the orchestration entry the cluster pass calls.
"""

from __future__ import annotations

import itertools
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, cast

from sqlalchemy import select
from sqlalchemy.orm import Session

from core.config import LabelsSettings
from core.database import ClusterLabel
from core.log import item, warn
from modules.labels.cases import REGISTRY
from modules.labels.prompts import naming_prompt_for
from modules.labels.validation import CLUSTER_LABEL_MAX_CHARS

# A label is 2-3 space-separated Title-Case words. Each word starts uppercase
# and may carry ONE hyphenated continuation ("Hip-Hop", "Avant-Garde",
# "Synth-Driven"). This is BOTH the post-parse validator and the grammar
# pattern (``naming_schema``) so the model is constrained to the exact shape
# the validator accepts — no "model targets X, validator wants Y" drift.
_MIN_WORDS = 2
_MAX_WORDS = 3
_WORD = r"[A-Z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)?"
_LABEL_PATTERN = rf"{_WORD}(?: {_WORD}){{{_MIN_WORDS - 1},{_MAX_WORDS - 1}}}"
_LABEL_RE = re.compile(rf"^{_LABEL_PATTERN}$")

# Grammar pattern variant: xgrammar wants a raw (un-anchored elsewhere) pattern;
# it anchors implicitly. Re-use the same body so validator and grammar agree.
_LABEL_GRAMMAR_PATTERN = f"^{_LABEL_PATTERN}$"

# Filler words that describe every cluster and so distinguish none. Mirrors the
# per-cluster ``cluster_label`` ban list in config so both passes agree.
BANNED_WORDS: frozenset[str] = frozenset(
    {"aesthetic", "aesthetics", "curated", "cluster", "vibe", "vibes"}
)

# Words ignored when measuring cross-label lexical overlap.
_STOPWORDS: frozenset[str] = frozenset(
    {"and", "the", "of", "a", "an", "to", "in", "on", "with", "for", "or", "by"}
)

# A significant word may appear in at most this many labels before the overlap
# report flags it and the feedback loop is asked to re-diversify. 2 (rather than
# 1) tolerates the unavoidable shared term in a genuinely narrow domain while
# still killing the "ambient x7" degeneracy.
MAX_SHARED_WORD = 2


def _norm(label: str) -> str:
    return " ".join(label.strip().lower().split())


def content_words(label: str) -> set[str]:
    """Significant (non-stopword, len>2) lowercase tokens, hyphens split out."""
    out: set[str] = set()
    for raw in label.replace("-", " ").split():
        w = raw.strip().lower()
        if len(w) > 2 and w not in _STOPWORDS:
            out.add(w)
    return out


def valid_label(label: object) -> bool:
    """Title-Case 2-3 words, ``<= CLUSTER_LABEL_MAX_CHARS``, no banned filler."""
    if not isinstance(label, str):
        return False
    s = label.strip()
    if not s or len(s) > CLUSTER_LABEL_MAX_CHARS:
        return False
    if not _LABEL_RE.match(s):
        return False
    return not (content_words(s) & BANNED_WORDS)


@dataclass(frozen=True)
class OverlapReport:
    """Cross-label lexical / exact-duplicate diagnostics for one case."""

    labels_by_cid: dict[int, str]
    duplicates: dict[str, list[int]]  # normalised label -> cids (only len > 1)
    word_to_cids: dict[str, list[int]]  # content word -> cids whose label uses it

    @property
    def max_word_freq(self) -> int:
        return max((len(c) for c in self.word_to_cids.values()), default=0)

    def overused(self, max_shared: int = MAX_SHARED_WORD) -> dict[str, list[int]]:
        return {w: c for w, c in self.word_to_cids.items() if len(c) > max_shared}

    def is_clean(self, max_shared: int = MAX_SHARED_WORD) -> bool:
        return not self.duplicates and not self.overused(max_shared)

    def feedback(self, max_shared: int = MAX_SHARED_WORD) -> str:
        lines: list[str] = []
        if self.duplicates:
            dup = "; ".join(
                f'"{self.labels_by_cid[c[0]]}" (clusters {c})'
                for c in self.duplicates.values()
            )
            lines.append(f"EXACT DUPLICATES to break apart: {dup}.")
        over = self.overused(max_shared)
        if over:
            worst = sorted(over.items(), key=lambda kv: (-len(kv[1]), kv[0]))
            joined = "; ".join(f'"{w}" (clusters {cids})' for w, cids in worst)
            lines.append(
                "OVERUSED words — each appears in too many labels; keep it in at "
                f"most ONE label and rename the others: {joined}."
            )
        return " ".join(lines)


def overlap_report(labels_by_cid: dict[int, str]) -> OverlapReport:
    by_norm: dict[str, list[int]] = defaultdict(list)
    word_to_cids: dict[str, list[int]] = defaultdict(list)
    for cid in sorted(labels_by_cid):
        label = labels_by_cid[cid]
        by_norm[_norm(label)].append(cid)
        for w in content_words(label):
            word_to_cids[w].append(cid)
    duplicates = {n: cids for n, cids in by_norm.items() if len(cids) > 1}
    return OverlapReport(
        labels_by_cid=dict(labels_by_cid),
        duplicates=duplicates,
        word_to_cids=dict(word_to_cids),
    )


# ---------------------------------------------------------------------------
# Roster + prompt
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RosterEntry:
    cluster_id: int
    summary: str
    tags: list[str] = field(default_factory=list)


def _top_tags(payload: dict, repertoire_key: str, n: int = 6) -> list[str]:
    """Distinguishing elements for one cluster: leading repertoire tags, then
    aesthetic-logic tags as backfill. These are what let the model tell two
    musically-similar clusters apart, so they drive the naming, not the summary
    alone."""
    tags: list[str] = []
    for block_key in (repertoire_key, "dominant_aesthetic_logic"):
        for entry in payload.get(block_key) or []:
            if isinstance(entry, dict):
                tag = entry.get("tag")
                if isinstance(tag, str) and tag.strip() and tag not in tags:
                    tags.append(tag.strip())
            if len(tags) >= n:
                return tags[:n]
    return tags[:n]


def build_roster(rows: list[ClusterLabel], *, repertoire_key: str) -> list[RosterEntry]:
    roster: list[RosterEntry] = []
    for row in sorted(rows, key=lambda r: r.cluster_id):
        payload = row.payload or {}
        summary = (payload.get("cluster_summary") or "").strip()
        roster.append(
            RosterEntry(
                cluster_id=row.cluster_id,
                summary=summary,
                tags=_top_tags(payload, repertoire_key),
            )
        )
    return roster


def build_naming_prompt(
    instructions: str,
    roster: list[RosterEntry],
    labels_by_cid: dict[int, str],
    *,
    max_shared: int = MAX_SHARED_WORD,
    feedback: str = "",
) -> str:
    """Compose the single all-clusters naming prompt.

    ``instructions`` is the operator-tunable lead-in (config
    ``labels.cluster_naming_prompt``); the roster, the current provisional
    names, the structural output contract and any retry feedback are appended
    here so they stay locked to the parser.
    """
    del labels_by_cid  # working names are deliberately NOT shown — see below.
    n = len(roster)
    banned = ", ".join(sorted(BANNED_WORDS))
    parts = [
        instructions.strip(),
        (
            f"\nYou are naming {n} clusters. Each entry below gives the cluster's "
            "id, a one-line summary, and its key distinguishing elements."
        ),
        (
            "\nRULES:\n"
            f"- Each label: {_MIN_WORDS} to {_MAX_WORDS} words, Title Case, at "
            f"most {CLUSTER_LABEL_MAX_CHARS} characters.\n"
            f"- NEVER use these filler words: {banned}.\n"
            "- DECISIVE RULE: across the whole set, no significant word may "
            f"appear in more than {max_shared} label(s). Treat every label as "
            "competing for distinct vocabulary.\n"
            "- The summaries are written in similar generic language on purpose; "
            "do NOT name from the summary's mood words. Name each cluster from "
            "its most DISTINCTIVE key element — the specific genre, sound or "
            "texture that sets it apart from the others.\n"
            '- Keep one concise noun phrase per label; no "X and Y" lists.'
        ),
        "\nCLUSTERS:",
    ]
    for e in roster:
        tags = ", ".join(e.tags) if e.tags else "(none)"
        summary = e.summary or "(no summary)"
        parts.append(f"[{e.cluster_id}] summary: {summary} | key elements: {tags}")
    if feedback:
        parts.append(f"\nFIX THESE PROBLEMS FROM THE PREVIOUS ATTEMPT: {feedback}")
    example = ", ".join(
        f'{{"cluster_id": {e.cluster_id}, "label": "..."}}' for e in roster[:2]
    )
    parts.append(
        "\nReturn ONLY a JSON array with exactly one object per cluster, each "
        'shaped {"cluster_id": <int>, "label": "<Title Case label>"}, e.g. '
        f"[{example}, ...]. Cover every cluster id listed above exactly once. "
        "Emit only the JSON array — no code fences, no commentary."
    )
    return "\n".join(parts)


def naming_schema() -> dict:
    """Grammar for the naming response: an array of {cluster_id, label} objects,
    label constrained to the Title-Case 2-3 word pattern."""
    return {
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": False,
            "required": ["cluster_id", "label"],
            "properties": {
                "cluster_id": {"type": "integer"},
                "label": {
                    "type": "string",
                    "maxLength": CLUSTER_LABEL_MAX_CHARS,
                    "pattern": _LABEL_GRAMMAR_PATTERN,
                },
            },
        },
    }


def parse_naming_response(raw: str, expected_cids: set[int]) -> dict[int, str]:
    """Parse the model's JSON array into ``{cluster_id: label}``.

    Keeps only entries whose id is expected and whose label passes
    ``valid_label``; first occurrence of each id wins. Tolerant of a single
    wrapping object ({"labels": [...]}) and of curly quotes.
    """
    data = _loads(raw.strip().translate(_CURLY))
    if isinstance(data, dict):
        mapping = cast("dict[str, Any]", data)
        for key in ("labels", "clusters", "result", "data"):
            inner = mapping.get(key)
            if isinstance(inner, list):
                data = inner
                break
    if not isinstance(data, list):
        return {}
    out: dict[int, str] = {}
    for raw_entry in data:
        if not isinstance(raw_entry, dict):
            continue
        entry = cast("dict[str, Any]", raw_entry)
        cid = entry.get("cluster_id")
        label = entry.get("label")
        if not isinstance(cid, int) or cid not in expected_cids or cid in out:
            continue
        if isinstance(label, str) and valid_label(label):
            out[cid] = label.strip()
    return out


_CURLY = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})


def _loads(text: str) -> object:
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        pass
    try:
        from json_repair import repair_json

        return repair_json(text, return_objects=True)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Deterministic exact-uniqueness backstop
# ---------------------------------------------------------------------------


def _title(word: str) -> str:
    return word[:1].upper() + word[1:] if word else word


def _disambiguate(label: str, tags: list[str], taken: set[str]) -> str:
    have = content_words(label)
    for tag in tags:
        for raw in tag.replace("-", " ").split():
            w = raw.strip().lower()
            if len(w) <= 2 or w in _STOPWORDS or w in have or w in BANNED_WORDS:
                continue
            candidate = f"{label} {_title(w)}"
            if (
                len(candidate) <= CLUSTER_LABEL_MAX_CHARS
                and _norm(candidate) not in taken
            ):
                return candidate
            have.add(w)
    # Exhausted distinguishing words: append an incrementing ordinal that keeps
    # the label inside the cap. Only reached if every key element is already in
    # the label — effectively never for real data.
    n = 2
    while True:
        suffix = f" {n}"
        base = label[: CLUSTER_LABEL_MAX_CHARS - len(suffix)].rstrip()
        candidate = f"{base}{suffix}"
        if _norm(candidate) not in taken:
            return candidate
        n += 1


# ---------------------------------------------------------------------------
# Deterministic lexical-diversity guarantee
# ---------------------------------------------------------------------------


def phrase_to_label(phrase: str) -> str | None:
    """Turn a tag phrase ("ethereal minimalism", "club-ready groove") into a
    valid 2-3 word Title-Case label, or ``None`` if it cannot make one.

    Splits on spaces AND hyphens (so multi-hyphen compounds like
    "cinematic-industrial-epic" become separate words), drops connector words,
    Title-cases the rest and keeps the leading 2-3 content words. Used to mine
    each cluster's OWN aesthetic / repertoire tags for replacement labels when
    the model's choice collides on an already-used word.
    """
    words = [w for w in re.split(r"[ \-]+", phrase.strip()) if w]
    kept: list[str] = []
    for w in words:
        lw = w.lower()
        if lw in _STOPWORDS:
            continue
        kept.append(_title(lw))
        if len(kept) == _MAX_WORDS:
            break
    if len(kept) < _MIN_WORDS:
        return None
    label = " ".join(kept)
    return label if valid_label(label) else None


def candidate_phrases(payload: dict, repertoire_key: str) -> list[str]:
    """Ordered replacement-label candidates for one cluster.

    Aesthetic-logic tags first (they are already descriptive 2-3 word phrases),
    then repertoire genres (the distinctive sound), each converted to a valid
    label. Generic genre prefixes shared by the whole set ("electronic ...")
    naturally rank themselves out because their leading word caps quickly.
    """
    out: list[str] = []
    seen: set[str] = set()
    for block_key in ("dominant_aesthetic_logic", repertoire_key):
        for entry in payload.get(block_key) or []:
            tag = entry.get("tag") if isinstance(entry, dict) else None
            if not isinstance(tag, str):
                continue
            label = phrase_to_label(tag)
            if label and _norm(label) not in seen:
                out.append(label)
                seen.add(_norm(label))
    return out


def resolve_lexical_overlap(
    model_labels: dict[int, str],
    candidates_by_cid: dict[int, list[str]],
    *,
    max_shared: int = MAX_SHARED_WORD,
) -> dict[int, str]:
    """Guarantee labels are exact-distinct AND no content word recurs in more
    than ``max_shared`` labels.

    Greedy in cluster-id order. For each cluster the first acceptable label
    wins among: the model's label, then the cluster's own candidate phrases
    (aesthetic logic, then genres). "Acceptable" = valid, not already taken,
    and introducing no word that is already used in ``max_shared`` labels. If
    every candidate is blocked, fall back to the exact-uniqueness disambiguator
    so the result is always complete and distinct.

    This is the robustness backbone: it makes the final quality bar a property
    of the algorithm, not of the model's willingness to self-diversify.
    """
    word_count: Counter[str] = Counter()
    taken: set[str] = set()
    out: dict[int, str] = {}
    for cid in sorted(model_labels):
        # Prefer the model/provisional label; if it is not already a valid
        # label (e.g. lowercase, >3 words, or a banned word) coerce it via
        # phrase_to_label so a good name is normalised rather than discarded.
        raw = (model_labels.get(cid) or "").strip()
        primary = raw if valid_label(raw) else phrase_to_label(raw)
        cands: list[str] = []
        for c in [primary, *candidates_by_cid.get(cid, [])]:
            c = (c or "").strip()
            if c and valid_label(c) and _norm(c) not in {_norm(x) for x in cands}:
                cands.append(c)
        chosen: str | None = None
        for cand in cands:
            words = content_words(cand)
            if _norm(cand) in taken:
                continue
            if any(word_count[w] >= max_shared for w in words):
                continue
            chosen = cand
            break
        if chosen is None:
            # Every candidate is blocked: keep the model label (or first valid
            # candidate) and make it exact-unique via the disambiguator.
            base = next((c for c in cands if _norm(c) not in taken), None)
            if base is None:
                base = cands[0] if cands else model_labels[cid].strip()
            chosen = _disambiguate_lexical(
                base, candidates_by_cid.get(cid, []), taken, word_count, max_shared
            )
        out[cid] = chosen
        taken.add(_norm(chosen))
        for w in content_words(chosen):
            word_count[w] += 1
    return out


def _disambiguate_lexical(
    label: str,
    candidate_tags: list[str],
    taken: set[str],
    word_count: Counter[str],
    max_shared: int,
) -> str:
    """Build a fresh, valid, exact-unique label from the cluster's OWN
    distinctive words when every prepared candidate is blocked.

    Words still under the shared-frequency cap are tried first so the
    "no content word in more than ``max_shared`` labels" guarantee is preserved
    wherever the cluster's vocabulary allows it; only if no cap-respecting
    combination exists is the cap relaxed. Every returned label passes
    ``valid_label`` (2-3 Title-Case words, ``<= CLUSTER_LABEL_MAX_CHARS``, no
    banned filler). The ordinal disambiguator is reached only when the cluster
    has too few usable words to form ANY fresh distinct label — effectively
    never for real clusters, which carry many tag words."""
    # Distinct distinctive words from this cluster's own tags, ordered
    # under-cap-first then rarest-first.
    pool: list[str] = []
    seen: set[str] = set()
    for phrase in candidate_tags:
        for raw in re.split(r"[ \-]+", phrase.lower()):
            w = raw.strip()
            if (
                len(w) > 2
                and w not in _STOPWORDS
                and w not in BANNED_WORDS
                and w not in seen
            ):
                pool.append(w)
                seen.add(w)
    pool.sort(key=lambda w: (word_count[w] >= max_shared, word_count[w]))

    # Pass 1 — cap-respecting fresh labels (2 words preferred, then 3).
    for size in (_MIN_WORDS, _MAX_WORDS):
        for combo in itertools.combinations(pool, size):
            if any(word_count[w] >= max_shared for w in combo):
                continue
            cand = " ".join(_title(w) for w in combo)
            if valid_label(cand) and _norm(cand) not in taken:
                return cand
    # Pass 2 — relax the cap but still emit a valid, distinct label.
    for size in (_MIN_WORDS, _MAX_WORDS):
        for combo in itertools.combinations(pool, size):
            cand = " ".join(_title(w) for w in combo)
            if valid_label(cand) and _norm(cand) not in taken:
                return cand
    # Truly degenerate: guaranteed-unique ordinal suffix (mathematical last
    # resort; unreachable for clusters with >= 2 usable tag words).
    return _disambiguate(label, candidate_tags, taken)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _success_rows(session: Session, case: str) -> list[ClusterLabel]:
    return list(
        session.execute(
            select(ClusterLabel).where(
                ClusterLabel.embedding_case == case,
                ClusterLabel.status == "success",
            )
        )
        .scalars()
        .all()
    )


def assign_distinct_labels(
    session: Session,
    *,
    case: str,
    labels: LabelsSettings,
    generator,
) -> None:
    """Replace every cluster's ``cluster_label`` with a globally-coordinated,
    lexically-distinct, length-consistent name.

    The model proposes a full set of labels (seeing every cluster at once); the
    deterministic ``resolve_lexical_overlap`` then GUARANTEES the quality bar —
    exact-distinct and no content word recurring in more than ``MAX_SHARED_WORD``
    labels — by rebuilding any colliding label from that cluster's own
    distinctive aesthetic / genre phrases. The rich per-cluster payload
    (summary, repertoire, aesthetic logic, ...) is untouched — only
    ``cluster_label`` is rewritten.
    """
    spec = REGISTRY[case]
    rows = _success_rows(session, case)
    if not rows:
        return
    row_by_cid = {r.cluster_id: r for r in rows}
    roster = build_roster(rows, repertoire_key=spec.repertoire_key)
    labels_by_cid = {
        r.cluster_id: (r.payload or {}).get("cluster_label", "") for r in rows
    }
    # Idempotent guard: the cluster pass calls this every run, even when the
    # fingerprint gate regenerated nothing. If the existing labels already meet
    # the bar they were coordinated on a prior run — skip the expensive model
    # pass and leave them untouched so reruns neither reload the 30B nor churn
    # names under the same fingerprint.
    if (
        all(valid_label(v) for v in labels_by_cid.values())
        and overlap_report(labels_by_cid).is_clean()
    ):
        return
    candidates_by_cid = {
        cid: candidate_phrases(row.payload or {}, spec.repertoire_key)
        for cid, row in row_by_cid.items()
    }
    expected = set(row_by_cid)

    if len(rows) > 1:
        labels_by_cid = _name_via_model(
            case=case,
            labels=labels,
            generator=generator,
            roster=roster,
            labels_by_cid=labels_by_cid,
            expected=expected,
        )

    final = resolve_lexical_overlap(labels_by_cid, candidates_by_cid)
    _persist(session, case=case, row_by_cid=row_by_cid, final=final)


def _name_via_model(
    *,
    case: str,
    labels: LabelsSettings,
    generator,
    roster: list[RosterEntry],
    labels_by_cid: dict[int, str],
    expected: set[int],
) -> dict[int, str]:
    instructions = naming_prompt_for(labels, case=case)
    schema = naming_schema()
    rounds = max(labels.cluster_dedup_max_rounds, 1)
    feedback = ""
    current = dict(labels_by_cid)
    for round_i in range(rounds):
        prompt = build_naming_prompt(instructions, roster, current, feedback=feedback)
        seed = labels.generation_seed + labels.cluster_max_attempts + 1 + round_i
        try:
            raw = generator.run_text_batch(
                [prompt],
                max_new_tokens=labels.cluster_max_new_tokens,
                seeds=[seed],
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                schema=schema,
            )[0]
        except Exception as exc:
            warn(
                "GET", "qwen3-cluster", err=exc, stats={"case": case, "phase": "naming"}
            )
            break
        proposed = parse_naming_response(raw, expected)
        for cid, label in proposed.items():
            current[cid] = label
        report = overlap_report(current)
        if report.is_clean():
            break
        feedback = report.feedback()
    generator.reclaim_memory()
    return current


def _persist(
    session: Session,
    *,
    case: str,
    row_by_cid: dict[int, ClusterLabel],
    final: dict[int, str],
) -> None:
    for cid, label in final.items():
        row = row_by_cid[cid]
        if (row.payload or {}).get("cluster_label") == label:
            continue
        with item("WRITE", f"{case}/{cid}"):
            row.payload = {**(row.payload or {}), "cluster_label": label}
    session.commit()


__all__ = [
    "BANNED_WORDS",
    "MAX_SHARED_WORD",
    "OverlapReport",
    "RosterEntry",
    "assign_distinct_labels",
    "build_naming_prompt",
    "build_roster",
    "candidate_phrases",
    "content_words",
    "naming_schema",
    "overlap_report",
    "parse_naming_response",
    "phrase_to_label",
    "resolve_lexical_overlap",
    "valid_label",
]
