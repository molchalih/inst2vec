"""Pure helpers for the per-cluster labelling pass: sampling and prompt body rendering.

No DB access. Inputs are primitive dataclasses; output is a `str` ready to
be appended to ``prompt_for_cluster(labels)`` and a `list[clip_id]` for
provenance tracking.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ClipCandidate:
    """One clip's stage-1 payload, ready to be rendered into the prompt."""

    clip_id: int
    warning_count: int
    payload: dict


class _Member(Protocol):
    user_id: int
    centrality: float
    # ``list`` (not ``Sequence``) because Protocol attribute matching is
    # invariant — ``cluster_pass._ClusterMember`` holds a list it mutates
    # via ``.append``; a wider ``Sequence`` annotation here would reject the
    # concrete ``list[ClipCandidate]`` at the ``pick_clips`` call site.
    clips: list[ClipCandidate]


def estimate_tokens(text: str) -> int:
    """Rough char-based token estimator (~chars/4).

    Deliberately rough: the budget is itself a soft guard with a wide
    safety margin against ``cluster_max_new_tokens``. Avoids pulling in
    the tokenizer just for sampling.
    """
    return len(text) // 4


def _rank_user_clips(clips: Sequence[ClipCandidate]) -> list[ClipCandidate]:
    """Within one user: clean payloads first (fewer warnings), then lowest clip_id."""
    return sorted(clips, key=lambda c: (c.warning_count, c.clip_id))


def _rank_members(members: Sequence[_Member]) -> list[_Member]:
    """Across users in a cluster: centrality desc, user_id asc tiebreaker."""
    return sorted(members, key=lambda m: (-m.centrality, m.user_id))


def render_prompt_body(candidates: Sequence[ClipCandidate]) -> str:
    """JSON list of {clip_id, payload}; stable key order (sort_keys=True)."""
    return json.dumps(
        [{"clip_id": c.clip_id, "payload": c.payload} for c in candidates],
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def pick_clips(
    members: Sequence[_Member],
    *,
    prompt_overhead_tokens: int,
    max_per_user: int,
    max_clips_total: int,
    token_budget: int,
) -> list[ClipCandidate]:
    """Greedy round-robin selection under all three caps.

    Order: members sorted by centrality desc / user_id asc; within each
    member, clips sorted by warning_count asc / clip_id asc.

    Round 1 picks one clip per member in member order; round 2 picks the
    second clip per member; etc., up to ``max_per_user`` rounds. Stops on
    ``max_clips_total`` or when the next render would exceed
    ``prompt_overhead_tokens + token_budget`` (estimated).

    Members with empty ``clips`` are skipped silently.
    """
    ordered_members = _rank_members(members)
    per_user = [_rank_user_clips(m.clips) for m in ordered_members]

    picked: list[ClipCandidate] = []
    for round_idx in range(max_per_user):
        added_in_round = False
        for clips in per_user:
            if round_idx >= len(clips):
                continue
            cand = clips[round_idx]
            trial = [*picked, cand]
            if (
                estimate_tokens(render_prompt_body(trial)) + prompt_overhead_tokens
                > token_budget
            ):
                return picked
            picked.append(cand)
            added_in_round = True
            if len(picked) >= max_clips_total:
                return picked
        if not added_in_round:
            return picked
    return picked
