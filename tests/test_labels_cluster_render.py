import json
from dataclasses import dataclass

from modules.labels.cluster_render import (
    ClipCandidate,
    estimate_tokens,
    pick_clips,
    render_prompt_body,
)


@dataclass(frozen=True)
class _M:
    user_id: int
    centrality: float
    clips: list  # list[ClipCandidate]


def _candidate(
    *, clip_id: int, warnings: int = 0, payload_size: int = 200
) -> ClipCandidate:
    payload = {"x": "a" * payload_size}
    return ClipCandidate(clip_id=clip_id, warning_count=warnings, payload=payload)


def test_estimate_tokens_is_chars_over_four() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("a" * 100) == 25
    assert estimate_tokens("hello world") == len("hello world") // 4


def test_pick_clips_orders_by_centrality_then_user_id() -> None:
    members = [
        _M(user_id=5, centrality=0.1, clips=[_candidate(clip_id=51)]),
        _M(user_id=2, centrality=0.9, clips=[_candidate(clip_id=21)]),
        _M(user_id=3, centrality=0.9, clips=[_candidate(clip_id=31)]),
    ]
    picked = pick_clips(
        members,
        prompt_overhead_tokens=0,
        max_per_user=1,
        max_clips_total=10,
        token_budget=1_000_000,
    )
    assert [c.clip_id for c in picked] == [21, 31, 51]


def test_pick_clips_round_robin_under_per_user_cap() -> None:
    members = [
        _M(
            user_id=1,
            centrality=0.9,
            clips=[
                _candidate(clip_id=11),
                _candidate(clip_id=12),
                _candidate(clip_id=13),
            ],
        ),
        _M(
            user_id=2,
            centrality=0.8,
            clips=[
                _candidate(clip_id=21),
                _candidate(clip_id=22),
            ],
        ),
    ]
    picked = pick_clips(
        members,
        prompt_overhead_tokens=0,
        max_per_user=2,
        max_clips_total=10,
        token_budget=1_000_000,
    )
    assert [c.clip_id for c in picked] == [11, 21, 12, 22]


def test_pick_clips_respects_max_clips_total() -> None:
    members = [
        _M(user_id=i, centrality=1 - i * 0.01, clips=[_candidate(clip_id=i * 10)])
        for i in range(1, 11)
    ]
    picked = pick_clips(
        members,
        prompt_overhead_tokens=0,
        max_per_user=1,
        max_clips_total=3,
        token_budget=1_000_000,
    )
    assert len(picked) == 3


def test_pick_clips_respects_token_budget() -> None:
    members = [
        _M(
            user_id=1,
            centrality=0.9,
            clips=[
                _candidate(clip_id=11, payload_size=200),
                _candidate(clip_id=12, payload_size=200),
            ],
        ),
        _M(
            user_id=2,
            centrality=0.8,
            clips=[
                _candidate(clip_id=21, payload_size=200),
                _candidate(clip_id=22, payload_size=200),
            ],
        ),
    ]
    one_render = render_prompt_body([members[0].clips[0]])
    one_tokens = estimate_tokens(one_render)
    body_overhead = estimate_tokens(render_prompt_body([])) - 0
    budget = body_overhead + one_tokens * 2 + one_tokens // 2
    picked = pick_clips(
        members,
        prompt_overhead_tokens=0,
        max_per_user=2,
        max_clips_total=10,
        token_budget=budget,
    )
    assert len(picked) == 2


def test_pick_clips_prefers_clean_clips_within_user() -> None:
    members = [
        _M(
            user_id=1,
            centrality=0.9,
            clips=[
                _candidate(clip_id=11, warnings=2),
                _candidate(clip_id=12, warnings=0),
                _candidate(clip_id=13, warnings=1),
            ],
        ),
    ]
    picked = pick_clips(
        members,
        prompt_overhead_tokens=0,
        max_per_user=1,
        max_clips_total=10,
        token_budget=1_000_000,
    )
    assert [c.clip_id for c in picked] == [12]


def test_render_prompt_body_returns_json_list() -> None:
    cands = [_candidate(clip_id=11), _candidate(clip_id=22)]
    body = render_prompt_body(cands)
    parsed = json.loads(body)
    assert isinstance(parsed, list)
    assert {e["clip_id"] for e in parsed} == {11, 22}


def test_render_prompt_body_is_stable_for_same_input() -> None:
    cands = [_candidate(clip_id=11), _candidate(clip_id=22)]
    assert render_prompt_body(cands) == render_prompt_body(cands)
