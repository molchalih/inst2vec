"""Tests for core/console.py — universal log() + dual-bar progress."""

from __future__ import annotations

import io

import pytest
from rich.console import Console


def _capture(fn) -> str:
    """Invoke fn() with the single console writing to an in-memory buffer."""
    from core import console as c

    buf = io.StringIO()
    real_console = c._console
    c._console = Console(file=buf, force_terminal=False, width=200, no_color=True)
    # Reset dedup state so assertions about scope/time presence are not
    # silently suppressed by leftover state from a prior test.
    c._render_state.last_time = ""
    c._render_state.last_scope = ""
    try:
        fn()
    finally:
        c._console = real_console
    return buf.getvalue()


# ---------- format snapshots ----------


def test_log_minimal_emits_canonical_line() -> None:
    from core.console import log

    out = _capture(lambda: log("hiker", "GET", "user_by_id/1", "200"))
    # No [scope] prefix in body.
    assert "[hiker]" not in out
    # Body present.
    assert "GET user_by_id/1 200" in out
    # Scope appears on the right margin.
    assert "hiker" in out


def test_log_with_stats_renders_bracket_block() -> None:
    from core.console import log

    out = _capture(
        lambda: log(
            "hiker",
            "GET",
            "user_by_id/1",
            "200",
            stats={"time": 0.42, "size": 1234},
        )
    )
    assert "GET user_by_id/1 200" in out
    assert "time=0.42s" in out
    assert "size=1.2KB" in out
    assert "[hiker]" not in out


def test_log_stats_preserve_insertion_order() -> None:
    from core.console import log

    out = _capture(
        lambda: log(
            "embed:video",
            "EMB",
            "clip_8f3a",
            "ok",
            stats={"time": 0.18, "dim": 2048},
        )
    )
    assert "[embed:video]" not in out
    assert "time=0.18s, dim=2048" in out
    assert "embed:video" in out


# ---------- time/size humanization ----------


@pytest.mark.parametrize(
    "seconds, expected",
    [
        (0.042, "42ms"),
        (0.42, "0.42s"),
        (1.21, "1.21s"),
        (12.4, "12.4s"),
        (61.0, "1m01s"),
        (192.0, "3m12s"),
    ],
)
def test_format_time(seconds: float, expected: str) -> None:
    from core.console import _format_time

    assert _format_time(seconds) == expected


@pytest.mark.parametrize(
    "n_bytes, expected",
    [
        (512, "512B"),
        (1234, "1.2KB"),
        (2_400_000, "2.4MB"),
        (1_100_000_000, "1.1GB"),
    ],
)
def test_format_size(n_bytes: int, expected: str) -> None:
    from core.console import _format_size

    assert _format_size(n_bytes) == expected


# ---------- error path ----------


def test_log_err_emits_indented_err_line() -> None:
    from core.console import log

    out = _capture(
        lambda: log(
            "acr",
            "ID",
            "clip_1",
            "ERR",
            stats={"err": "quota exceeded"},
        )
    )
    assert "[acr]" not in out
    assert "ID clip_1 ERR" in out
    assert "acr" in out
    assert "err: quota exceeded" in out


# ---------- verb validation ----------


def test_log_rejects_unknown_verb() -> None:
    from core.console import log

    with pytest.raises(ValueError, match="unknown verb"):
        log("hiker", "FOO", "x", "ok")


# ---------- highlighter regex ----------


def test_highlighter_regex_matches_canonical_form() -> None:
    import re

    from core.console import _WORKER_REGEX

    line = "GET user_by_id/1 200 [time=0.42s, size=1.2KB]"
    m = re.match(_WORKER_REGEX, line)
    assert m is not None
    assert m.group("verb") == "GET"
    assert m.group("target") == "user_by_id/1"
    assert m.group("result") == "200"
    assert m.group("stats") == "time=0.42s, size=1.2KB"


# ---------- pipeline lifecycle ----------


def test_pipeline_advances_one_tick_per_phase() -> None:
    from core import console as c
    from core.console import phase, pipeline

    def _task():
        return next(
            t for t in c._pipeline_progress.tasks if t.id == c._pipeline_task_id
        )

    with pipeline(total_stages=2):
        phase("a")
        assert _task().completed == 1
        phase("b")
        assert _task().completed == 2


def test_progress_inside_pipeline_creates_stage_task() -> None:
    from core import console as c
    from core.console import phase, pipeline, progress

    with pipeline(total_stages=1):
        phase("stage")
        with progress(10, "work") as advance:
            assert len(c._stage_progress.tasks) == 1
            assert c._stage_progress.tasks[0].total == 10
            advance(5)
            assert c._stage_progress.tasks[0].completed == 5
        assert len(c._stage_progress.tasks) == 0


# ---------- dedup behaviour ----------


def test_log_consecutive_same_second_dedupes_time(monkeypatch) -> None:
    """Two log() calls in the same second print the time once, then blanks."""
    # Pin datetime.now() so both calls produce the same HH:MM:SS.
    import datetime as _dt

    from core import console as c
    from core.console import log

    class _FixedDT(_dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return _dt.datetime(2026, 5, 20, 4, 30, 20)

    monkeypatch.setattr(c, "datetime", _FixedDT)
    c._render_state.last_time = ""
    c._render_state.last_scope = ""

    out = _capture(
        lambda: (
            log("hiker", "GET", "a", "ok"),
            log("hiker", "GET", "b", "ok"),
        )
    )
    # First call prints the timestamp; second call uses blank padding.
    assert out.count("[04:30:20]") == 1


def test_log_consecutive_same_scope_dedupes_scope(monkeypatch) -> None:
    from core import console as c
    from core.console import log

    c._render_state.last_time = ""
    c._render_state.last_scope = ""

    out = _capture(
        lambda: (
            log("hiker", "GET", "a", "ok"),
            log("hiker", "GET", "b", "ok"),
            log("acr", "ID", "c", "ok"),
        )
    )
    # "hiker" appears once on the right (first line), then suppressed;
    # "acr" appears when the scope changes.
    assert out.count("hiker") == 1
    assert "acr" in out


def test_phase_resets_scope_so_next_log_reprints(monkeypatch) -> None:
    from core import console as c
    from core.console import log, phase

    c._render_state.last_time = ""
    c._render_state.last_scope = "hiker"  # pretend last line was hiker

    def _do():
        phase("New Phase")
        log("hiker", "GET", "x", "ok")

    out = _capture(_do)
    # After phase(), the scope state was reset, so hiker must appear.
    assert "hiker" in out
