"""Unit tests for core.log — the pattern layer above core.console.

Each test installs a capturing _render stub via monkeypatch so we assert on
the (scope, verb, target, result, stats) tuples emitted, not on Rich output.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from core import log as cl


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, ...]]:
    """Replace core.log._render with a capture list of emitted log calls."""
    sink: list[tuple[Any, ...]] = []

    def fake_render(
        scope: str,
        verb: str,
        target: str,
        result: str = "ok",
        *,
        stats: dict[str, Any] | None = None,
    ) -> None:
        sink.append((scope, verb, target, result, dict(stats or {})))

    monkeypatch.setattr(cl, "_render", fake_render)
    return sink


@pytest.fixture(autouse=True)
def _reset_scope() -> Iterator[None]:
    """Ensure each test starts with no active scope (ContextVar isolation)."""
    token = cl._scope_var.set(None)
    yield
    cl._scope_var.reset(token)


def test_item_logs_time_and_stats_on_clean_exit(
    captured: list[tuple[Any, ...]],
) -> None:
    cl._scope_var.set("embed:video")
    with cl.item("EXTRACT", "clip_42") as t:
        t.stats(dim=2048)
    assert len(captured) == 1
    scope, verb, target, result, stats = captured[0]
    assert (scope, verb, target, result) == ("embed:video", "EXTRACT", "clip_42", "ok")
    assert stats["dim"] == 2048
    assert "time" in stats
    assert stats["time"] >= 0.0
    assert t.failed is False
    assert t.exc is None
    assert t.elapsed_s == stats["time"]


def test_item_logs_err_and_suppresses_on_exception(
    captured: list[tuple[Any, ...]],
) -> None:
    cl._scope_var.set("embed:video")
    with cl.item("EXTRACT", "clip_42") as t:
        t.stats(dim=2048)
        raise RuntimeError("CudaOOM")
    # Exception suppressed:
    assert t.failed is True
    assert isinstance(t.exc, RuntimeError)
    assert t.elapsed_s >= 0.0
    # One ERR line emitted with err+time+earlier stats:
    assert len(captured) == 1
    scope, verb, target, result, stats = captured[0]
    assert (scope, verb, target, result) == (
        "embed:video",
        "EXTRACT",
        "clip_42",
        "ERR",
    )
    assert "RuntimeError" in stats["err"]
    assert stats["dim"] == 2048
    assert stats["time"] == t.elapsed_s


def test_item_does_not_suppress_keyboard_interrupt(
    captured: list[tuple[Any, ...]],
) -> None:
    cl._scope_var.set("embed:video")
    with pytest.raises(KeyboardInterrupt), cl.item("EXTRACT", "clip_42"):
        raise KeyboardInterrupt
    # No log line emitted — the exception propagates without going through ERR path.
    assert captured == []


def test_item_stats_merge_on_multiple_calls(
    captured: list[tuple[Any, ...]],
) -> None:
    cl._scope_var.set("speech")
    with cl.item("EXTRACT", "clip_7") as t:
        t.stats(lang="ru")
        t.stats(text=180, lang="en")  # later wins on conflict
    assert captured[0][4]["lang"] == "en"
    assert captured[0][4]["text"] == 180


# ---------------------------------------------------------------------------
# T4: event()
# ---------------------------------------------------------------------------


def test_event_logs_one_shot_with_active_scope(
    captured: list[tuple[Any, ...]],
) -> None:
    cl._scope_var.set("ingest")
    cl.event("SCAN", "users", stats={"todo": 240})
    assert captured == [("ingest", "SCAN", "users", "ok", {"todo": 240})]


def test_event_default_result_is_ok(
    captured: list[tuple[Any, ...]],
) -> None:
    cl._scope_var.set("upload")
    cl.event("SKIP", "bucket")
    assert captured[0][3] == "ok"


def test_event_outside_scope_raises() -> None:
    with pytest.raises(RuntimeError, match="no active scope"):
        cl.event("SCAN", "users")


# ---------------------------------------------------------------------------
# T5: warn()
# ---------------------------------------------------------------------------


def test_warn_renders_as_warn_result_with_err_repr(
    captured: list[tuple[Any, ...]],
) -> None:
    cl._scope_var.set("embed:local")
    exc = ConnectionError("service down")
    cl.warn("SCAN", "probe", err=exc)
    assert captured == [
        (
            "embed:local",
            "SCAN",
            "probe",
            "WARN",
            {"err": "ConnectionError('service down')"},
        ),
    ]


def test_warn_accepts_string_err(
    captured: list[tuple[Any, ...]],
) -> None:
    cl._scope_var.set("embed:fleet")
    cl.warn("SCAN", "topup", err="timeout after 30s")
    assert captured[0][4] == {"err": "timeout after 30s"}


def test_warn_outside_scope_raises() -> None:
    with pytest.raises(RuntimeError, match="no active scope"):
        cl.warn("SCAN", "topup")


# ---------------------------------------------------------------------------
# T6: @scope decorator
# ---------------------------------------------------------------------------


def test_scope_literal_sets_contextvar_for_function_body(
    captured: list[tuple[Any, ...]],
) -> None:
    @cl.scope("embed:video")
    def inner() -> None:
        cl.event("SCAN", "clips", stats={"todo": 1})

    inner()
    assert captured[0][0] == "embed:video"


def test_scope_template_binds_function_args(
    captured: list[tuple[Any, ...]],
) -> None:
    @cl.scope("embed:{case}")
    def inner(case: str) -> None:
        cl.event("SCAN", "clips")

    inner(case="sandwich")
    assert captured[0][0] == "embed:sandwich"


def test_scope_unknown_placeholder_raises() -> None:
    @cl.scope("embed:{case}")
    def inner(other: str) -> None:
        cl.event("SCAN", "clips")

    with pytest.raises(KeyError, match="case"):
        inner(other="x")


def test_scope_restores_outer_contextvar_on_exit(
    captured: list[tuple[Any, ...]],
) -> None:
    @cl.scope("embed:sandwich")
    def inner() -> None:
        cl.event("SCAN", "clips")

    cl._scope_var.set("embed")
    inner()
    cl.event("SEAL", "embed")
    assert captured[0][0] == "embed:sandwich"
    assert captured[1][0] == "embed"


def test_scope_invalid_literal_raises() -> None:
    with pytest.raises(ValueError, match="invalid scope"):

        @cl.scope("Embed Video")  # uppercase + space
        def inner() -> None: ...


# ---------------------------------------------------------------------------
# T7: @stage decorator
# ---------------------------------------------------------------------------


def test_stage_emits_seal_with_time_on_clean_exit(
    captured: list[tuple[Any, ...]],
) -> None:
    @cl.stage("mir")
    def run() -> None:
        cl.event("GET", "maest+effnet", stats={"clips": 5})

    run()
    assert len(captured) == 2
    assert captured[0][:4] == ("mir", "GET", "maest+effnet", "ok")
    assert captured[1][:4] == ("mir", "SEAL", "mir", "ok")
    assert "time" in captured[1][4]


def test_stage_emits_seal_with_stageresult_stats(
    captured: list[tuple[Any, ...]],
) -> None:
    @cl.stage("mir")
    def run() -> cl.StageResult:
        return cl.StageResult(done=5, failed=1)

    run()
    seal_stats = captured[-1][4]
    assert seal_stats["done"] == 5
    assert seal_stats["failed"] == 1
    assert "time" in seal_stats


def test_stage_emits_seal_err_and_reraises(
    captured: list[tuple[Any, ...]],
) -> None:
    @cl.stage("mir")
    def run() -> None:
        raise RuntimeError("explode")

    with pytest.raises(RuntimeError, match="explode"):
        run()
    assert captured[-1][:4] == ("mir", "SEAL", "mir", "ERR")
    assert "RuntimeError" in captured[-1][4]["err"]
    assert "time" in captured[-1][4]


def test_stage_invokes_maybe_advance_phase(
    captured: list[tuple[Any, ...]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[None] = []
    monkeypatch.setattr(
        "core.console._maybe_advance_phase",
        lambda: calls.append(None),
    )

    @cl.stage("mir")
    def run() -> None: ...

    run()
    assert len(calls) == 1


def test_stage_invalid_name_raises() -> None:
    with pytest.raises(ValueError, match="invalid scope"):

        @cl.stage("MIR Stage")  # uppercase + space
        def run() -> None: ...


# ---------------------------------------------------------------------------
# T8: Strictness guards
# ---------------------------------------------------------------------------


def test_item_outside_scope_raises() -> None:
    with (
        pytest.raises(RuntimeError, match="no active scope"),
        cl.item("EXTRACT", "clip_42"),
    ):
        pass


def test_item_invalid_verb_raises() -> None:
    cl._scope_var.set("mir")
    # "EMB" is not in the 11-verb closed set
    with pytest.raises(ValueError, match="unknown verb"), cl.item("EMB", "clip_42"):  # type: ignore[arg-type]
        pass


def test_event_invalid_verb_raises() -> None:
    cl._scope_var.set("mir")
    with pytest.raises(ValueError, match="unknown verb"):
        cl.event("ASR", "clip_42")  # type: ignore[arg-type]  # not in closed set


def test_event_invalid_result_raises() -> None:
    cl._scope_var.set("mir")
    with pytest.raises(ValueError, match="unknown result"):
        cl.event("SCAN", "clips", result="stale")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# T9: Integration smoke (real renderer, no monkeypatch)
# ---------------------------------------------------------------------------


def test_real_renderer_smoke(capsys: pytest.CaptureFixture[str]) -> None:
    """Don't mock _render — let core.console actually emit a Rich line."""

    @cl.stage("mir")
    def run() -> cl.StageResult:
        with cl.item("EXTRACT", "clip_1") as t:
            t.stats(dim=2048)
        return cl.StageResult(done=1)

    run()
    out = capsys.readouterr().out
    assert "EXTRACT clip_1 ok" in out
    assert "SEAL mir ok" in out
    assert "time=" in out
    assert "dim=2048" in out
    assert "done=1" in out
