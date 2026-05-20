"""Universal worker console for the inst2vec pipeline.

One log helper, one regex highlighter, two stacked progress bars (current
stage above, pipeline-wide N/16 below) rendered inside a shared Live group
that stays glued to the bottom of the terminal.

Canonical worker line (rendered):

    [HH:MM:SS]  VERB target result [k1=v1, k2=v2]   scope

The timestamp column and scope column de-duplicate against the previously
rendered line so consecutive lines from the same scope in the same second
show clean output without repetition.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar, Literal, cast, get_args

from rich.console import Console, Group
from rich.highlighter import RegexHighlighter
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.style import Style
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# ---------------------------------------------------------------------------
# Verb vocabulary (closed set, 17)
# ---------------------------------------------------------------------------

Verb = Literal[
    "INIT",
    "LOAD",
    "SCAN",
    "SKIP",
    "GET",
    "PUT",
    "EXTRACT",
    "ID",
    "ASR",
    "MT",
    "CLEAN",
    "EMB",
    "AGG",
    "FIT",
    "SCORE",
    "WRITE",
    "SEAL",
]
_VERBS: frozenset[str] = frozenset(get_args(Verb))


# ---------------------------------------------------------------------------
# Highlighter
# ---------------------------------------------------------------------------

_WORKER_REGEX = (
    r"^(?P<verb>[A-Z]+)\s+"
    r"(?P<target>\S+)\s+"
    r"(?P<result>\S+)"
    r"(?:\s+\[(?P<stats>[^\]]+)\])?$"
)

_WORKER_PATTERN = re.compile(_WORKER_REGEX)


class _WorkerHighlighter(RegexHighlighter):
    base_style = "w."
    highlights: ClassVar[list[str]] = [_WORKER_REGEX]


_THEME_OK = Theme(
    {
        "w.verb": Style.parse("bold yellow"),
        "w.target": Style.parse("magenta"),
        "w.result": Style.parse("green"),
        "w.stats": Style.parse("dim"),
        "log.time": Style.parse("dim cyan"),
        "log.scope": Style.parse("dim"),
    }
)

# ---------------------------------------------------------------------------
# Console + shared progress group
# ---------------------------------------------------------------------------

_console = Console(theme=_THEME_OK, highlighter=_WorkerHighlighter())

# Fixed-width description column shared by both progress bars; the pipeline bar
# adds a 2-space lead to compensate for the stage bar's SpinnerColumn (~2 cells).
_DESC_WIDTH = 28

_stage_progress = Progress(
    SpinnerColumn(),
    TextColumn(f"{{task.description:<{_DESC_WIDTH}}}"),
    BarColumn(),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
    TextColumn("[dim]{task.fields[detail]}[/dim]"),
    console=_console,
    transient=False,
)

_pipeline_progress = Progress(
    TextColumn(f"  [bold blue]{{task.description:<{_DESC_WIDTH}}}"),
    BarColumn(),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
    console=_console,
    transient=False,
)

_live: Live | None = None
_pipeline_task_id: TaskID | None = None


# ---------------------------------------------------------------------------
# Render state (dedup)
# ---------------------------------------------------------------------------


@dataclass
class _RenderState:
    last_time: str = field(default="")
    last_scope: str = field(default="")


_render_state = _RenderState()


# ---------------------------------------------------------------------------
# Stats formatters
# ---------------------------------------------------------------------------


def _format_time(seconds: float) -> str:
    if seconds < 0.1:
        return f"{round(seconds * 1000)}ms"
    if seconds < 10.0:
        return f"{seconds:.2f}s"
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    mins = int(seconds // 60)
    secs = int(seconds - mins * 60)
    return f"{mins}m{secs:02d}s"


def _format_size(n_bytes: int) -> str:
    size: float = float(n_bytes)
    if size < 1000:
        return f"{int(size)}B"
    for unit in ("KB", "MB", "GB"):
        size /= 1000
        if size < 1000 or unit == "GB":
            return f"{size:.1f}{unit}"
    return f"{size:.1f}GB"  # unreachable


def _format_stats(stats: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key, value in stats.items():
        if key == "err":
            continue  # rendered as separate indented line
        if key == "time":
            parts.append(f"time={_format_time(float(value))}")
        elif key == "size":
            parts.append(f"size={_format_size(int(value))}")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------


def _render_log(scope: str, body: str, *, is_err: bool = False) -> None:
    """Render one worker line in Rich-log style.

    Layout: ``[HH:MM:SS]  body  scope`` where the time column and the
    scope column are *deduplicated* — if either matches the previously
    rendered line, that column is blanked (but its width is preserved
    so the body column does not shift).

    All output goes through `_console`, the single console owned by
    the shared `Live` group, so worker lines never scroll the
    bottom-pinned progress bars off-screen.
    """
    now = datetime.now().strftime("%H:%M:%S")
    time_prefix = f"[{now}]"

    show_time = time_prefix != _render_state.last_time
    show_scope = scope != _render_state.last_scope

    time_cell = Text(
        time_prefix if show_time else " " * len(time_prefix),
        style="log.time",
    )
    scope_cell = Text(scope if show_scope else " " * len(scope), style="log.scope")
    body_text = Text(body)
    cast(_WorkerHighlighter, _console.highlighter).highlight(body_text)
    if is_err:
        m = _WORKER_PATTERN.fullmatch(body)
        if m is not None:
            # `bold red` is layered over the highlighter's `w.result` span;
            # Rich merges them and the bolder/redder style wins on render.
            body_text.stylize("bold red", m.start("result"), m.end("result"))

    grid = Table.grid(expand=True, padding=(0, 1))
    grid.add_column(width=len(time_prefix), no_wrap=True)
    grid.add_column(ratio=1, overflow="fold")
    grid.add_column(justify="right", no_wrap=True)
    grid.add_row(time_cell, body_text, scope_cell)

    _console.print(grid)

    _render_state.last_time = time_prefix
    _render_state.last_scope = scope


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def phase(name: str) -> None:
    """Advance the pipeline bar by 1 and reset scope dedup.

    The phase name is intentionally not rendered: stage progress + worker
    log lines already convey what is running.
    """
    del name  # kept in signature for call-site readability
    _render_state.last_scope = ""
    if _pipeline_task_id is not None:
        _pipeline_progress.update(_pipeline_task_id, advance=1)


def log(
    scope: str,
    verb: str,
    target: str,
    result: str = "ok",
    *,
    stats: Mapping[str, Any] | None = None,
) -> None:
    """Emit a structured worker line.

    Output (rendered): ``[HH:MM:SS]  VERB target result [k=v, ...]   scope``
    where the time column and scope column de-duplicate against the
    previously rendered line.
    """
    if verb not in _VERBS:
        raise ValueError(f"unknown verb {verb!r}; allowed: {sorted(_VERBS)}")

    body = f"{verb} {target} {result}"
    if stats:
        rendered_stats = _format_stats(stats)
        if rendered_stats:
            body += f" [{rendered_stats}]"

    is_err = result.upper() == "ERR"
    _render_log(scope, body, is_err=is_err)

    if stats and "err" in stats:
        _console.print(f"   └─ err: {stats['err']}", style="dim red", highlight=False)


@contextmanager
def progress(total: int, description: str) -> Iterator[Callable[..., None]]:
    """Add a stage task to the shared `_stage_progress` for this with-block.

    The stage task renders above the pipeline task because `_stage_progress`
    is first in the Live group (see `pipeline()`).
    """
    task_id = _stage_progress.add_task(description, total=total, detail="")

    def advance(n: int = 1, detail: str = "") -> None:
        _stage_progress.update(
            task_id, advance=n, detail=f"→ {detail}" if detail else ""
        )

    try:
        yield advance
    finally:
        _stage_progress.remove_task(task_id)


@contextmanager
def pipeline(total_stages: int) -> Iterator[None]:
    """Wrap the full pipeline run. Owns the shared Live + pipeline bar."""
    global _live, _pipeline_task_id
    _render_state.last_time = ""
    _render_state.last_scope = ""
    _pipeline_task_id = _pipeline_progress.add_task("Pipeline", total=total_stages)
    group = Group(_stage_progress, _pipeline_progress)
    _live = Live(group, console=_console, refresh_per_second=10, transient=False)
    _live.start()
    try:
        yield
    finally:
        _live.stop()
        _pipeline_progress.remove_task(_pipeline_task_id)
        _live = None
        _pipeline_task_id = None
