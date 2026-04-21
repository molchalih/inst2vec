"""Unified console output for the inst2vec pipeline."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from typing import Callable, Generator, Literal

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.text import Text

_console = Console()

_LEVEL_STYLES: dict[str, str] = {
    "ok": "green",
    "warn": "yellow",
    "err": "red",
}


def startup() -> None:
    """print a startup banner."""
    _console.rule(style="dim")
    _console.print(f"  inst2vec execution {datetime.now().strftime('%Y-%m-%d %H:%M')}", style="bold")
    _console.rule(style="dim")
    _console.print()


def phase(name: str) -> None:
    """print a section header."""
    _console.print() # empty line
    title = Text() # create a new text object
    title.append("▸ ", style="dim cyan").append(name, style="bold") # apply styles
    _console.print(title) # print the text object
    _console.print() # empty line


def log(scope: str, msg: str, level: Literal["info", "ok", "warn", "err"] = "info") -> None:
    """print a log line (controls the message text color)."""
    style = _LEVEL_STYLES.get(level, "")
    line = Text()
    line.append(f"[{scope}]", style="dim").append(f" {msg}", style=style) # apply styles
    _console.print(line) # print the text object


@contextmanager
def progress(
    total: int, description: str
) -> Generator[Callable[..., None], None, None]:
    """Context manager yielding advance(n=1, detail="").

    Renders a live rich progress bar for the duration of the block.
    detail is displayed inline after the bar, overwriting on each call.
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("  {task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TextColumn("[dim]{task.fields[detail]}[/dim]"),
        console=_console,
    ) as p:
        task_id = p.add_task(description, total=total, detail="")

        def advance(n: int = 1, detail: str = "") -> None:
            p.update(task_id, advance=n, detail=f"→ {detail}" if detail else "")

        yield advance
