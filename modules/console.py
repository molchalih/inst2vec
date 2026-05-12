"""Unified console output for the inst2vec pipeline."""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from datetime import datetime
from typing import Literal

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
    _console.rule(style="dim")
    _console.print(
        f"inst2vec execution {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        style="bold",
    )
    _console.rule(style="dim")
    _console.print()


def phase(name: str) -> None:
    _console.print()
    title = Text()
    title.append("▸ ", style="dim cyan").append(name, style="bold")
    _console.print(title)
    _console.print()


def log(
    scope: str, msg: str, level: Literal["info", "ok", "warn", "err"] = "info"
) -> None:
    style = _LEVEL_STYLES.get(level, "")
    line = Text()
    line.append(f"[{scope}]", style="dim").append(
        f" {msg}", style=style
    )
    _console.print(line)


@contextmanager
def progress(
    total: int, description: str
) -> Generator[Callable[..., None], None, None]:
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
