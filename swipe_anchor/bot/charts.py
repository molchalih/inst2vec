"""Render the admin ``/stats`` charts as PNG bytes (dark, on-brand, Agg-only).

Each function returns a standalone PNG so the bot can send one chart per message.
Palette mirrors the Mini App tokens (canvas ``#0b0b0f``, accent ``#ff3b1f``,
teal ``#37e0c4``) so the graphs feel like part of the same product.
"""

from __future__ import annotations

import io
from datetime import datetime

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

BG = "#0b0b0f"
SURFACE = "#15151c"
FG = "#f4f1ea"
MUTED = "#9a978f"
FAINT = "#5f5d58"
ACCENT = "#ff3b1f"
AFFIRM = "#37e0c4"
VIOLET = "#7c5cff"
AMBER = "#f5a623"

_GRID = "#23232c"


def _base_axes(width: float = 7.2, height: float = 4.3):
    fig, ax = plt.subplots(figsize=(width, height), dpi=200)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(axis="y", color=_GRID, linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    return fig, ax


def _title(ax, title: str, subtitle: str = "") -> None:
    # Both placed above the axes in axes-fraction coords with a clear vertical gap
    # (a plain set_title + text would collide). bbox_inches="tight" keeps them.
    ax.text(
        0.0,
        1.16,
        title,
        transform=ax.transAxes,
        color=FG,
        fontsize=15,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    if subtitle:
        ax.text(
            0.0,
            1.05,
            subtitle,
            transform=ax.transAxes,
            color=MUTED,
            fontsize=9.5,
            ha="left",
            va="bottom",
        )


def _render(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
        pad_inches=0.35,
    )
    plt.close(fig)
    return buf.getvalue()


def _gradient_under(ax, x, y, color: str) -> None:
    """Fill the area under (x, y) with a vertical fade — the 'premium' look."""
    rgb = mcolors.to_rgb(color)
    grad = np.empty((256, 1, 4))
    grad[:, :, :3] = rgb
    grad[:, :, 3] = np.linspace(0.0, 0.45, 256)[:, None]
    ymax = float(max(y)) if len(y) else 1.0
    im = ax.imshow(
        grad,
        aspect="auto",
        origin="lower",
        extent=[float(min(x)), float(max(x)), 0.0, ymax],
        zorder=1,
    )
    verts = np.column_stack(
        [np.concatenate([x, x[::-1]]), np.concatenate([y, np.zeros_like(y)])]
    )
    clip = Polygon(verts, closed=True, transform=ax.transData)
    im.set_clip_path(clip)


def cumulative_chart(response_times: list[str]) -> bytes:
    """Cumulative datapoints (judgments) collected over time — the headline."""
    times = sorted(datetime.fromisoformat(t) for t in response_times)
    fig, ax = _base_axes()
    n = len(times)
    if n == 1:
        x = np.array([mdates.date2num(times[0]) - 0.02, mdates.date2num(times[0])])
        y = np.array([0.0, 1.0])
    else:
        x = mdates.date2num(times)
        y = np.arange(1, n + 1, dtype=float)

    ax.plot(x, y, color=ACCENT, lw=2.6, solid_capstyle="round", zorder=3)
    ax.scatter([x[-1]], [y[-1]], s=70, color=ACCENT, zorder=4, edgecolors=BG, linewidths=2)
    _gradient_under(ax, x, y, ACCENT)

    ax.text(
        x[-1],
        y[-1],
        f"  {int(y[-1])}",
        color=FG,
        fontsize=13,
        fontweight="bold",
        va="center",
        ha="left",
    )
    ax.set_ylim(0, max(1.0, float(y[-1])) * 1.18)
    ax.margins(x=0.02)
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    _title(ax, "datapoints collected", "cumulative judgments over time")
    return _render(fig)


def status_donut(comparisons: dict) -> bytes:
    """Comparisons by lifecycle status as a donut, with the total in the hole."""
    order = [
        ("retired", AFFIRM),
        ("ambiguous", VIOLET),
        ("gold", AMBER),
        ("open", FAINT),
    ]
    vals = [(k, comparisons.get(k, 0), c) for k, c in order]
    vals = [v for v in vals if v[1] > 0]
    fig, ax = _base_axes(width=6.0, height=5.0)
    ax.grid(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=BG)
    total = sum(v[1] for v in vals)
    wedges, _ = ax.pie(
        [v[1] for v in vals],
        colors=[v[2] for v in vals],
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.42, "edgecolor": BG, "linewidth": 4},
    )
    ax.text(0, 0.12, str(total), color=FG, fontsize=30, fontweight="bold", ha="center", va="center")
    ax.text(0, -0.18, "comparisons", color=MUTED, fontsize=11, ha="center", va="center")
    ax.legend(
        wedges,
        [f"{k} · {v}" for k, v, _ in vals],
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
        labelcolor=FG,
        fontsize=11,
    )
    ax.set_title("comparisons by status", color=FG, fontsize=15, fontweight="bold", loc="left", pad=12)
    return _render(fig)


def contributors_bar(per_annotator: list[dict]) -> bytes:
    """Top contributors by judgments submitted — horizontal bars, biggest on top."""
    data = list(per_annotator)[:10]
    data.reverse()  # biggest ends up on top in a horizontal bar
    labels = [d["label"] for d in data]
    vals = [int(d["n"]) for d in data]
    fig, ax = _base_axes(width=7.2, height=max(3.0, 0.5 * len(data) + 1.5))
    ax.grid(axis="x", color=_GRID, linewidth=0.8, alpha=0.7)
    ax.grid(axis="y", visible=False)
    y = np.arange(len(data))
    ax.barh(y, vals, color=AFFIRM, height=0.62, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=FG, fontsize=10)
    top = max(vals) if vals else 1
    for yi, v in zip(y, vals, strict=True):
        ax.text(v + top * 0.015, yi, str(v), color=FG, va="center", ha="left", fontsize=10, fontweight="bold")
    ax.set_xlim(0, top * 1.12)
    _title(ax, "top contributors", "judgments submitted per person")
    return _render(fig)
