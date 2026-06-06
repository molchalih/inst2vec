"""The stats charts render to non-empty PNG bytes for realistic + edge inputs."""

from __future__ import annotations

from swipe_anchor.bot import charts

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_cumulative_chart_many_points() -> None:
    times = [f"2026-01-01T00:{m:02d}:00+00:00" for m in range(0, 40, 3)]
    png = charts.cumulative_chart(times)
    assert png.startswith(PNG_MAGIC)
    assert len(png) > 1000


def test_cumulative_chart_single_point() -> None:
    png = charts.cumulative_chart(["2026-01-01T00:00:00+00:00"])
    assert png.startswith(PNG_MAGIC)


def test_status_donut() -> None:
    png = charts.status_donut({"open": 5, "retired": 12, "ambiguous": 3, "gold": 2})
    assert png.startswith(PNG_MAGIC)


def test_status_donut_partial_zero_categories() -> None:
    png = charts.status_donut({"open": 0, "retired": 4, "ambiguous": 0, "gold": 0})
    assert png.startswith(PNG_MAGIC)


def test_contributors_bar() -> None:
    per = [{"label": "dasha", "n": 30}, {"label": "bob", "n": 12}, {"label": "tg_ab…", "n": 4}]
    png = charts.contributors_bar(per)
    assert png.startswith(PNG_MAGIC)
