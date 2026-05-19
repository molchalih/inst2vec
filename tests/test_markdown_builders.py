"""Unit tests for docs/reporting/_markdown.py table builders."""

from __future__ import annotations

import pytest

from docs.reporting._markdown import (
    build_metric_value_table,
    build_multi_column_table,
)


def test_metric_value_table_basic():
    out = build_metric_value_table([("Users", "1,234"), ("Clips", "9,876")])
    assert out == "\n".join(
        [
            "| Metric | Value |",
            "|---|---:|",
            "| Users | 1,234 |",
            "| Clips | 9,876 |",
        ]
    )


def test_metric_value_table_left_aligned_value():
    out = build_metric_value_table([("Range", "0-1")], value_align="---")
    assert out.splitlines()[1] == "|---|---|"


def test_metric_value_table_preserves_latex_math_in_labels():
    out = build_metric_value_table([(r"$N$", "10")])
    assert "$N$" in out


def test_metric_value_table_custom_metric_header():
    out = build_metric_value_table([("x", "1")], metric_header="Field")
    assert out.splitlines()[0] == "| Field | Value |"


def test_multi_column_table_basic():
    out = build_multi_column_table(
        first_col_header="Metric",
        column_headers=("video", "audio"),
        rows=[
            ("DBCV", ["0.42", "0.31"]),
            ("k", ["8", "5"]),
        ],
    )
    assert out == "\n".join(
        [
            "| Metric | video | audio |",
            "|---|---:|---:|",
            "| DBCV | 0.42 | 0.31 |",
            "| k | 8 | 5 |",
        ]
    )


def test_multi_column_table_with_caption():
    out = build_multi_column_table(
        first_col_header="Field",
        column_headers=("a",),
        rows=[("x", ["1"])],
        caption="Cap.",
    )
    lines = out.splitlines()
    assert lines[-2] == ""
    assert lines[-1] == ": Cap."


def test_multi_column_table_rejects_empty_columns():
    with pytest.raises(ValueError):
        build_multi_column_table(
            first_col_header="Metric",
            column_headers=(),
            rows=[],
        )


def test_multi_column_table_single_column():
    out = build_multi_column_table(
        first_col_header="Metric",
        column_headers=("only",),
        rows=[("row1", ["v1"])],
    )
    lines = out.splitlines()
    assert lines[0] == "| Metric | only |"
    assert lines[1] == "|---|---:|"
    assert lines[2] == "| row1 | v1 |"
