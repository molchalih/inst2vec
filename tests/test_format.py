"""Unit tests for docs/reporting/_format.py number formatters."""

from __future__ import annotations

import math

from docs.reporting._format import (
    fmt_count_share,
    fmt_distribution,
    fmt_float,
    fmt_int,
    fmt_mean,
    fmt_median,
    fmt_minmax,
)


def test_fmt_int_formats_thousands():
    assert fmt_int(1234567) == "1,234,567"


def test_fmt_int_handles_none():
    assert fmt_int(None) == "-"


def test_fmt_float_default_four_places():
    assert fmt_float(0.123456) == "0.1235"


def test_fmt_float_custom_places():
    assert fmt_float(0.5, places=2) == "0.50"


def test_fmt_float_handles_none():
    assert fmt_float(None) == "-"


def test_fmt_float_handles_nan():
    assert fmt_float(float("nan")) == "-"


def test_fmt_float_handles_inf():
    assert fmt_float(math.inf) == "-"


def test_fmt_count_share_basic():
    assert fmt_count_share(123, 1000) == "123 (12.3%)"


def test_fmt_count_share_zero_total():
    assert fmt_count_share(0, 0) == "0"


def test_fmt_count_share_negative_total():
    assert fmt_count_share(5, -1) == "0"


def test_fmt_median_empty():
    assert fmt_median([]) == "-"


def test_fmt_median_basic():
    assert fmt_median([1, 2, 3, 4, 5]) == "3"


def test_fmt_mean_empty():
    assert fmt_mean([]) == "-"


def test_fmt_mean_basic():
    assert fmt_mean([1.0, 2.0, 3.0]) == "2.0"


def test_fmt_minmax_empty():
    assert fmt_minmax([]) == "-"


def test_fmt_minmax_basic():
    assert fmt_minmax([5, 1, 9, 3]) == "1-9"


def test_fmt_distribution_empty():
    assert fmt_distribution([]) == ("-", "-", "-")


def test_fmt_distribution_basic():
    median, mean, minmax = fmt_distribution([10, 20, 30, 40, 50])
    assert median == "30"
    assert mean == "30.0"
    assert minmax == "10-50"
