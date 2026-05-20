"""Tests for core/splash.py."""

from __future__ import annotations

import io
from unittest.mock import patch


def _capture_boot(cols: int, rows: int, isatty: bool = True) -> str:
    from core.splash import boot

    buf = io.StringIO()
    buf.isatty = lambda: isatty  # type: ignore[method-assign]
    with (
        patch("core.splash.shutil.get_terminal_size", return_value=(cols, rows)),
        patch("core.splash.sys.stdout", buf),
    ):
        boot()
    return buf.getvalue()


def test_boot_renders_logo_when_terminal_is_wide() -> None:
    from core.splash import _LOGO_LINES

    out = _capture_boot(cols=120, rows=40)
    # Every line of the prerendered logo appears in the output.
    for line in _LOGO_LINES:
        assert line in out
    # Should pad: some blank lines above the art and some below.
    lines = out.splitlines()
    assert len(lines) >= 10  # logo + padding


def test_boot_renders_logo_even_in_narrow_terminal() -> None:
    from core.splash import _LOGO_LINES

    out = _capture_boot(cols=40, rows=24)
    # Logo still prints; centering just no-ops when cols < line width.
    for line in _LOGO_LINES:
        assert line in out


def test_boot_is_noop_when_not_a_tty() -> None:
    out = _capture_boot(cols=120, rows=40, isatty=False)
    assert out == ""
