"""Smoke test: the package re-exports the pipeline entry point."""

from __future__ import annotations


def test_run_mir_is_the_pipeline_function():
    """No rebinding wrapper — `modules.mir.run_mir` IS the impl."""
    from modules.mir import run_mir
    from modules.mir.pipeline import run_mir as impl

    assert run_mir is impl
