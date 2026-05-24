"""Placeholder for the Stage-2 export-atlas tests."""

import pytest


def test_stage1_placeholder():
    """Stage 1 ships a stub; the real tests land in Stage 2."""
    from scripts import export_atlas

    with pytest.raises(NotImplementedError):
        export_atlas.main()
