"""Tests for gemini_mm case gating and explicit-request rejection."""

from types import SimpleNamespace

import pytest

from modules.embeddings import embed_clip_embeddings
from modules.embeddings.cases import default_cases


def _stub_settings(gemini_enabled: bool):
    """Create a minimal settings stub for gating tests."""
    return SimpleNamespace(embeddings=SimpleNamespace(gemini_enabled=gemini_enabled))


def test_default_cases_excludes_gemini_when_disabled():
    """default_cases should not include gemini_mm when gemini_enabled=False."""
    assert "gemini_mm" not in default_cases(_stub_settings(gemini_enabled=False))


def test_default_cases_includes_gemini_when_enabled():
    """default_cases should include gemini_mm when gemini_enabled=True."""
    assert "gemini_mm" in default_cases(_stub_settings(gemini_enabled=True))


def test_explicit_gemini_request_raises_when_disabled():
    """Requesting gemini_mm explicitly should raise when gemini_enabled=False."""
    settings = _stub_settings(gemini_enabled=False)
    with pytest.raises(RuntimeError, match="gemini_enabled"):
        embed_clip_embeddings(settings, cases=["gemini_mm"])
