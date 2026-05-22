"""Pinned recipe-version and dependency-column assertions for speech-aware cases.

These pins exist so any future change to the case recipe (text builder
logic, depended-on columns) is forced through a deliberate version bump
that invalidates sealed embeddings.
"""

from __future__ import annotations

from modules.embeddings.cases import AUDIO_CASE, GEMINI_CASE, SANDWICH_CASE


def test_sandwich_recipe_version_is_v2():
    assert SANDWICH_CASE.recipe_version == "sandwich_v2"


def test_audio_recipe_version_is_v2():
    assert AUDIO_CASE.recipe_version == "audio_v2"


def test_gemini_recipe_version_is_v2():
    assert GEMINI_CASE.recipe_version == "gemini_v2"


def test_sandwich_depends_on_is_speech_detected():
    assert "is_speech_detected" in SANDWICH_CASE.dependency_columns


def test_audio_depends_on_is_speech_detected():
    assert "is_speech_detected" in AUDIO_CASE.dependency_columns


def test_gemini_depends_on_is_speech_detected():
    assert "is_speech_detected" in GEMINI_CASE.dependency_columns
