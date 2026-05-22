"""Pinned recipe-version and dependency-column assertions.

Any change to the case recipe (text builder logic, depended-on columns)
must be a deliberate version bump that invalidates sealed embeddings.
"""

from __future__ import annotations

from modules.embeddings.cases import AUDIO_CASE, GEMINI_CASE, MAEST_CASE, SANDWICH_CASE


def test_sandwich_recipe_version_is_v3():
    assert SANDWICH_CASE.recipe_version == "sandwich_v3"


def test_audio_recipe_version_is_v3():
    assert AUDIO_CASE.recipe_version == "audio_v3"


def test_gemini_recipe_version_is_v2():
    assert GEMINI_CASE.recipe_version == "gemini_v2"


def test_maest_recipe_version_is_v1():
    assert MAEST_CASE.recipe_version == "maest_v1"


def test_sandwich_depends_on_audio_mir_row():
    assert "_audio_mir_row" in SANDWICH_CASE.dependency_columns
    assert "_music_row" not in SANDWICH_CASE.dependency_columns


def test_audio_depends_on_audio_mir_row():
    assert "_audio_mir_row" in AUDIO_CASE.dependency_columns
    assert "_music_row" not in AUDIO_CASE.dependency_columns


def test_sandwich_depends_on_is_speech_detected():
    assert "is_speech_detected" in SANDWICH_CASE.dependency_columns


def test_audio_depends_on_is_speech_detected():
    assert "is_speech_detected" in AUDIO_CASE.dependency_columns


def test_gemini_depends_on_is_speech_detected():
    assert "is_speech_detected" in GEMINI_CASE.dependency_columns


def test_maest_depends_only_on_audio_file_stat():
    assert MAEST_CASE.dependency_columns == ("_audio_file_stat",)
