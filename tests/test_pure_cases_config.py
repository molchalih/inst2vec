"""Config rework: embedding_preprocess + per-case label prompts."""

from __future__ import annotations

from core.config import _load_settings

SETTINGS = _load_settings()


def test_embedding_preprocess_keys():
    pp = SETTINGS.search.embedding_preprocess
    assert "auditory" in pp and pp["auditory"] == "center"
    assert "spoken" in pp and "textual" in pp
    assert "audio" not in pp
    assert "maest" not in pp


def test_case_prompts_resolve():
    cp = SETTINGS.labels.case_prompts
    assert "auditory" in cp
    assert "spoken" in cp and "textual" in cp
    assert "audio" not in cp
    assert "maest" not in cp


def test_cluster_case_prompts_non_empty_for_new_cases():
    ccp = SETTINGS.labels.cluster_case_prompts
    assert "audio" not in ccp
    assert "maest" not in ccp
    for case in ("auditory", "spoken", "textual"):
        assert case in ccp
        assert ccp[case].strip(), f"{case} cluster prompt must be non-empty"
    # Repertoire keys wired into the prompts match the label-spec keys.
    assert "dominant_audio_repertoire" in ccp["spoken"]
    assert "dominant_textual_repertoire" in ccp["textual"]
    assert "dominant_music_repertoire" in ccp["auditory"]
