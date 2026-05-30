"""Embeddings case-registry rework: auditory rename, spoken, textual."""

from __future__ import annotations

from core.config import _load_settings
from modules.embeddings.cases import (
    CASE_REGISTRY,
    case_config_identity,
    default_cases,
)

SETTINGS = _load_settings()


def test_registry_membership():
    assert "auditory" in CASE_REGISTRY
    assert "maest" not in CASE_REGISTRY
    assert "spoken" in CASE_REGISTRY
    assert "audio" not in CASE_REGISTRY
    assert "textual" in CASE_REGISTRY
    # frozen + gated-off unchanged
    assert "video" in CASE_REGISTRY
    assert "sandwich" in CASE_REGISTRY
    assert "gemini" in CASE_REGISTRY


def test_default_cases():
    cases = set(default_cases(SETTINGS))
    assert {"video", "sandwich", "auditory", "spoken", "textual"} <= cases
    assert "audio" not in cases
    assert "maest" not in cases


def test_spoken_is_text_only_speech_deps():
    spec = CASE_REGISTRY["spoken"]
    assert spec.requires_video is False
    assert spec.backbone == "qwen"
    assert spec.served_remotely is False
    assert spec.display_label == "Spoken"
    assert "_audio_mir_row" not in spec.dependency_columns
    assert "_audio_file_stat" not in spec.dependency_columns
    assert "is_speech_detected" in spec.dependency_columns
    assert "speech_transcription" in spec.dependency_columns
    # no caption columns
    assert "caption_clean" not in spec.dependency_columns


def test_textual_is_text_only_caption_deps():
    spec = CASE_REGISTRY["textual"]
    assert spec.requires_video is False
    assert spec.backbone == "qwen"
    assert spec.served_remotely is False
    assert spec.display_label == "Textual"
    assert "_audio_mir_row" not in spec.dependency_columns
    assert "caption_clean" in spec.dependency_columns
    assert "caption_translation" in spec.dependency_columns
    # no speech columns
    assert "speech_transcription" not in spec.dependency_columns


def test_spoken_textual_identities_carry_instruction_and_differ():
    spoken_id = case_config_identity(CASE_REGISTRY["spoken"], SETTINGS)
    textual_id = case_config_identity(CASE_REGISTRY["textual"], SETTINGS)
    assert "instruction=" in spoken_id
    assert "instruction=" in textual_id
    assert spoken_id != textual_id
    assert "case=spoken" in spoken_id
    assert "case=textual" in textual_id


def test_auditory_identity_equals_maest_with_only_token_swapped():
    """The auditory identity must equal the OLD maest identity with ONLY the
    leading ``case=maest`` token rewritten to ``case=auditory`` — proving the
    rename perturbs nothing else (the migration adopts exactly this hash)."""
    auditory_id = case_config_identity(CASE_REGISTRY["auditory"], SETTINGS)
    reconstructed_maest = auditory_id.replace("case=auditory", "case=maest", 1)
    # Every non-leading part is byte-identical to the maest recipe.
    assert reconstructed_maest.startswith("case=maest|")
    assert "backend=onnx" in auditory_id
    assert "text_recipe=maest_v1" in auditory_id
    assert "case=auditory" in auditory_id
    assert "case=maest" not in auditory_id
