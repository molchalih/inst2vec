"""Tests for the labels case registry rework."""

from __future__ import annotations

from core.pipeline import Stage
from modules.captions.state import SCOPE_CAPTIONS
from modules.labels.cases import REGISTRY
from modules.mir.state import SCOPE_MIR
from modules.speech.state import SCOPE_SPEECH


def test_registry_membership():
    assert "auditory" in REGISTRY and "maest" not in REGISTRY
    assert "spoken" in REGISTRY and "audio" not in REGISTRY
    assert "textual" in REGISTRY
    assert set(REGISTRY) == {"video", "sandwich", "auditory", "spoken", "textual"}


def test_modalities():
    assert REGISTRY["auditory"].modality == "music"
    assert REGISTRY["spoken"].modality == "audio"
    assert REGISTRY["textual"].modality == "textual"


def test_new_cases_skip_clip_pass():
    for name in ("auditory", "spoken", "textual"):
        assert REGISTRY[name].runs_clip_pass is False


def test_dependencies():
    spoken = REGISTRY["spoken"]
    assert (Stage.SPEECH, SCOPE_SPEECH) in spoken.stage1_dependency_stages
    assert (Stage.MIR, SCOPE_MIR) not in spoken.stage1_dependency_stages

    textual = REGISTRY["textual"]
    assert (Stage.CAPTIONS, SCOPE_CAPTIONS) in textual.stage1_dependency_stages

    auditory = REGISTRY["auditory"]
    assert (Stage.MIR, SCOPE_MIR) in auditory.stage1_dependency_stages


def test_consumes_nothing():
    for name in ("auditory", "spoken", "textual"):
        assert REGISTRY[name].consumes_label_cases == ()


def test_repertoire_keys():
    assert REGISTRY["auditory"].repertoire_key == "dominant_music_repertoire"
    assert REGISTRY["spoken"].repertoire_key == "dominant_audio_repertoire"
    assert REGISTRY["textual"].repertoire_key == "dominant_textual_repertoire"


def test_textual_clip_key_shape():
    spec = REGISTRY["textual"]
    assert spec.observable_key == "observable_textual_tags"
    assert spec.sentence_key == "one_sentence_textual_reading"
    assert "observable_textual_tags" in spec.clip_required_keys


def test_none_input_errors():
    assert REGISTRY["spoken"].none_input_error == "no_speech"
    assert REGISTRY["textual"].none_input_error == "no_caption"
