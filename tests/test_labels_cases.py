"""Round-trip the per-case ``LabelCaseSpec`` registry."""

from __future__ import annotations

from types import SimpleNamespace

from core.pipeline import Stage
from modules.labels.cases import REGISTRY, LabelCaseSpec


def test_registry_has_all_default_cases() -> None:
    expected = {"video", "audio", "sandwich", "maest", "gemini"}
    assert set(REGISTRY) == expected
    for name, spec in REGISTRY.items():
        assert isinstance(spec, LabelCaseSpec)
        assert spec.name == name


def test_each_case_has_required_clip_keys_and_cluster_keys() -> None:
    expected_clip = {
        "video": {
            "observable_visual_tags",
            "aesthetic_tags",
            "community_signalling_tags",
            "one_sentence_visual_reading",
        },
        "audio": {
            "observable_audio_tags",
            "aesthetic_tags",
            "community_signalling_tags",
            "one_sentence_audio_reading",
        },
        "sandwich": {
            "observable_multimodal_tags",
            "aesthetic_tags",
            "community_signalling_tags",
            "one_sentence_multimodal_reading",
        },
        "maest": {
            "observable_music_tags",
            "aesthetic_tags",
            "community_signalling_tags",
            "one_sentence_music_reading",
        },
    }
    # gemini follows the sandwich recipe.
    expected_clip["gemini"] = expected_clip["sandwich"]

    expected_repertoire = {
        "video": "dominant_visual_repertoire",
        "audio": "dominant_audio_repertoire",
        "sandwich": "dominant_multimodal_repertoire",
        "maest": "dominant_music_repertoire",
        "gemini": "dominant_multimodal_repertoire",
    }
    common = {
        "cluster_label",
        "cluster_summary",
        "dominant_aesthetic_logic",
        "taste_signalling",
        "visibility_orientation",
        "internal_variations",
        "boundary_notes",
        "tool_tags",
    }

    for case, spec in REGISTRY.items():
        assert set(spec.clip_required_keys) == expected_clip[case], case
        assert set(spec.cluster_required_keys) == common | {
            expected_repertoire[case]
        }, case


def test_video_clip_uses_video_true_others_false() -> None:
    for case, spec in REGISTRY.items():
        assert spec.clip_uses_video is (case == "video"), case


def test_stage1_dependency_stages_match_spec() -> None:
    # Each upstream stage seals its own scope row in StageState; pairing the
    # scope with the stage in the spec is what stops _dependency_hash from
    # silently looking up the missing-row digest under the wrong scope.
    speech = (Stage.SPEECH, "all")
    captions = (Stage.CAPTIONS, "all")
    mir = (Stage.MIR, "all")
    expected = {
        "video": (),
        "audio": (speech, mir),
        "sandwich": (captions, speech, mir),
        "maest": (mir,),
        "gemini": (captions, speech, mir),
    }
    for case, stages in expected.items():
        assert REGISTRY[case].stage1_dependency_stages == stages, case


def test_audio_input_returns_speech_or_music_text() -> None:
    spec = REGISTRY["audio"]
    clip = SimpleNamespace(
        is_speech_detected=True,
        speech_language="en",
        speech_translation=None,
        speech_transcription="hello world from a podcast",
    )
    mir = None
    text = spec.clip_input(clip, mir, None)
    assert text is not None
    assert "hello world" in text


def test_audio_input_returns_none_when_no_speech_and_no_music() -> None:
    spec = REGISTRY["audio"]
    clip = SimpleNamespace(
        is_speech_detected=False,
        speech_language=None,
        speech_translation=None,
        speech_transcription=None,
    )
    assert spec.clip_input(clip, None, None) is None


def test_sandwich_input_returns_none_when_visual_payload_missing() -> None:
    spec = REGISTRY["sandwich"]
    clip = SimpleNamespace(
        caption_language="en",
        caption_translation=None,
        caption_clean="hi",
        caption_text="hi",
        is_speech_detected=False,
        speech_language=None,
        speech_translation=None,
        speech_transcription=None,
    )
    assert spec.clip_input(clip, None, None) is None


def test_sandwich_input_embeds_visual_payload_when_present() -> None:
    spec = REGISTRY["sandwich"]
    clip = SimpleNamespace(
        caption_language="en",
        caption_translation=None,
        caption_clean="a kitchen scene",
        caption_text="a kitchen scene",
        is_speech_detected=False,
        speech_language=None,
        speech_translation=None,
        speech_transcription=None,
    )
    visual = {"one_sentence_visual_reading": "warm kitchen vignette"}
    text = spec.clip_input(clip, None, visual)
    assert text is not None
    assert "warm kitchen vignette" in text
    assert "a kitchen scene" in text


def test_maest_input_returns_none_when_music_not_detected() -> None:
    spec = REGISTRY["maest"]
    mir = SimpleNamespace(is_music_detected=False)
    assert spec.clip_input(None, mir, None) is None


def test_video_clip_input_is_sentinel_returning_none() -> None:
    spec = REGISTRY["video"]
    # The video case routes through ``LabelsGenerator.run`` directly; the
    # input adapter must always return ``None`` so it cannot accidentally
    # feed text into the visual branch.
    assert spec.clip_input(None, None, None) is None


def test_consumes_label_cases_reference_known_cases() -> None:
    for spec in REGISTRY.values():
        for dep in spec.consumes_label_cases:
            assert dep in REGISTRY, (spec.name, dep)


def test_observable_sentence_repertoire_keys_match_required_sets() -> None:
    for spec in REGISTRY.values():
        assert spec.observable_key in spec.clip_required_keys
        assert spec.sentence_key in spec.clip_required_keys
        assert spec.repertoire_key in spec.cluster_required_keys


def test_video_case_has_no_none_input_error() -> None:
    assert REGISTRY["video"].none_input_error is None


def test_non_video_cases_declare_none_input_error() -> None:
    for name, spec in REGISTRY.items():
        if name == "video":
            continue
        assert spec.none_input_error is not None
