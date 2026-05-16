from modules.embeddings.cases import (
    CASE_REGISTRY,
    DEFAULT_CASES,
    EmbeddingCaseSpec,
)


def test_default_cases_exact_tuple():
    assert DEFAULT_CASES == ("video", "sandwich", "audio")


def test_registry_contains_all_default_cases():
    for name in DEFAULT_CASES:
        assert name in CASE_REGISTRY
        assert isinstance(CASE_REGISTRY[name], EmbeddingCaseSpec)
        assert CASE_REGISTRY[name].name == name


def test_video_case_shape():
    spec = CASE_REGISTRY["video"]
    assert spec.requires_video is True
    assert spec.apply_video_token_fallback is True
    assert spec.text_builder is None


def test_sandwich_case_shape():
    spec = CASE_REGISTRY["sandwich"]
    assert spec.requires_video is True
    assert spec.apply_video_token_fallback is True
    assert spec.text_builder is not None


def test_audio_case_shape():
    spec = CASE_REGISTRY["audio"]
    assert spec.requires_video is False
    assert spec.apply_video_token_fallback is False
    assert spec.text_builder is not None


def test_spec_has_no_instruction_or_requires_text_fields():
    """Regression guard: these fields were removed because they were
    dead/redundant. instruction lives inside payload builders; text
    usage is derived from text_builder is not None.
    """
    field_names = {f.name for f in EmbeddingCaseSpec.__dataclass_fields__.values()}
    assert "instruction" not in field_names
    assert "requires_text" not in field_names
