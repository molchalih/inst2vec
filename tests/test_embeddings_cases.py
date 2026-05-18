from types import SimpleNamespace

from modules.embeddings.cases import (
    CASE_REGISTRY,
    VIDEO_CASE,
    EmbeddingCaseSpec,
    case_config_identity,
)


def test_default_cases_exact_tuple():
    assert tuple(name for name, spec in CASE_REGISTRY.items() if not spec.requires) == (
        "video",
        "sandwich",
        "audio",
    )


def test_registry_contains_all_default_cases():
    for name, spec in CASE_REGISTRY.items():
        if spec.requires:
            continue
        assert isinstance(spec, EmbeddingCaseSpec)
        assert spec.name == name


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


def _identity_settings() -> SimpleNamespace:
    """Minimal settings stub satisfying case_config_identity's reads."""
    return SimpleNamespace(
        paths=SimpleNamespace(model_path="/fake/Qwen3-VL-Embedding-8B"),
        embeddings=SimpleNamespace(
            embed_max_length=1024,
            adaptive_max_frames=8,
            adaptive_default_fps=1.0,
        ),
    )


def test_case_config_identity_uses_factory_dunder_name():
    identity = case_config_identity(VIDEO_CASE, _identity_settings())

    assert "provider=qwen_provider_video" in identity
    assert "_qwen_video_factory" not in identity


def test_qwen_provider_is_single_function():
    from modules.embeddings import cases as cm

    assert hasattr(cm, "qwen_provider")
    assert not hasattr(cm, "_qwen_video_factory")
    assert not hasattr(cm, "_qwen_text_factory")


def test_spec_has_no_instruction_or_requires_text_fields():
    """Regression guard: these fields were removed because they were
    dead/redundant. instruction lives inside payload builders; text
    usage is derived from text_builder is not None.
    """
    field_names = {f.name for f in EmbeddingCaseSpec.__dataclass_fields__.values()}
    assert "instruction" not in field_names
    assert "requires_text" not in field_names
