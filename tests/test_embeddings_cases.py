import types
from types import SimpleNamespace

from modules.embeddings.cases import (
    CASE_REGISTRY,
    VIDEO_CASE,
    EmbeddingCaseSpec,
    case_config_identity,
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
    # Recipe version + dependencies — any drift here invalidates sealed
    # sandwich embeddings, so they're pinned alongside the spec shape.
    assert spec.recipe_version == "sandwich_v3"
    assert "_audio_mir_row" in spec.dependency_columns
    assert "_music_row" not in spec.dependency_columns
    assert "is_speech_detected" in spec.dependency_columns


def test_spoken_case_shape():
    spec = CASE_REGISTRY["spoken"]
    assert spec.requires_video is False
    assert spec.apply_video_token_fallback is False
    assert spec.text_builder is not None
    assert spec.recipe_version == "spoken_v1"
    assert "_audio_mir_row" not in spec.dependency_columns
    assert "is_speech_detected" in spec.dependency_columns


def test_textual_case_shape():
    spec = CASE_REGISTRY["textual"]
    assert spec.requires_video is False
    assert spec.apply_video_token_fallback is False
    assert spec.text_builder is not None
    assert spec.recipe_version == "textual_v1"
    assert "_audio_mir_row" not in spec.dependency_columns
    assert "caption_clean" in spec.dependency_columns


def test_gemini_case_shape():
    from modules.embeddings.cases import GEMINI_CASE

    assert GEMINI_CASE.recipe_version == "gemini_v2"
    assert "is_speech_detected" in GEMINI_CASE.dependency_columns


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


def test_auditory_case_registered_with_expected_shape():
    from modules.embeddings.cases import AUDITORY_CASE, CASE_REGISTRY

    assert CASE_REGISTRY["auditory"] is AUDITORY_CASE
    assert AUDITORY_CASE.name == "auditory"
    assert AUDITORY_CASE.text_builder is None
    assert AUDITORY_CASE.requires_video is False
    assert AUDITORY_CASE.apply_video_token_fallback is False
    assert AUDITORY_CASE.dependency_columns == ("_audio_file_stat",)
    assert AUDITORY_CASE.recipe_version == "maest_v1"
    assert AUDITORY_CASE.requires == ()
    assert AUDITORY_CASE.served_remotely is False


def test_default_cases_includes_auditory():
    from modules.embeddings.cases import default_cases

    settings_stub = types.SimpleNamespace(
        embeddings=types.SimpleNamespace(gemini_enabled=False)
    )
    assert "auditory" in default_cases(settings_stub)


def test_case_config_identity_for_auditory_captures_onnx_backend(tmp_path):
    """The auditory identity must mark the onnx backend, the :7 output, the
    aggregation, the patch geometry, and keep the .pb sidecar sha256 anchor.
    The presence of ``backend=onnx`` is what forces the one-time re-extraction
    when migrating off the Essentia .pb path."""
    from modules.embeddings.cases import AUDITORY_CASE, case_config_identity

    model_dir = tmp_path / "mir_models"
    model_dir.mkdir()
    pb = model_dir / "discogs-maest-30s-pw-519l-1.pb"
    pb.write_bytes(b"x")
    (model_dir / "discogs-maest-30s-pw-519l-1.pb.sha256").write_text(
        '{"sha256": "abc123", "size": 1, "mtime_ns": 0}'
    )

    settings = types.SimpleNamespace(
        paths=types.SimpleNamespace(model_path="/fake/Qwen"),
        embeddings=types.SimpleNamespace(
            embed_max_length=1024,
            adaptive_max_frames=8,
            adaptive_default_fps=1.0,
        ),
        mir=types.SimpleNamespace(
            model_dir=str(model_dir),
            maest_checkpoint="discogs-maest-30s-pw-519l-1.pb",
            maest_onnx_checkpoint="discogs-maest-30s-pw-519l-1.onnx",
            maest_input="serving_default_melspectrogram",
            inference_sample_rate=16000,
            maest_patch_seconds=30.0,
        ),
    )
    identity = case_config_identity(AUDITORY_CASE, settings)

    assert "case=auditory" in identity
    assert "backend=onnx" in identity
    assert "maest_onnx_checkpoint=discogs-maest-30s-pw-519l-1.onnx" in identity
    assert "output_op=layer_4_tokens" in identity
    assert "pb_equiv=StatefulPartitionedCall:7" in identity
    assert "aggregation=concat_cls_dist_mean_v1" in identity
    assert "patch_reduction=mean" in identity
    assert "patch_frames=1876" in identity
    assert "patch_hop=1875" in identity
    assert "input_sample_rate=16000" in identity
    assert "checkpoint_sha256=abc123" in identity
    # The Essentia-only knobs are gone now that the backend is onnx.
    assert "input_op=" not in identity
    assert "min_samples=" not in identity
