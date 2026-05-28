from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from core.config import Secrets, Settings

MINIMAL_TOML = b"""
[paths]
video_dir = "data/source/videos"
model_path = "./models/Qwen3-VL-Embedding-8B"
profile_pic_dir = "data/source/profile_pics"
thumbnail_dir = "data/source/thumbnails"
speech_audio_dir = "data/source/audio"
audio_dir = "data/audio"
data_csv_path = "data/data.csv"

[download]
max_attempts = 3
retry_delay = 15
retry_jitter = 5
concurrency = 5

[filter]
min_video_duration = 3
max_video_duration = 80
min_taken_at = 1640995200
creator_min_median_views = 10000
min_eligible_clips_per_user = 10
global_low_percentile = 5
global_high_percentile = 99
creator_low_z_threshold = -3.5
selection_pool_percent = 0.20
selected_clips_per_user = 10
selection_random_seed = 42

[speech]
whisper_model = "large-v3-turbo"
commit_every = 50
translate_model = "google/translategemma-4b-it"
translate_target_lang = "en"
translation_max_chars = 1000
translate_max_new_tokens = 200
logprob_threshold = -0.8
compression_threshold = 2.4
min_meaningful_chars = 8
dirty_min_chars = 5
dirty_min_letter_ratio = 0.3
vad_enabled = true
vad_sampling_rate = 16000
vad_threshold = 0.5
vad_min_speech_ms = 250
vad_min_silence_ms = 100
vad_speech_pad_ms = 150
vad_min_total_speech_s = 0.5
vad_ffmpeg_timeout_s = 60

[captions]
commit_every = 50
translate_model = "google/translategemma-4b-it"
translate_target_lang = "en"
translation_max_chars = 1000
translate_max_new_tokens = 200

[embeddings]
exclude_disqualified_users = true
embed_max_length = 32768
adaptive_max_frames = 96
adaptive_default_fps = 2.0

[search]

[validation]
plateau_drop_threshold = 0.05
max_noise_ratio = 0.3
min_clusters = 3
max_clusters = 20

[visualization]
export_dir = "data/visualization"
default_case = "video"

[labels]
prompt = "test prompt"
cluster_max_new_tokens = 1400
cluster_sample_token_budget = 7500
cluster_max_clips_per_cluster = 60
cluster_max_clips_per_user = 2
cluster_min_tags = 3
cluster_max_tags = 12
cluster_min_sentence_chars = 20
cluster_max_sentence_chars = 320
cluster_max_attempts = 3
cluster_prompt = "test cluster prompt"
"""

FAKE_SECRETS = {
    "DATABASE_URL": "sqlite:///:memory:",
    "IDENTITY_DB_URL": "sqlite:///:memory:",
    "HIKER_API_KEY": "hiker-key",
    "HUGGINGFACE_TOKEN": "hf-token",
}


def _load_with_fake_toml(tmp_path, env_overrides=None):
    from core import config as config_mod

    toml_file = tmp_path / "config.toml"
    toml_file.write_bytes(MINIMAL_TOML)
    env = {**FAKE_SECRETS, **(env_overrides or {})}
    with (
        patch.object(config_mod, "_CONFIG_PATH", toml_file),
        patch.dict(os.environ, env, clear=True),
    ):
        return config_mod.load_runtime_config()


def test_returns_two_objects(tmp_path):
    result = _load_with_fake_toml(tmp_path)
    assert isinstance(result, tuple)
    assert len(result) == 2
    settings, secrets = result
    assert isinstance(settings, Settings)
    assert isinstance(secrets, Secrets)


def test_settings_sections_present(tmp_path):
    settings, _ = _load_with_fake_toml(tmp_path)
    for section in (
        "paths",
        "download",
        "filter",
        "speech",
        "captions",
        "embeddings",
        "search",
        "validation",
    ):
        assert hasattr(settings, section), f"settings.{section} missing"


def test_settings_values_correct(tmp_path):
    settings, _ = _load_with_fake_toml(tmp_path)
    assert settings.download.max_attempts == 3
    assert settings.download.retry_delay == 15
    assert settings.download.retry_jitter == 5
    assert settings.download.concurrency == 5
    assert settings.filter.creator_low_z_threshold == -3.5
    assert settings.embeddings.exclude_disqualified_users is True


def test_secrets_from_env(tmp_path):
    _, secrets = _load_with_fake_toml(tmp_path)
    assert secrets.database_url == "sqlite:///:memory:"
    assert secrets.identity_db_url == "sqlite:///:memory:"
    assert secrets.hiker_api_key == "hiker-key"
    assert secrets.huggingface_token == "hf-token"


def test_paths_data_csv_path_present(tmp_path):
    settings, _ = _load_with_fake_toml(tmp_path)
    assert settings.paths.data_csv_path == "data/data.csv"


def test_env_overrides_runpod_and_storage_deployment_fields(tmp_path):
    # Deployment-specific values live in .env, not the tracked config.toml.
    settings, _ = _load_with_fake_toml(
        tmp_path,
        env_overrides={
            "RUNPOD_NETWORK_VOLUME_ID": "vol123",
            "RUNPOD_DATA_CENTER_ID": "EU-RO-1",
            "RUNPOD_GPU_TYPE_ID": "NVIDIA L4",
            "RUNPOD_GPU_MAX_PRICE_HR": "0.5",
            "RUNPOD_GPU_MIN_VRAM_GB": "32",
        },
    )
    assert settings.runpod.network_volume_id == "vol123"
    assert settings.runpod.data_center_id == "EU-RO-1"
    assert settings.runpod.gpu_type_id == "NVIDIA L4"
    assert settings.runpod.gpu_max_price_hr == 0.5
    assert settings.runpod.gpu_min_vram_gb == 32
    # bucket == the volume and region == its DC, derived so .env stays DRY.
    assert settings.storage.bucket == "vol123"
    assert settings.storage.region == "EU-RO-1"


def test_explicit_storage_env_wins_over_derivation(tmp_path):
    settings, _ = _load_with_fake_toml(
        tmp_path,
        env_overrides={
            "RUNPOD_NETWORK_VOLUME_ID": "vol123",
            "RUNPOD_DATA_CENTER_ID": "EU-RO-1",
            "STORAGE_BUCKET": "other-bucket",
            "STORAGE_REGION": "US-KS-2",
        },
    )
    assert settings.storage.bucket == "other-bucket"
    assert settings.storage.region == "US-KS-2"


def test_missing_secret_raises(tmp_path):
    from core import config as config_mod

    toml_file = tmp_path / "config.toml"
    toml_file.write_bytes(MINIMAL_TOML)
    incomplete = {k: v for k, v in FAKE_SECRETS.items() if k != "HIKER_API_KEY"}
    with (
        patch.object(config_mod, "_CONFIG_PATH", toml_file),
        patch.dict(os.environ, incomplete, clear=True),
        pytest.raises(KeyError),
    ):
        config_mod.load_runtime_config()


def test_download_settings_has_concurrency_and_jitter(tmp_path):
    settings, _ = _load_with_fake_toml(tmp_path)
    assert isinstance(settings.download.concurrency, int)
    assert isinstance(settings.download.retry_jitter, int)
    assert settings.download.concurrency >= 1


def test_settings_has_no_pipeline_field():
    from core.config import Settings

    assert "pipeline" not in Settings.model_fields


def test_speech_settings_has_vad_fields(tmp_path):
    settings, _ = _load_with_fake_toml(tmp_path)
    assert settings.paths.speech_audio_dir == "data/source/audio"
    assert settings.speech.vad_enabled is True
    assert settings.speech.vad_sampling_rate == 16000
    assert settings.speech.vad_threshold == 0.5
    assert settings.speech.vad_min_speech_ms == 250
    assert settings.speech.vad_min_silence_ms == 100
    assert settings.speech.vad_speech_pad_ms == 150
    assert settings.speech.vad_min_total_speech_s == 0.5
    assert settings.speech.vad_ffmpeg_timeout_s == 60


def test_secrets_optional_when_gemini_disabled(tmp_path):
    # Default MINIMAL_TOML has no gemini_enabled key — EmbeddingsSettings
    # defaults it to False. GEMINI_API_KEY is intentionally absent from
    # FAKE_SECRETS, so load must succeed and gemini_api_key must be None.
    settings, secrets = _load_with_fake_toml(tmp_path)
    assert settings.embeddings.gemini_enabled is False
    assert secrets.gemini_api_key is None


def test_secrets_required_when_gemini_enabled(tmp_path):
    # Flip gemini_enabled on via TOML override and confirm absence of the
    # env var produces a loud RuntimeError mentioning GEMINI_API_KEY.
    from core import config as config_mod

    toml_with_gemini = MINIMAL_TOML.replace(
        b"[embeddings]\nexclude_disqualified_users = true",
        b"[embeddings]\nexclude_disqualified_users = true\ngemini_enabled = true",
    )
    toml_file = tmp_path / "config.toml"
    toml_file.write_bytes(toml_with_gemini)
    with (
        patch.object(config_mod, "_CONFIG_PATH", toml_file),
        patch.dict(os.environ, FAKE_SECRETS, clear=True),
        pytest.raises(RuntimeError, match="GEMINI_API_KEY"),
    ):
        config_mod.load_runtime_config()


def test_secrets_present_when_gemini_enabled_and_key_set(tmp_path):
    from core import config as config_mod

    toml_with_gemini = MINIMAL_TOML.replace(
        b"[embeddings]\nexclude_disqualified_users = true",
        b"[embeddings]\nexclude_disqualified_users = true\ngemini_enabled = true",
    )
    toml_file = tmp_path / "config.toml"
    toml_file.write_bytes(toml_with_gemini)
    env = {**FAKE_SECRETS, "GEMINI_API_KEY": "gem-key"}
    with (
        patch.object(config_mod, "_CONFIG_PATH", toml_file),
        patch.dict(os.environ, env, clear=True),
    ):
        settings, secrets = config_mod.load_runtime_config()
    assert settings.embeddings.gemini_enabled is True
    assert secrets.gemini_api_key == "gem-key"


def test_load_runpod_config_needs_no_pipeline_secrets(tmp_path):
    # The GPU-listing helper must work from a RunPod-only env: no DB / Hiker /
    # HuggingFace vars present, and Gemini validation must not run.
    from core import config as config_mod

    toml_file = tmp_path / "config.toml"
    toml_file.write_bytes(MINIMAL_TOML)
    env = {
        "RUNPOD_API_KEY": "rp-key",
        "RUNPOD_NETWORK_VOLUME_ID": "vol123",
        "RUNPOD_DATA_CENTER_ID": "EU-RO-1",
    }
    with (
        patch.object(config_mod, "_CONFIG_PATH", toml_file),
        patch.dict(os.environ, env, clear=True),
    ):
        settings, runpod_api_key = config_mod.load_runpod_config()
    assert runpod_api_key == "rp-key"
    # Deployment overrides are applied, so the volume's DC is available.
    assert settings.runpod.network_volume_id == "vol123"
    assert settings.runpod.data_center_id == "EU-RO-1"


def test_settings_includes_mir_section(tmp_path):
    settings, _ = _load_with_fake_toml(tmp_path)
    assert hasattr(settings, "mir")
    mir = settings.mir
    assert mir.binary_threshold == 0.5
    assert mir.topk_genre == 10
    assert mir.topk_moodtheme == 10
    assert mir.topk_instrument == 10
    assert mir.inference_sample_rate == 16_000
    assert mir.download_concurrency == 4
    assert mir.commit_every == 25
    assert mir.prefetch_queue_size == 2
    assert mir.http_timeout == 30.0
    assert mir.model_dir == "models/mir"
    assert mir.maest_checkpoint == "discogs-maest-30s-pw-519l-1.pb"
    assert mir.maest_output == "StatefulPartitionedCall:0"
    assert mir.effnet_checkpoint == "discogs-effnet-bs64-1.pb"
    assert mir.effnet_embed_output == "PartitionedCall:1"


def test_mir_settings_rejects_invalid_threshold():
    from pydantic import ValidationError

    from core.config import MirSettings

    with pytest.raises(ValidationError):
        MirSettings(binary_threshold=0.0)
    with pytest.raises(ValidationError):
        MirSettings(binary_threshold=1.0)


def test_mir_settings_rejects_non_positive_topk():
    from pydantic import ValidationError

    from core.config import MirSettings

    with pytest.raises(ValidationError):
        MirSettings(topk_genre=0)


def test_audio_extraction_mir_defaults_match_inference_sample_rate():
    """MIR WAV defaults align with Essentia's MonoLoader(sampleRate=16000)."""
    from core.config import AudioExtractionSettings, MirSettings

    ae = AudioExtractionSettings()
    mir = MirSettings()
    assert ae.mir_sample_rate_hz == 16_000
    assert ae.mir_channels == 1
    assert ae.mir_sample_rate_hz == mir.inference_sample_rate
