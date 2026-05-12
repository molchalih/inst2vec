from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

MINIMAL_TOML = b"""
[pipeline]
batch_size = 9999
max_clips = 5

[paths]
video_dir = "data/source/videos"
plots_dir = "data/plots"
model_path = "./models/Qwen3-VL-Embedding-8B"
profile_pic_dir = "data/source/profile_pics"
thumbnail_dir = "data/source/thumbnails"
data_csv_path = "data/data.csv"

[parse]
fetch_retry_delays_sec = [0, 30, 60, 90]

[download]
max_attempts = 3
retry_delay = 2

[finalize]
target_clips_per_user = 4
require_min_text_clips = false
pass_a_recompute_from_scratch = true
global_min_plays = 0
global_min_plays_percentile = 5.0
creator_robust_z_threshold = -2.5
creator_min_clips = 4

[music]
audio_fingerprint_confidence = 0.8
commit_every = 50
http_timeout = 20.0
spotify_search_limit = 5
spotify_token_skew_seconds = 30
spotify_request_timeout = 8.0
reccobeats_batch_size = 20
reccobeats_delay_min = 2.0
reccobeats_delay_max = 3.0
manual_features_max_seconds = 20
manual_features_sample_rate = 44100
manual_features_max_mb = 5.0
manual_features_mp3_bitrate = "128k"

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

[overrides]
video = ""
sandwich = ""
audio = ""
"""

FAKE_SECRETS = {
    "DATABASE_URL": "sqlite:///:memory:",
    "IDENTITY_DB_URL": "sqlite:///:memory:",
    "HIKER_API_KEY": "hiker-key",
    "ARC_HOST": "arc-host",
    "ARC_ACCESS_KEY": "arc-access",
    "ARC_SECRET_KEY": "arc-secret",
    "SPOTIFY_CLIENT_ID": "sp-id",
    "SPOTIFY_CLIENT_SECRET": "sp-secret",
    "HUGGINGFACE_TOKEN": "hf-token",
}


def _load_with_fake_toml(tmp_path, env_overrides=None):
    from modules import config as config_mod

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
    assert isinstance(settings, SimpleNamespace)
    assert isinstance(secrets, SimpleNamespace)


def test_settings_sections_present(tmp_path):
    settings, _ = _load_with_fake_toml(tmp_path)
    for section in (
        "pipeline",
        "paths",
        "parse",
        "download",
        "finalize",
        "music",
        "speech",
        "captions",
        "embeddings",
        "search",
        "validation",
        "overrides",
    ):
        assert hasattr(settings, section), f"settings.{section} missing"


def test_settings_values_correct(tmp_path):
    settings, _ = _load_with_fake_toml(tmp_path)
    assert settings.pipeline.batch_size == 9999
    assert settings.pipeline.max_clips == 5
    assert settings.music.commit_every == 50
    assert settings.finalize.creator_robust_z_threshold == -2.5
    assert settings.parse.fetch_retry_delays_sec == [0, 30, 60, 90]
    assert settings.embeddings.exclude_disqualified_users is True


def test_secrets_from_env(tmp_path):
    _, secrets = _load_with_fake_toml(tmp_path)
    assert secrets.database_url == "sqlite:///:memory:"
    assert secrets.identity_db_url == "sqlite:///:memory:"
    assert secrets.hiker_api_key == "hiker-key"
    assert secrets.arc_host == "arc-host"
    assert secrets.spotify_client_id == "sp-id"
    assert secrets.huggingface_token == "hf-token"


def test_paths_data_csv_path_present(tmp_path):
    settings, _ = _load_with_fake_toml(tmp_path)
    assert settings.paths.data_csv_path == "data/data.csv"


def test_missing_secret_raises(tmp_path):
    from modules import config as config_mod

    toml_file = tmp_path / "config.toml"
    toml_file.write_bytes(MINIMAL_TOML)
    incomplete = {k: v for k, v in FAKE_SECRETS.items() if k != "HIKER_API_KEY"}
    with (
        patch.object(config_mod, "_CONFIG_PATH", toml_file),
        patch.dict(os.environ, incomplete, clear=True),
        pytest.raises(KeyError),
    ):
        config_mod.load_runtime_config()
