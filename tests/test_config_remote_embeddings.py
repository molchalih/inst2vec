import os
from unittest.mock import patch

import pytest

from core.config import EmbeddingsSettings, StorageSettings


def test_embeddings_settings_provider_defaults_to_local():
    s = EmbeddingsSettings(
        exclude_disqualified_users=True,
        embed_max_length=32768,
        adaptive_max_frames=96,
        adaptive_default_fps=2.0,
    )
    assert s.provider == "local"
    assert s.inflight == 1
    assert s.request_timeout_s == 120
    assert s.max_retries == 3


def test_embeddings_settings_remote_requires_valid_provider_literal():
    with pytest.raises(ValueError):
        EmbeddingsSettings(
            exclude_disqualified_users=True,
            embed_max_length=32768,
            adaptive_max_frames=96,
            adaptive_default_fps=2.0,
            provider="cloud",  # invalid
        )


def test_storage_settings_fields():
    s = StorageSettings(
        backend="s3",
        bucket="my-bucket",
        prefix="videos/",
        signed_url_ttl_s=3600,
    )
    assert s.bucket == "my-bucket"
    assert s.signed_url_ttl_s == 3600


def test_secrets_includes_remote_embedder_and_storage():
    env = {
        "DATABASE_URL": "sqlite:///:memory:",
        "IDENTITY_DB_URL": "sqlite:///:memory:",
        "HIKER_API_KEY": "x",
        "HUGGINGFACE_TOKEN": "x",
        "EMBEDDER_REMOTE_URL": "https://pod.example/",
        "EMBEDDER_TOKEN": "tok",
        "OBJECT_STORE_ENDPOINT": "https://r2.example/",
        "OBJECT_STORE_ACCESS_KEY": "ak",
        "OBJECT_STORE_SECRET_KEY": "sk",
    }
    with patch.dict(os.environ, env, clear=True):
        from core.config import load_runtime_config

        _settings, secrets = load_runtime_config()
        assert secrets.embedder_remote_url == "https://pod.example/"
        assert secrets.embedder_token == "tok"
        assert secrets.object_store_endpoint == "https://r2.example/"
        assert secrets.object_store_access_key == "ak"
        assert secrets.object_store_secret_key == "sk"


def test_secrets_remote_fields_default_empty_when_not_set():
    """The remote-embedder secrets are optional — pipeline runs local by default."""
    env = {
        "DATABASE_URL": "sqlite:///:memory:",
        "IDENTITY_DB_URL": "sqlite:///:memory:",
        "HIKER_API_KEY": "x",
        "HUGGINGFACE_TOKEN": "x",
    }
    with patch.dict(os.environ, env, clear=True):
        from core.config import load_runtime_config

        _settings, secrets = load_runtime_config()
        assert secrets.embedder_remote_url == ""
        assert secrets.embedder_token == ""
