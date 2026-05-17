"""Provider-factory switch: local vs. remote via settings.embeddings.provider."""

from unittest.mock import MagicMock, patch

from modules.embeddings.cases import (
    AUDIO_CASE,
    SANDWICH_CASE,
    VIDEO_CASE,
)
from modules.embeddings.providers import RemoteQwenProvider


def _stub_settings(provider: str):
    s = MagicMock()
    s.embeddings.provider = provider
    s.embeddings.embed_max_length = 1024
    s.embeddings.adaptive_max_frames = 32
    s.embeddings.adaptive_default_fps = 1.0
    s.embeddings.request_timeout_s = 5
    s.embeddings.max_retries = 1
    s.paths.model_path = "./models/x"
    s.storage.bucket = "b"
    return s


def _stub_secrets():
    s = MagicMock()
    s.embedder_remote_url = "https://pod.example"
    s.embedder_token = "tok"
    s.object_store_endpoint = ""
    s.object_store_access_key = "ak"
    s.object_store_secret_key = "sk"
    return s


def test_video_factory_returns_remote_when_provider_remote():
    settings, secrets = _stub_settings("remote"), _stub_secrets()
    with patch("modules.embeddings.cases.get_object_store") as gos:
        gos.return_value = MagicMock()
        p = VIDEO_CASE.provider_factory(settings, secrets)
    assert isinstance(p, RemoteQwenProvider)


def test_sandwich_factory_returns_remote_when_provider_remote():
    settings, secrets = _stub_settings("remote"), _stub_secrets()
    with patch("modules.embeddings.cases.get_object_store") as gos:
        gos.return_value = MagicMock()
        p = SANDWICH_CASE.provider_factory(settings, secrets)
    assert isinstance(p, RemoteQwenProvider)


def test_audio_factory_returns_remote_when_provider_remote():
    settings, secrets = _stub_settings("remote"), _stub_secrets()
    with patch("modules.embeddings.cases.get_object_store") as gos:
        gos.return_value = MagicMock()
        p = AUDIO_CASE.provider_factory(settings, secrets)
    assert isinstance(p, RemoteQwenProvider)


def test_local_factory_does_not_construct_remote():
    """When provider='local', the factory must not even touch storage/secrets."""
    from modules.embeddings.providers import LocalQwenProvider

    settings, secrets = _stub_settings("local"), _stub_secrets()
    # We don't want to actually load the Qwen model in tests — patch the class.
    with patch.object(LocalQwenProvider, "__init__", return_value=None) as init:
        p = VIDEO_CASE.provider_factory(settings, secrets)
        init.assert_called_once()
    assert isinstance(p, LocalQwenProvider)
