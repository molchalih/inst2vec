import pytest

from modules.embeddings.remote import RemoteEmbeddingProvider


def test_remote_provider_embed_raises_not_implemented():
    provider = RemoteEmbeddingProvider()
    with pytest.raises(NotImplementedError):
        provider.embed({"text": "anything"})
