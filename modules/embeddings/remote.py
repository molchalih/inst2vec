"""Placeholder for future remote embedding providers.

This module exists purely as an extension point. When a remote/API
embedding case ships, it gets a NEW ``embedding_case`` name (e.g.
``"remote_visual_v1"``); existing case names are never repointed at a
different provider, since different providers produce incompatible
vector spaces.
"""

from __future__ import annotations


class RemoteEmbeddingProvider:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def embed(self, payload: dict):
        raise NotImplementedError("Remote embedding provider is not implemented yet.")
