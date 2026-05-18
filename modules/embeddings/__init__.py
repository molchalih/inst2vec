"""Embeddings package — modular embeddings pipeline.

Public API:

    embed_clip_embeddings(settings, secrets=None, cases=None)
    embed_user_embeddings(settings, cases=None)

Adding a new embedding case generally only requires:
  1. Add provider/factory (if not reusing one) in providers.py / remote.py
  2. Add text/payload builder helpers if needed
  3. Register a new EmbeddingCaseSpec under a NEW name in cases.py
     (do not reuse "video" / "sandwich" / "audio" — see spec)
"""

from core.config import Secrets, Settings
from modules.embeddings.cases import EmbeddingSecrets, default_cases
from modules.embeddings.runner import embed_clip_embeddings
from modules.embeddings.users import embed_user_embeddings

__all__ = [
    "EmbeddingSecrets",
    "embed_clip_embeddings",
    "embed_user_embeddings",
    "run_clip",
    "run_users",
]


def run_clip(settings: Settings, secrets: Secrets) -> None:
    """Clip-level embeddings across all configured cases."""
    embed_clip_embeddings(
        settings,
        EmbeddingSecrets(
            gemini_api_key=secrets.gemini_api_key,
            embedder_remote_url=secrets.embedder_remote_url,
            embedder_token=secrets.embedder_token,
            object_store_endpoint=secrets.object_store_endpoint,
            object_store_access_key=secrets.object_store_access_key,
            object_store_secret_key=secrets.object_store_secret_key,
        ),
    )


def run_users(settings: Settings, secrets: Secrets) -> None:
    """Aggregate clip embeddings into per-user vectors."""
    embed_user_embeddings(settings, cases=list(default_cases(settings)))
