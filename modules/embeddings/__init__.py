"""Embeddings package — modular embeddings pipeline.

Public API (pipeline entrypoints):

    run_clip(settings, secrets)
    run_users(settings, secrets)

Direct (non-pipeline) callers:

    embed_clip_embeddings(settings, secrets=None, cases=None)
    embed_user_embeddings(settings, cases=None)

Adding a new embedding case generally only requires:
  1. Add provider/factory (if not reusing one) in providers.py / cases.py
  2. Add text/payload builder helpers if needed
  3. Register a new EmbeddingCaseSpec under a NEW name in cases.py
     (do not reuse "video" / "sandwich" / "audio" — see spec)
"""

from core.config import Secrets, Settings
from modules.embeddings.cases import EmbeddingSecrets, default_cases
from modules.embeddings.runner import embed_clip_embeddings, run_clip
from modules.embeddings.users import embed_user_embeddings

__all__ = [
    "EmbeddingSecrets",
    "embed_clip_embeddings",
    "embed_user_embeddings",
    "run_clip",
    "run_users",
]


def run_users(settings: Settings, secrets: Secrets) -> None:
    """Aggregate clip embeddings into per-user vectors."""
    embed_user_embeddings(settings, cases=list(default_cases(settings)))
