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

from modules.embeddings.cases import DEFAULT_CASES, EmbeddingSecrets
from modules.embeddings.runner import embed_clip_embeddings
from modules.embeddings.users import embed_user_embeddings

__all__ = [
    "DEFAULT_CASES",
    "EmbeddingSecrets",
    "embed_clip_embeddings",
    "embed_user_embeddings",
]
