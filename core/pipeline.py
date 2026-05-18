"""Pipeline stage identifiers used by the fingerprint layer."""

from __future__ import annotations

from enum import StrEnum


class Stage(StrEnum):
    AUDIO_EXTRACT = "audio_extract"
    FILTER = "filter"
    CAPTIONS = "captions"
    MUSIC_CLASSIFY = "music_classify"
    MUSIC_FEATURES = "music_features"
    SPEECH = "speech"
    CLIP_EMBEDDINGS = "clip_embeddings"
    USER_EMBEDDINGS = "user_embeddings"
    CLUSTER_SEARCH = "cluster_search"
    CLUSTER_VALIDATION = "cluster_validation"
    CLUSTER_ASSIGN = "cluster_assign"
