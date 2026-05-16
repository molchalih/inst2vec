from sqlalchemy import create_engine
from sqlalchemy.engine import Engine as _Engine
from sqlalchemy.orm import Session
from sqlalchemy.sql import func

from modules.database.models import (
    Base,
    Clip,
)
from modules.database.models import (
    ClipEmbedding as ClipEmbedding,
)
from modules.database.models import (
    ClusterRun as ClusterRun,
)
from modules.database.models import (
    Music as Music,
)
from modules.database.models import (
    StageState as StageState,
)
from modules.database.models import (
    User as User,
)
from modules.database.models import (
    UserCluster as UserCluster,
)
from modules.database.models import (
    UserEmbedding as UserEmbedding,
)
from modules.database.models import (
    UserStats as UserStats,
)

_engine: _Engine | None = None


def get_engine() -> _Engine:
    assert _engine is not None, "Call init_db() before using the database"
    return _engine


def init_db(database_url: str, identity_db_url: str) -> None:
    global _engine
    from modules.identity import init_identity_db

    _engine = create_engine(database_url)
    Base.metadata.create_all(_engine)
    init_identity_db(identity_db_url)


def get_session() -> Session:
    return Session(get_engine())


def clip_used_in_analysis():
    """Canonical filter: clips that should drive downstream computation.

    Returns a tuple of clauses for `query.filter(*clip_used_in_analysis())`.
    """
    return (
        Clip.is_selected.is_(True),
        Clip.is_downloaded.is_(True),
    )


def clip_needs_speech_detection():
    """Clips eligible for Whisper transcription: selected, downloaded, unresolved."""
    return (
        *clip_used_in_analysis(),
        Clip.is_speech_detected.is_(None),
    )


def clip_has_detected_speech():
    """Clips that Whisper marked as containing meaningful speech."""
    return (
        *clip_used_in_analysis(),
        Clip.is_speech_detected.is_(True),
    )


def clip_needs_speech_translation():
    """Clips with detected non-English speech that still lack a translation."""
    return (
        *clip_has_detected_speech(),
        Clip.speech_transcription.is_not(None),
        Clip.speech_transcription != "",
        Clip.speech_language.is_not(None),
        Clip.speech_language != "",
        func.lower(Clip.speech_language).notlike("en%"),
        (Clip.speech_translation.is_(None)) | (Clip.speech_translation == ""),
    )


def has_raw_caption():
    """Clips that have any non-empty raw scraped caption."""
    return (
        Clip.caption_text.is_not(None),
        func.trim(Clip.caption_text) != "",
    )


def has_clean_caption():
    """Clips that already have a non-empty normalized caption."""
    return (
        Clip.caption_clean.is_not(None),
        func.trim(Clip.caption_clean) != "",
    )


def needs_caption_cleaning():
    """Selected, downloaded clips with raw text but no caption_clean yet."""
    return (
        *clip_used_in_analysis(),
        Clip.caption_text.is_not(None),
        func.trim(Clip.caption_text) != "",
        Clip.caption_clean.is_(None),
    )


def needs_caption_language_detection():
    """Selected, downloaded clips with caption_clean and no language tag."""
    return (
        *clip_used_in_analysis(),
        Clip.caption_clean.is_not(None),
        func.trim(Clip.caption_clean) != "",
        (Clip.caption_language.is_(None)) | (Clip.caption_language == ""),
    )


def needs_caption_translation():
    """Selected, downloaded clips with detected non-English clean caption and no translation."""
    return (
        *clip_used_in_analysis(),
        Clip.caption_clean.is_not(None),
        func.trim(Clip.caption_clean) != "",
        Clip.caption_language.is_not(None),
        Clip.caption_language != "",
        func.lower(Clip.caption_language).notlike("en%"),
        (Clip.caption_translation.is_(None)) | (Clip.caption_translation == ""),
    )


# ── Public-API surface re-exports (transitional; collapsed by package refactor) ──
from modules.identity import (  # noqa: E402,F401  -- re-exports for modules.database.*
    ClipIdentity,
    IdentityBase,
    UserIdentity,
    get_api_pk,
    get_identity_session,
    get_or_create_clip_identity,
    get_or_create_user_identity,
    get_profile_pic_url,
    get_username,
    update_user_identity,
)


def get_identity_engine():
    """Return the identity DB engine. Transitional shim — replaced by engine.py in Task 5."""
    from modules import identity as _identity_mod

    assert _identity_mod._engine is not None, "Call init_db() before using the identity DB"
    return _identity_mod._engine
