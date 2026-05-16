from sqlalchemy import create_engine
from sqlalchemy.engine import Engine as _Engine
from sqlalchemy.orm import Session

from modules.database.models import (  # noqa: F401  -- re-exports for modules.database.*
    Base,
    Clip,
    ClipEmbedding,
    ClusterRun,
    Music,
    StageState,
    User,
    UserCluster,
    UserEmbedding,
    UserStats,
)
from modules.database.predicates import (  # noqa: F401  -- re-exports for modules.database.*
    clip_has_detected_speech,
    clip_needs_speech_detection,
    clip_needs_speech_translation,
    clip_used_in_analysis,
    has_clean_caption,
    has_raw_caption,
    needs_caption_cleaning,
    needs_caption_language_detection,
    needs_caption_translation,
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
