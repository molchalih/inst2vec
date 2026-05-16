"""Public-API smoke test for modules.database.

Enumerates every name documented in the package spec and asserts each one
is importable and is the expected kind of object. Catches accidentally-missed
re-exports in __init__.py during the database-package refactor.
"""

from __future__ import annotations

import inspect

import modules.database as db


CLASS_NAMES = [
    # ORM bases + models
    "Base",
    "User",
    "UserStats",
    "Clip",
    "Music",
    "ClipEmbedding",
    "UserEmbedding",
    "UserCluster",
    "StageState",
    "ClusterRun",
    # identity ORM
    "IdentityBase",
    "UserIdentity",
    "ClipIdentity",
]

CALLABLE_NAMES = [
    # engine / session
    "init_db",
    "get_engine",
    "get_session",
    "get_identity_engine",
    "get_identity_session",
    # identity CRUD
    "get_or_create_user_identity",
    "update_user_identity",
    "get_username",
    "get_api_pk",
    "get_profile_pic_url",
    "get_or_create_clip_identity",
    # predicates
    "clip_used_in_analysis",
    "clip_needs_speech_detection",
    "clip_has_detected_speech",
    "clip_needs_speech_translation",
    "has_raw_caption",
    "has_clean_caption",
    "needs_caption_cleaning",
    "needs_caption_language_detection",
    "needs_caption_translation",
]


def test_all_public_classes_exported():
    for name in CLASS_NAMES:
        obj = getattr(db, name, None)
        assert obj is not None, f"modules.database does not export class {name!r}"
        assert inspect.isclass(obj), f"modules.database.{name} should be a class"


def test_all_public_callables_exported():
    for name in CALLABLE_NAMES:
        obj = getattr(db, name, None)
        assert obj is not None, f"modules.database does not export {name!r}"
        assert callable(obj), f"modules.database.{name} should be callable"
