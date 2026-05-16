# Database Package Refactor — Design

**Date:** 2026-05-16
**Status:** Approved (design phase). Implementation plan to follow.

## Goal

Replace the two flat files `modules/database.py` (422 lines, four mixed responsibilities) and `modules/identity.py` (140 lines) with a `modules/database/` package of four focused submodules plus `__init__.py`. Each file owns exactly one responsibility. Match the architectural standard of `modules/embeddings/`, `modules/speech/`, `modules/captions/`, and `modules/music/`.

## Non-goals

- No ORM schema change. Column names, types, table names, indexes, and unique constraints are byte-identical.
- No change to the two-database (main + identity) topology.
- No change to engine/session creation semantics (still SQLAlchemy `create_engine`, still `Session(engine)`).
- No migration to Alembic or any schema-management tool.
- No consolidation of `modules.identity` into "regular" main-DB models — the PII boundary stays visible at the file level.
- No new universal idempotence/fingerprinting module (tracked separately).
- No call-site migration beyond what the file deletions strictly require.

## Public API

The package preserves the current import surface so existing call sites compile unchanged:

```python
from modules.database import (
    # engine / session
    init_db, get_engine, get_session,
    get_identity_engine, get_identity_session,
    # main-DB ORM
    Base, User, UserStats, Clip, Music,
    ClipEmbedding, UserEmbedding, UserCluster, ClusterRun,
    # identity-DB ORM + CRUD
    IdentityBase, UserIdentity, ClipIdentity,
    get_or_create_user_identity, update_user_identity,
    get_username, get_api_pk, get_profile_pic_url,
    get_or_create_clip_identity,
    # query predicates
    clip_used_in_analysis, clip_needs_speech_detection,
    clip_has_detected_speech, clip_needs_speech_translation,
    has_raw_caption, has_clean_caption,
    needs_caption_cleaning, needs_caption_language_detection,
    needs_caption_translation,
)
```

`init_db(database_url: str, identity_db_url: str) -> None` keeps its signature.

## Package layout

```
modules/database/
    __init__.py     # re-export the full public API above
    engine.py       # both engines + sessions + init_db()
    models.py       # main-DB ORM: Base + 8 model classes
    identity.py     # identity-DB ORM (IdentityBase + 2 models) + CRUD helpers
    predicates.py   # 9 reusable SQL filter-clause helpers
```

Exactly one responsibility per file.

### `engine.py` — engine/session lifecycle for both DBs

Sole owner of engine globals; no one else in the package or codebase holds engine handles directly.

```python
_main_engine: Engine | None = None
_identity_engine: Engine | None = None

def get_engine() -> Engine: ...                  # main
def get_session() -> Session: ...                # main, non-context-manager (today's behavior)
def get_identity_engine() -> Engine: ...         # identity
@contextmanager
def get_identity_session() -> Iterator[Session]: ...  # identity, context-manager (today's behavior)

def init_db(database_url: str, identity_db_url: str) -> None:
    # 1) Build both engines (identity URL auto-wrapped with sqlite:/// when needed,
    #    matching current init_identity_db behavior).
    # 2) Base.metadata.create_all(_main_engine)
    # 3) IdentityBase.metadata.create_all(_identity_engine)
```

Imports `Base` from `.models` and `IdentityBase` from `.identity`. One-way dependencies; no cycles.

Replaces the current cross-module call at `modules/database.py:325` (`from modules.identity import init_identity_db` inside `init_db`) — that local-scope import disappears entirely.

### `models.py` — main-DB schema only

Verbatim move of the current `Base` class and these 8 ORM classes from `modules/database.py`:

- `User`, `UserStats`, `Clip`, `Music`
- `ClipEmbedding`, `UserEmbedding`
- `UserCluster`, `ClusterRun`

No engine handles, no predicates, no `init_db`. Pure declarative ORM.

### `identity.py` — identity domain (ORM + CRUD)

Owns everything PII-adjacent:

- `IdentityBase`, `UserIdentity`, `ClipIdentity` (verbatim from current `modules/identity.py`).
- All six CRUD helpers verbatim: `get_or_create_user_identity`, `update_user_identity`, `get_username`, `get_api_pk`, `get_profile_pic_url`, `get_or_create_clip_identity`.

**Change vs. today:** the private module-level `_engine` and `get_identity_session()` are removed from this file. CRUD helpers obtain a session by calling `get_identity_session()` from `engine.py`. The current `init_identity_db()` function is deleted (its work moves into `engine.init_db`).

This keeps `identity.py` purely about identity domain logic, with zero engine bookkeeping.

### `predicates.py` — reusable SQL clause helpers

Verbatim move of the 9 functions currently at the bottom of `modules/database.py`:

`clip_used_in_analysis`, `clip_needs_speech_detection`, `clip_has_detected_speech`, `clip_needs_speech_translation`, `has_raw_caption`, `has_clean_caption`, `needs_caption_cleaning`, `needs_caption_language_detection`, `needs_caption_translation`.

Imports `Clip` from `.models` and `func` from `sqlalchemy.sql`. No engine, no session, no side effects — purely tuple-returning clause builders consumed by `query.filter(*...)`.

### `__init__.py` — public API surface

Re-export the full list shown in the Public API section above. No business logic. Matches the role `__init__.py` plays in `modules/captions/`, `modules/speech/`, `modules/embeddings/`, `modules/music/`.

## Dependency graph

Top-level (module-load) imports only:

```
__init__.py  ──> engine.py, models.py, identity.py, predicates.py

engine.py    ──> models.Base, identity.IdentityBase   (needed by init_db's create_all)
models.py    ──> (no intra-package imports)
identity.py  ──> (no intra-package top-level imports)
predicates.py──> models.Clip
```

Function-scope (deferred) imports:

```
identity.py CRUD bodies ──> engine.get_identity_session
```

Rationale: `engine.py` needs `IdentityBase` at module top so `init_db` can call `IdentityBase.metadata.create_all` without a function-scope import. `identity.py` CRUD needs a session helper, but importing `engine` at the top would create a cycle (`engine → identity → engine`). The cycle is broken by deferring `from .engine import get_identity_session` to inside each CRUD function body — a one-line cost in a small number of helpers, paid once per call.

No cycles at import time. Each file is independently readable and testable.

## Migration mechanics

What actually changes outside the new package:

1. Delete `modules/database.py` and `modules/identity.py` (Git tracks the rename via the new package).
2. Create the 5 new files under `modules/database/`.
3. Rewrite identity import sites — `modules.identity` ceases to exist as a top-level module:
   - `modules/parse.py:8` → `from modules.database import (UserIdentity..., ...)` (preserve the imported names).
   - `modules/utils.py:7` → same swap of module path.
   - `modules/download.py:30` → same swap of module path.
   - Any test importing `from modules.identity` — same swap.
4. Drop the now-orphan local import in (removed) `modules/database.py:325`.
5. Everything else — every `from modules.database import …` line in modules, tests, generators, scripts (~128 sites) — is **unchanged** because `__init__.py` re-exports the same names.

Test-suite considerations:

- `tests/conftest.py` currently overrides `DATABASE_URL` to `sqlite:///:memory:` and calls `init_db`. Unchanged.
- `tests/test_database.py` includes patterns like `import modules.database as db_mod` — still works (the package is `modules.database`).
- `tests/test_download.py:13`'s `import modules.database as db_mod` followed by symbol access (e.g. `db_mod.Clip`) still works for the same reason.

## Testing strategy

- Existing tests must pass unchanged. Any test failure indicates a behavior-changing mistake in the refactor.
- Add a single smoke test that imports every name listed in the Public API section from `modules.database` and asserts each is the expected type (class vs callable). Catches an accidentally-missed re-export in one place.

## Trends this aligns with

- Mirrors the recent package shapes of `captions/`, `speech/`, `embeddings/`, `music/`: small focused files, `__init__.py` exports the public API.
- One responsibility per file: engine lifecycle / main schema / identity domain / query clauses.
- Eliminates the existing mid-function cross-module import inside `init_db`.
- Removes duplicated engine-global bookkeeping (today both `database.py` and `identity.py` keep their own private `_engine`).

## Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Missed re-export in `__init__.py` breaks an unrelated call site. | Public-API smoke test (above) enumerates every exported name. |
| Identity import-site rewrites miss a file. | Grep for `from modules.identity` and `import modules.identity` after the refactor; CI will catch any miss via test collection failure regardless. |
| Engine import cycle (engine → identity → engine). | `identity.py` imports `get_identity_session` lazily inside function bodies, not at module top level. |
| SQLite URL wrapping behavior diverges from current `init_identity_db`. | Port the exact `startswith("sqlite://")` / `sqlite:///{path}` branch unchanged into `engine.init_db`. |

## Out of scope (explicitly deferred)

- Renaming `get_session()` to `get_main_session()` for symmetry with `get_identity_session()`. Would touch ~30 call sites for cosmetic gain.
- Converting `get_session()` to a context manager for symmetry with `get_identity_session()`. Behavior change; out of scope.
- Splitting `models.py` further (e.g. one file per logical group like `clusters.py`). The 8 classes coexist readably in one file; revisit only if a single class grows large.
- Introducing Alembic-style migrations.
- A universal idempotence/fingerprinting module (separate design).
