# Music clients relocation — design

**Date:** 2026-05-16
**Scope:** Relocate `modules/services.py` into `modules/music/` as `clients.py`, fold the in-file utility `chunked()` into the only class that uses it, and rename the matching test file. No behavior changes.

## 1. Motivation

`modules/services.py` is a 297-line file containing `SpotifyClient`, `ReccoBeatsClient`, the shared `TransientError` exception, and two module-private helpers. Today it lives at the shared-infrastructure tier of `modules/`, alongside cross-cutting concerns like `database/`, `config.py`, and `utils.py`.

The split between client classes and stage orchestration is justified — `SpotifyClient` (~95 lines) and `ReccoBeatsClient` (~150 lines) each carry their own OAuth, retry-with-backoff, and pacing logic. Folding them into `modules/music/features.py` would push that file past 500 lines and mix two distinct concerns: "how to talk to the APIs" versus "which rows to process and in what order."

The location is wrong. Only `modules/music/` ever imports `services.py`:

- `modules/music/features.py` imports `SpotifyClient`, `ReccoBeatsClient`, `TransientError`
- `modules/music/classify.py` imports `TransientError` (raised by the ACR fingerprint helper)
- `scripts/retry_failed_music_recognition.py` imports `TransientError`
- `tests/test_services.py` and `tests/test_music_features.py` import from `modules.services`

Nothing outside the music stage touches it. The "services" label and the shared-infrastructure placement misrepresent the actual coupling.

Additionally, `chunked()` is a 3-line generator used only twice — both call sites are inside `ReccoBeatsClient`. It does not warrant a module-level home; it belongs to the class that uses it.

## 2. Target structure

```
modules/music/
├── __init__.py
├── audio_sample.py
├── classify.py
├── clients.py        ← new (was modules/services.py)
├── features.py
└── state.py
```

`modules/services.py` is deleted. No new modules are introduced; no existing modules in `modules/music/` change responsibility.

## 3. `modules/music/clients.py` contents

The relocated file is the existing `services.py` with two structural changes:

1. **Class-private `_chunked` replaces module-level `chunked`.** Defined on `ReccoBeatsClient` as a `@staticmethod`:

   ```python
   @staticmethod
   def _chunked(lst: list, n: int):
       for i in range(0, len(lst), n):
           yield lst[i : i + n]
   ```

   Both call sites (`get_ids`, `get_features`) become `self._chunked(ids, self._batch)`. The free function is removed.

2. **Module docstring updated** to reflect the new scope: `"""Music-stage HTTP clients: Spotify and ReccoBeats."""`

Everything else is preserved verbatim:

- `class TransientError(Exception)` — same semantics
- `_is_transient_http(exc)` — module-private; shared by both clients, kept as a free function
- `_spotify_id_from_href(href)` — module-private; used only by `ReccoBeatsClient.get_ids`, kept as a free function (already underscore-prefixed, no benefit to converting to a static method)
- `SpotifyClient` — unchanged
- `ReccoBeatsClient` — unchanged except for the `_chunked` introduction

The two module-private helpers stay free functions because `_is_transient_http` is genuinely shared between the two classes, and `_spotify_id_from_href` is small, file-scoped, and already private. Moving them onto a class would create asymmetry without simplification.

## 4. Import updates

| File | Before | After |
|---|---|---|
| `modules/music/features.py` | `from modules.services import ReccoBeatsClient, SpotifyClient, TransientError` | `from modules.music.clients import ReccoBeatsClient, SpotifyClient, TransientError` |
| `modules/music/classify.py` | `from modules.services import TransientError` | `from modules.music.clients import TransientError` |
| `scripts/retry_failed_music_recognition.py` | `from modules.services import TransientError` | `from modules.music.clients import TransientError` |
| `tests/test_music_features.py` | `from modules.services import TransientError` | `from modules.music.clients import TransientError` |
| `tests/test_services.py` → `tests/test_music_clients.py` (renamed via `git mv`) | `from modules.services import ReccoBeatsClient, SpotifyClient, TransientError` | `from modules.music.clients import ReccoBeatsClient, SpotifyClient, TransientError` |

One docstring is updated: `modules/music/classify.py:38` currently reads `"Raises modules.services.TransientError after exhausted retries."` and becomes `"Raises modules.music.clients.TransientError after exhausted retries."`

After the refactor, `grep -rn "modules\.services\|modules.services"` must return no matches.

## 5. Mechanics

The relocation uses `git mv` to preserve file history:

```bash
git mv modules/services.py modules/music/clients.py
git mv tests/test_services.py tests/test_music_clients.py
```

Then in-place edits:

- `modules/music/clients.py` — update module docstring; remove module-level `chunked`; add `_chunked` static method to `ReccoBeatsClient`; update both call sites to `self._chunked(...)`
- `modules/music/features.py`, `modules/music/classify.py`, `scripts/retry_failed_music_recognition.py`, `tests/test_music_features.py`, `tests/test_music_clients.py` — rewrite import paths
- `modules/music/classify.py:38` — update docstring reference

## 6. Verification

All checks must pass after the refactor:

- `uv run pytest` — full suite (the existing tests in what is now `tests/test_music_clients.py` already cover both clients, OAuth refresh, retry-on-transient, and `TransientError` raising)
- `uv run ruff check`
- `uv run ruff format`
- `uv run ty check`
- `grep -rn "modules\.services\|modules.services" --include="*.py"` returns nothing

## 7. Out of scope

- No behavior changes to either client. Retry budgets, pacing intervals, OAuth token refresh, and transient-error classification are untouched.
- No refactoring of `modules/music/features.py` or `modules/music/classify.py` internals.
- No changes to `modules/music/__init__.py` exports. Callers that today import `TransientError` from `modules.services` will import it from `modules.music.clients` directly, not from the package root.
- No new tests. The existing test suite is sufficient for a pure relocation.
