# Contributing

Thanks for considering a contribution. This is a research codebase first, so the bar is "clear, tested, and follows the stage architecture" — not framework polish.

## Setup

```bash
uv sync --no-group gpu --group analysis
cp .env.example .env  # then fill in the keys you need
```

The `analysis` group provides pandas/seaborn/scikit-posthocs for `scripts/`, which `ty check` covers. CI uses the same flags. Add `--group gpu` if you intend to run local GPU embeddings; omit it on machines without CUDA so `flash-attn` doesn't try to build.

## Workflow

1. Open an issue first for anything non-trivial. Drive-by PRs that change pipeline behavior are likely to bounce.
2. Branch from `main`. Keep PRs small and focused.
3. Run the full check suite locally before pushing:

   ```bash
   uv run ruff check
   uv run ruff format --check
   uv run ty check
   uv run pytest
   ```

4. CI runs the same four commands. PRs are not merged red.

## Database migrations

SQLite (dev/test) bootstraps instantly via `create_all`; no migration step is
needed. Postgres (production) is brought up and upgraded with Alembic, which
covers three databases selected by `-n`:

```bash
uv run alembic -n main     upgrade head   # DATABASE_URL
uv run alembic -n identity upgrade head   # IDENTITY_DB_URL
uv run alembic -n serving  upgrade head   # SERVING_DATABASE_URL
```

Each environment reads its URL from the matching env var (the same source as
`core.config`). After changing a SQLAlchemy model, autogenerate a revision for
the affected database, e.g. `uv run alembic -n main revision --autogenerate -m "<change>"`,
review it, and commit it alongside the model change.

## Serving data + read API

The frontend reads either the static JSON tree (default) or a read-only HTTP
API backed by a separate serving database. To populate and serve it:

```bash
# 1. Decompose the version-6 contract from the main DB into the serving DB.
uv run python scripts/offload_serving.py
# 2. Serve it (reads SERVING_DATABASE_URL; ATLAS_API_TOKEN / ATLAS_API_CORS_ORIGIN optional).
uv run python -m services.atlas_api
```

The API mirrors the static JSON paths 1:1 (`/manifest.json`,
`/runs/{run}/users.json`, etc.) and returns byte-identical payloads. Point the
frontend at it by setting `VITE_API_BASE_URL` (see `frontend/.env.example`);
leave it unset for the default static-JSON deploy.

## Conventions

- One pipeline stage per subpackage under `modules/`. Cross-cutting infra goes in `core/`. Scripts in `scripts/` only orchestrate.
- Public entry per stage: one function, clear I/O. No half-script modules.
- Idempotent stages. Safe to rerun. No duplicate rows. Failed states tracked explicitly.
- Conventional commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`, `ci:`.
- Tests live in `tests/`, mirror the module path, override `DATABASE_URL` to in-memory SQLite via `tests/conftest.py`.

## Reporting bugs / requesting features

Use the issue templates in `.github/ISSUE_TEMPLATE/`.
