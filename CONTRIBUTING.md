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

## Conventions

- One pipeline stage per subpackage under `modules/`. Cross-cutting infra goes in `core/`. Scripts in `scripts/` only orchestrate.
- Public entry per stage: one function, clear I/O. No half-script modules.
- Idempotent stages. Safe to rerun. No duplicate rows. Failed states tracked explicitly.
- Conventional commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`, `ci:`.
- Tests live in `tests/`, mirror the module path, override `DATABASE_URL` to in-memory SQLite via `tests/conftest.py`.

## Reporting bugs / requesting features

Use the issue templates in `.github/ISSUE_TEMPLATE/`.
