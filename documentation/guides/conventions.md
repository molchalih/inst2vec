# Conventions

How the codebase is organised and what every contribution is held to.

## Reuse and responsibilities

- **Reuse before you reimplement.** Look for an existing helper before writing
  one. `core/` holds the cross-cutting utilities (config, DB, ffmpeg,
  fingerprint, logging, model wrappers) every stage shares — call them; don't
  fork a second copy.
- **Mirror existing patterns.** Follow the shape of the surrounding code —
  naming, error handling, how a stage reads/writes the DB, how failures are
  recorded — unless there is a stated reason to diverge. Consistency is a feature.
- **Single source of truth.** A constant, schema, or piece of logic used in more
  than one place lives in exactly one module and is imported (e.g. the contract
  version `SCHEMA_VERSION`, the embedding-case registry). Never two copies that
  can drift.
- **One responsibility per unit.** A function does one thing; a module owns one
  concern; a stage owns one step. If you can't state a file's job without "and",
  split it. Every public symbol has one clear owner.
- **Extract on the second use** (or a non-obvious first use): pull the shared
  helper down into `core/` with a test rather than copy-pasting.
- **Simplicity, YAGNI.** Write the simplest thing that meets the requirement —
  no speculative abstractions, no unreferenced "we'll need it later" code.

The frontend enforces a stricter incarnation of these (layer rules, tokens as
the only source of visual constants) — see
[`../architecture/frontend.md`](../architecture/frontend.md).

## Stage architecture

- One pipeline stage per subpackage under `modules/`.
- Cross-cutting infrastructure (config, DB, ffmpeg, fingerprint, logging,
  vendored model wrappers) lives in `core/`, never in `modules/`.
- `scripts/` only orchestrate — no business logic; modules expose clean
  functions, so there are no half-script modules.
- Each stage has one public `run()` entry point with clear I/O.
- Stages are idempotent and safe to rerun: no duplicate rows, and failed
  states are tracked explicitly (pending / success / failed), not swallowed.
- Imports flow one way down the stack `scripts → services → modules → core`: a
  layer may import lower layers, never higher, and `core` depends on nothing
  internal. `uv run lint-imports` enforces this. Stages within `modules/` may
  depend on earlier stages (the pipeline DAG) and are deliberately unconstrained.

## Stage entry points

| Stage shape | Entry point |
|-------------|-------------|
| Single-phase | `run(settings, secrets)` |
| Multi-phase | one `run_<phase>(settings, secrets)` per phase |
| Case-aware | takes an extra `cases: tuple[str, ...]` argument |

Case-aware stages (clustering search/validation/assign, visualization) are
wired in `main.py` as `lambda s, x: stage.run(s, x, cases)`.

## Comments & docstrings

Two layers, each with one job: **docstrings** say what a unit is and how to use
it; **comments** say *why* — never *what*. The code already says what it does.

Keep a comment only when it captures what code can't:
- non-obvious rationale or a deliberate trade-off;
- an invariant or precondition the types don't enforce;
- a shape/enum annotation the type can't express (`Column(String)  # "user" | "cluster"`);
- a workaround and its reason; a `TODO(owner):` for a tracked follow-up;
- a tooling pragma (`# type:`, `# noqa`, `// eslint-disable`);
- a structural label in a long data definition whose grouping mirrors a contract
  (e.g. model column-group labels).

Remove:
- ASCII divider banners (`# ----`) and decorative section headers;
- comments that restate the next line of code;
- commented-out code (version control remembers it);
- attribution, changelog, or dated notes.

Docstrings stay condensed, no ceremony:
- module: one short paragraph — what it does and where it sits;
- public surface (stage entry points, exported functions/classes): one or two
  sentences plus only the non-obvious args/returns — no `Args:`/`Returns:`
  scaffolding that merely restates a typed signature;
- private helpers: a docstring only when the name and signature don't already
  make the purpose obvious; trivial one-liners get none.

Default to fewer comments. Prefer clear names and small functions over prose.

## Testing

- Tests live under `tests/` and mirror the module path of the code they cover.
- `tests/conftest.py` overrides `DATABASE_URL` to in-memory SQLite, so tests
  never touch a real database.
- Six gates must pass locally and in CI before a PR merges:

  ```bash
  uv run ruff check
  uv run ruff format --check
  uv run ty check
  uv run lint-imports          # import-layer contracts (see "Reuse and responsibilities")
  uv run python scripts/lint_docs.py
  uv run pytest
  ```

  CI runs the same four commands; PRs are not merged red.

## Database migrations

SQLite (dev/test) bootstraps instantly via `create_all`; no migration step is
needed. Postgres (production) is brought up and upgraded with Alembic across
three databases selected by `-n`, each reading its URL from the matching
environment variable:

```bash
uv run alembic -n main     upgrade head   # DATABASE_URL
uv run alembic -n identity upgrade head   # IDENTITY_DB_URL
uv run alembic -n serving  upgrade head   # SERVING_DATABASE_URL
```

After changing a SQLAlchemy model, autogenerate a revision for the affected
database, review it, and commit it alongside the model change:

```bash
uv run alembic -n main revision --autogenerate -m "<change>"
```
