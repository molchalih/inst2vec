# Documentation

This directory is the single source of truth for how inst2vec is built. Each
durable fact lives here exactly once; the root `README.md` and `CONTRIBUTING.md`
are thin front doors that link inward.

## Map

| Path | What it covers |
|------|----------------|
| [`architecture/overview.md`](architecture/overview.md) | Mission, the two-database design, the embedding-case concept, the pipeline as a DAG. |
| [`architecture/pipeline.md`](architecture/pipeline.md) | The pipeline stages: per-stage inputs/outputs, entry points, idempotency via fingerprints. |
| [`architecture/data-model.md`](architecture/data-model.md) | Main + identity database schemas, the embedding-case composite key, serving decomposition. |
| [`architecture/frontend.md`](architecture/frontend.md) | The frontend atlas viewer: layered import contract, state, data planes, rendering. |
| [`guides/setup.md`](guides/setup.md) | Install, environment variables, `config.toml`, model paths. |
| [`guides/serving.md`](guides/serving.md) | Serving database, the atlas read API, and the frontend's two data modes. |
| [`guides/conventions.md`](guides/conventions.md) | Stage architecture rules, idempotency, Alembic migrations, testing. |

## What goes where

- **Durable engineering knowledge** — architecture, data model, pipeline, setup,
  serving, conventions — lives in this hub, in prose any contributor can read.
- **One source of truth.** A fact is documented in exactly one file here. When you
  change a stage, a schema, or a layer rule, update the matching doc in the same
  change. The CI docs-lint gate (`scripts/lint_docs.py`) checks contract versions
  and internal links so drift surfaces fast.
- Keep each file focused. If a file outgrows two screens of one topic, split it.
