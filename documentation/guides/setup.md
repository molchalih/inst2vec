# Setup

Getting a development environment ready for the pipeline, the read API, and the
scripts.

## Install

Dependencies are managed with [uv](https://docs.astral.sh/uv/). The dependency
groups are opt-in, so you only pull what a given task needs:

```bash
uv sync --no-group gpu --group analysis
cp .env.example .env  # then fill in the keys you need
```

| Group | When to add it |
|-------|----------------|
| `--group analysis` | `scripts/` (pandas / seaborn / scikit-posthocs). Covered by `ty check`. |
| `--group serving` | The atlas read API (`services/atlas_api`). |
| `--group gpu` | Only on CUDA hosts that run local GPU embeddings. Omit it elsewhere so `flash-attn` doesn't try to build. |

CI syncs `--group analysis` and `--group serving`; `--group gpu` is reserved for
machines with CUDA.

## Configuration sources

Configuration is split in two, mirrored by two Pydantic models in
`core/config.py`:

| Source | Holds | Model |
|--------|-------|-------|
| `config.toml` | Non-secret tunables — paths, per-stage hyperparameters, embedding cases, search/validation knobs. | `Settings` |
| `.env` | Secrets — database URLs and API keys. | `Secrets` |

`load_runtime_config()` returns `(settings, secrets)`, and each pipeline stage
takes a typed slice of both rather than the whole object.

## Environment variables

`.env.example` is the authoritative list. Copy it to `.env` and fill in the
values you need:

| Variable | Purpose |
|----------|---------|
| `DATABASE_URL` | Main DB — anonymous data (users, clips, embeddings, clusters, labels, …). SQLite for dev, Postgres in production. |
| `IDENTITY_DB_URL` | Identity DB — PII only (Instagram API pk ↔ internal id maps). |
| `SERVING_DATABASE_URL` | Read-optimised serving store the offload script writes and the atlas API reads. Defaults to a local SQLite file. |
| `ATLAS_API_TOKEN` | Optional. When set, gates every atlas API endpoint behind a Bearer token; blank = open read access. |
| `ATLAS_API_CORS_ORIGIN` | Optional. Restricts browser access to a single origin; blank = no CORS header. |
| `HIKER_API_KEY` | Instagram profile + Reels ingest via HikerAPI. |
| `HUGGINGFACE_TOKEN` | Model downloads from Hugging Face. |

The three database URLs also drive the three Alembic environments — see
[conventions](conventions.md#database-migrations).

## Model files

Local GPU embedding runs read `Qwen3-VL-Embedding-8B` from the models
directory. It must be present at:

```
./models/Qwen3-VL-Embedding-8B
```
