# Serving data + read API

The frontend reads either a locally generated static JSON tree or a read-only
HTTP API backed by a separate serving database. This guide covers populating
the serving DB, running the API, and pointing the frontend at one or the other.

See [setup](setup.md) for the environment variables referenced below.

## Decompose to the serving DB

The offload script decomposes the version-7 frontend contract out of the main
DB and writes it into `SERVING_DATABASE_URL`:

```bash
uv run python scripts/offload_serving.py
```

## Run the read API

The atlas API serves the version-7 contract from the serving DB:

```bash
uv run python -m services.atlas_api
```

It reads `SERVING_DATABASE_URL`. `ATLAS_API_TOKEN` and `ATLAS_API_CORS_ORIGIN`
are optional (see [setup](setup.md#environment-variables)). The API mirrors the
static JSON paths 1:1 (`/manifest.json`, `/runs/{run}/users.json`, …) and
returns byte-identical payloads.

## Frontend data modes

The frontend chooses its data source from `VITE_API_BASE_URL`:

| `VITE_API_BASE_URL` | Source |
|---------------------|--------|
| Set | API mode — reads the atlas API. This is the default for the Pages deploy. |
| Unset | Static mode — reads a locally generated `public/data/` tree (generate it with `scripts/publish_visualization.py`). The tree is **not** committed and the deploy no longer ships one. |

See [`frontend/README.md`](../../frontend/README.md) for the frontend-side
configuration.
