# Embedder service

Stateless FastAPI service that exposes Qwen3-VL-Embedding-8B over HTTP.
Designed to run on a rented GPU pod (Runpod / Lambda / Vast); the local
`inst2vec` pipeline points at it via `EMBEDDER_REMOTE_URL`.

See the design spec at
`docs/superpowers/specs/2026-05-17-gpu-pod-embeddings-design.md` for
the full picture.

## Endpoints

| Method | Path       | Notes                                       |
|--------|------------|---------------------------------------------|
| GET    | /healthz   | liveness; reports `model_loaded`, gpu label |
| POST   | /embed     | single-clip embedding; bearer auth          |

Request shape:

```json
{
  "case": "video",
  "clip_id": 12345,
  "video_url": "https://r2.../12345.mp4?<signed>",
  "text": "song: ... · artist: ...",
  "fps": 1.0,
  "max_frames": 32
}
```

Response:

```json
{ "embedding": [...], "dim": 4096, "took_ms": 842 }
```

## One-time pod setup (Runpod example)

1. Create a Network Volume (~30 GB) → mount at `/workspace`.
2. Create a Pod: GPU = L40S or A100-40G, image = your pushed
   `inst2vec-embedder` tag, expose port 8000, attach the volume.
3. Set env: `EMBEDDER_TOKEN=<bearer>`, `HUGGINGFACE_TOKEN=<...>`.
4. First boot downloads weights to `/workspace/models` (one-time).
   Wait for `GET /healthz` to return `{"model_loaded": true}`.

## Building / pushing the image

```bash
docker build -t ghcr.io/<you>/inst2vec-embedder:latest \
    -f services/embedder/Dockerfile .
docker push  ghcr.io/<you>/inst2vec-embedder:latest
```

## Sanity check

```bash
EMBEDDER_REMOTE_URL=https://abc-8000.proxy.runpod.net \
EMBEDDER_TOKEN=... \
uv run python -m services.embedder.smoke
```

## Pointing the local pipeline at the pod

```env
# .env
EMBEDDER_REMOTE_URL=https://abc-8000.proxy.runpod.net
EMBEDDER_TOKEN=<same as on pod>
OBJECT_STORE_ENDPOINT=https://<account>.r2.cloudflarestorage.com
OBJECT_STORE_ACCESS_KEY=...
OBJECT_STORE_SECRET_KEY=...
```

```toml
# config.toml
[embeddings]
provider = "remote"
inflight = 8

[storage]
bucket = "inst2vec-videos"
prefix = "videos/"
signed_url_ttl_s = 3600
```

Then run as usual:

```bash
uv run python main.py
```

Only the clip-embeddings stage changes path; everything else still
runs locally.
