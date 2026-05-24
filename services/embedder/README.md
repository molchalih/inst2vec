# Embedder pod

Runs Qwen3-VL-Embedding-8B as a pull-worker on a GPU pod. The pod leases
clip-embedding jobs from the orchestrator coordinator, embeds them on its
own GPU, and reports results back. Scaling out is just launching more pods
pointed at the same coordinator host.

## Network volume layout

Mount a network volume at `/workspace`. It must contain:

- `models/Qwen3-VL-Embedding-8B/` — the model weights
- `videos/` — the video files the worker reads while embedding

## Building the image

```bash
docker build -t ghcr.io/<you>/inst2vec-embedder:latest \
    -f services/embedder/Dockerfile .
docker push  ghcr.io/<you>/inst2vec-embedder:latest
```

## Launching a pod

```bash
ORCHESTRATOR_HOST=ams-wsl01.core.240.agency:8765 \
EMBEDDER_TOKEN=<bearer> \
HUGGINGFACE_TOKEN=<...> \
docker compose -f services/embedder/compose.yml up
```

Honored env vars:

| Var                 | Required | Default                                    |
|---------------------|----------|--------------------------------------------|
| `ORCHESTRATOR_HOST` | yes      | —                                          |
| `EMBEDDER_TOKEN`    | yes      | —                                          |
| `MODEL_PATH`        | no       | `/workspace/models/Qwen3-VL-Embedding-8B`  |
| `VIDEO_ROOT`        | no       | `/workspace/videos`                        |

The pod exits 0 when the coordinator signals that the queue is drained.

## Scaling

Run more pods against the same `ORCHESTRATOR_HOST` (e.g. via Ansible). Each
pod is an independent pull-worker; the coordinator hands out jobs to
whichever pod asks next, so adding pods linearly increases throughput.
