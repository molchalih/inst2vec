# Aesthetic Clustering & Visibility Analysis

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/colored/logo-light.svg">
    <img src="docs/assets/colored/logo.svg" alt="inst2vec" width="100%">
  </picture>
</p>

`inst2vec` turns a CSV of Instagram usernames into a flat **user → cluster** table and ships an interactive **atlas viewer** — a static, pannable 2D map where every dot is a creator, colored by cluster and sized by how central they are inside it.

**Concepts:** `Semantic map` · `Latent representations` · `Unsupervised clustering` · `Platform vernaculars`

<p align="center">
  <img src="https://img.shields.io/badge/uv-%23DE5FE9.svg?style=for-the-badge&logo=uv&logoColor=white" alt="uv">
  <img src="https://img.shields.io/badge/Bun-%23000000.svg?style=for-the-badge&logo=bun&logoColor=white" alt="Bun">
  <img src="https://img.shields.io/badge/vite-%23646CFF.svg?style=for-the-badge&logo=vite&logoColor=white" alt="Vite">
  <img src="https://img.shields.io/badge/Qwen-6950EF?style=for-the-badge&logo=qwen&logoColor=white" alt="Qwen">
  <a href="./LICENSE"><img src="https://img.shields.io/github/license/molchalih/inst2vec?style=for-the-badge" alt="Licence"></a>
</p>

## Quickstart

```bash
cp .env.example .env                          # fill HIKER_API_KEY, HUGGINGFACE_TOKEN, …

uv sync --no-dev --group gpu --group embedder # GPU pipeline + coordinator deps
uv run --env-file .env main.py                # run the pipeline
```

Alternative:

```bash
scripts/start.sh                # orchestrator (default)
scripts/start.sh --pod          # GPU pull-worker, needs ORCHESTRATOR_HOST + EMBEDDER_TOKEN
```

## Pipeline

Stages run top-to-bottom from `main.py`. Each one is a single public `run()` per subpackage, idempotent, sealed by fingerprint.

| # | Stage | What it does |
|---|-------|--------------|
| 1 | **Ingest** | Seed usernames from CSV → HikerAPI profiles + Reels → download MP4 + thumbnails → extract MP3 (speech & MIR rates). |
| 2 | **Filter** | Flag ineligible users/clips, sample a per-user clip pool (`is_selected`). |
| 3 | **Upload** | Push selected videos to S3-compatible storage for the remote embedder. No-op without `storage.bucket`. |
| 4 | **MIR** | MAEST + EffNet-Discogs ONNX inference → genre / mood / instrument descriptors (+ raw 2304-d MAEST vector). |
| 5 | **Speech** | Silero VAD → Whisper → Gemma translation → hallucination & quality guards. |
| 6 | **Captions** | Clean, language-detect, translate Reels captions. |
| 7 | **Clip embeddings** | `Qwen3-VL-Embedding-8B` over per-clip cases (`video`, `sandwich`, `audio`, optional `maest` / `gemini`). Local GPU + auto-scaling RunPod pull-workers via the HTTP coordinator. |
| 8 | **User embeddings** | Mean-pool clip vectors per user, per case. |
| 9 | **Cluster search** | Grid sweep over UMAP + HDBSCAN hyperparameters. |
| 10 | **Cluster validation** | DBCV + silhouette, plateau detection. |
| 11 | **Clustering** | Two-pass UMAP (nD → 2D) + HDBSCAN; writes `UserCluster` + HDBSCAN soft-membership **centrality**. |
| 12 | **Visualization** | Per-case 2D layouts, cluster ellipses, atlas rows for the frontend viewer. |

## Layout

| Path | Role |
|------|------|
| `core/` | Cross-cutting infra: config, **two-DB** schema (anon main + PII identity), storage, ffmpeg, fingerprint, logging, vendored model wrappers. |
| `modules/<stage>/` | One subpackage per pipeline stage; `run(settings, secrets)` entry point. |
| `services/embedder/` | Container image for the GPU pull-worker pod (Dockerfile + compose). |
| `scripts/` | Orchestration only — `start.sh`, `publish_visualization.py`, `analyze.py`, `cluster_analysis/`. |
| `docs/` | Quarto paper + paper-facing tables/plots in `docs/reporting/`. |
| `frontend/` | Static atlas viewer (Vite + Bun). See [`frontend/README.md`](frontend/README.md). |

## Configuration

| Source | Holds | Loader |
|--------|-------|--------|
| `config.toml` | Non-secret tunables — paths, per-stage hyperparameters, embedding cases, search/validation knobs. | `Settings` (Pydantic) |
| `.env` | Secrets — `HIKER_API_KEY`, `HUGGINGFACE_TOKEN`, `EMBEDDER_TOKEN`, `DATABASE_URL`, `IDENTITY_DB_URL`, `OBJECT_STORE_*`, `RUNPOD_*`. | `Secrets` (Pydantic) |

Each stage takes a typed slice of both.

## License

Apache-2.0 — see [`LICENSE`](LICENSE).
