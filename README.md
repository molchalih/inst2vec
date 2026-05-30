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
cp .env.example .env             # fill HIKER_API_KEY, HUGGINGFACE_TOKEN, …

uv sync --no-dev --group gpu     # GPU pipeline deps
uv run --env-file .env main.py   # run the pipeline
```

Alternative:

```bash
scripts/start.sh                # sync GPU deps + run the pipeline
```

## Pipeline

Stages run top-to-bottom from `main.py`. Each one is a single public `run()` per subpackage, idempotent, sealed by fingerprint.

| # | Stage | What it does |
|---|-------|--------------|
| 1 | **Ingest** | Seed usernames from CSV → HikerAPI profiles + Reels → download MP4 + thumbnails → extract MP3 (speech & MIR rates). |
| 2 | **Filter** | Flag ineligible users/clips, sample a per-user clip pool (`is_selected`). |
| 3 | **MIR** | MAEST + EffNet-Discogs ONNX inference → genre / mood / instrument descriptors (+ raw 2304-d MAEST vector). |
| 4 | **Speech** | Silero VAD → Whisper → Gemma translation → hallucination & quality guards. |
| 5 | **Captions** | Clean, language-detect, translate Reels captions. |
| 6 | **Clip embeddings** | `Qwen3-VL-Embedding-8B` over per-clip cases (`video`, `sandwich`, `spoken`, `textual`, `auditory`), run locally on the GPU host. |
| 7 | **User embeddings** | Mean-pool clip vectors per user, per case. |
| 8 | **Cluster search** | Grid sweep over UMAP + HDBSCAN hyperparameters. |
| 9 | **Cluster validation** | DBCV + silhouette, plateau detection. |
| 10 | **Clustering** | Two-pass UMAP (nD → 2D) + HDBSCAN; writes `UserCluster` + HDBSCAN soft-membership **centrality**. |
| 11 | **Visual labels** | Two-pass Qwen3-VL labelling of clips and clusters. |
| 12 | **Visualization** | Per-case 2D layouts, cluster ellipses, atlas rows for the frontend viewer. |

## Layout

| Path | Role |
|------|------|
| `core/` | Cross-cutting infra: config, **two-DB** schema (anon main + PII identity), ffmpeg, fingerprint, logging, vendored model wrappers. |
| `modules/<stage>/` | One subpackage per pipeline stage; `run(settings, secrets)` entry point. |
| `services/atlas_api/` | Read-only FastAPI service over the serving DB for the frontend viewer. |
| `scripts/` | Orchestration only — `start.sh`, `publish_visualization.py`, `analyze.py`, `cluster_analysis/`. |
| `docs/` | Quarto paper + paper-facing tables/plots in `docs/reporting/`. |
| `frontend/` | Static atlas viewer (Vite + Bun). See [`frontend/README.md`](frontend/README.md). |

## Configuration

| Source | Holds | Loader |
|--------|-------|--------|
| `config.toml` | Non-secret tunables — paths, per-stage hyperparameters, embedding cases, search/validation knobs. | `Settings` (Pydantic) |
| `.env` | Secrets — `HIKER_API_KEY`, `HUGGINGFACE_TOKEN`, `DATABASE_URL`, `IDENTITY_DB_URL`, `SERVING_DATABASE_URL`. | `Secrets` (Pydantic) |

Each stage takes a typed slice of both.

## License

Apache-2.0 — see [`LICENSE`](LICENSE).
