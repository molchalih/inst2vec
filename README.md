<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/colored/logo-light.svg">
    <img src="docs/assets/colored/logo.svg" alt="inst2vec" width="320">
  </picture>
</p>

<!-- <p align="center">
  Parse Instagram Reels accounts. Embed their video, audio, and text. Cluster the users.
</p> -->

<p align="center">
  <a href="https://github.com/molchalih/inst2vec/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/molchalih/inst2vec/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://www.python.org/downloads/release/python-3120/"><img alt="Python" src="https://img.shields.io/badge/python-3.12+-blue.svg"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-green.svg"></a>
  <a href="https://github.com/astral-sh/uv"><img alt="uv" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json"></a>
  <a href="https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B"><img alt="Qwen3-VL" src="https://img.shields.io/badge/embeddings-Qwen3--VL--8B-purple"></a>
</p>

---

## Overview

`inst2vec` is a modular pipeline for clustering *Instagram creators* based on their *Reels* content-similarity. It parses profiles, embeds videos, music and speech; averages each user's clip vectors, and clusters the creators. While each stage provides it's own output, the final results is a flat user-to-cluster table.

## Pipeline

The 11 stages execute in order from `main.py`:

1. **Ingest** — seed accounts from CSV, pull profiles + Reels metadata via HikerAPI, download videos, extract MP3 audio.
2. **Filter** — flag ineligible users and clips, randomly select a per-user clip pool.
3. **Upload** — push selected videos to S3-compatible storage (no-op if `storage.bucket` is unset).
4. **Music** — fingerprint with ACRCloud, extract features via Spotify.
5. **Speech** — Silero VAD → Whisper transcription → translation → post-clean.
6. **Captions** — clean, detect language, translate clip captions.
7. **Clip embeddings** — Qwen3-VL-Embedding-8B across three cases per clip: `video`, `sandwich` (video + music text), `audio` (speech text). Runs locally or on a remote GPU pod.
8. **User embeddings** — average each user's clip vectors per case.
9. **Cluster search** — grid search over UMAP + HDBSCAN hyperparameters.
10. **Cluster validation** — DBCV + silhouette scoring, plateau detection.
11. **Clustering** — assign final labels and write `UserCluster` rows.

## Quickstart

```bash
# 1. Install
uv sync

# 2. Configure
cp .env.example .env
# fill in HIKER_API_KEY, ACR_*, SPOTIFY_*, HUGGINGFACE_TOKEN

# 3. Run
uv run python main.py
```

For local clip embeddings, Qwen3-VL weights are downloaded at `./models/Qwen3-VL-Embedding-8B`. For remote inference, point `EMBEDDER_REMOTE_URL` at a `services/embedder/` pod and the pipeline will offload there.

## Architecture

- `core/` — cross-cutting infra: config, database (two-DB design for PII isolation), storage, ffmpeg, fingerprint, console, third-party model wrappers.
- `modules/<stage>/` — one subpackage per pipeline stage with a single public entry function.
- `services/embedder/` — standalone GPU-pod service that serves Qwen3-VL embeddings remotely.
- `scripts/` — orchestration only (migrations, retry helpers, dataset analysis).
- `docs/` — the Quarto paper and reporting tables/plots.

Configuration splits into `Settings` (tunables in `config.toml`) and `Secrets` (env / `.env`). Each stage takes its own typed slice of both.

## License

Apache-2.0 — see [`LICENSE`](LICENSE).
