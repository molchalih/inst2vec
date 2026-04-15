# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the pipeline

```bash
# Run the full pipeline
python main.py

# Activate the virtual environment first if needed
source .venv/bin/activate
```

Copy `.env.example` to `.env` and fill in the required credentials before running.

## Environment variables

All configuration lives in `.env`. Key variables:

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | SQLAlchemy DB URL (default: `sqlite:///data/inst2vec.db`) |
| `HIKER_API_KEY` | HikerAPI token for Instagram data fetching |
| `ARC_ACCESS_KEY` / `ARC_SECRET_KEY` / `ARC_HOST` | ACRCloud music fingerprinting |
| `SPOTIFY_CLIENT_ID` / `SPOTIFY_CLIENT_SECRET` | Spotify track ID lookup |
| `HUGGINGFACE_TOKEN` | HF model downloads |
| `BATCH_SIZE` | Users to process per run |
| `MAX_CLIPS` | Max clips to fetch/process per user |
| `WHISPER_MODEL` | Whisper model variant (default: `large-v3-turbo`) |

## Architecture

`main.py` runs a linear pipeline of sequential phases, each idempotent (skips already-processed rows):

```
init_db()                  → create/migrate SQLite schema
load_usernames_from_csv()  → seed Users from data/data.csv (one Instagram URL per row)
fetch_profiles()           → HikerAPI: fill User metadata + create Clip rows
download_files()           → HTTP: save profile_pics, thumbnails, videos to data/source/

classify_music()           → ACRCloud fingerprint videos → link Clip → Music rows
extract_music_features()   → Spotify IDs → ReccoBeats catalog features (upload fallback via ffmpeg)

classify_speech()          → Whisper transcription → has_speech + quality gates
translate_speech()         → TranslateGemma: non-English transcriptions → English

detect_caption_language()  → Lingua: detect caption_language for each Clip
translate_captions()       → TranslateGemma: non-English captions → English

# embed_clips()            → (disabled) Qwen3-VL embeddings → embeddings table
```

## Database schema

Three core tables in `modules/database.py`:
- **`users`** — Instagram user profiles (seeded from `data/data.csv`)
- **`clips`** — Instagram Reels, linked to users; accumulates music/speech/caption fields across pipeline phases
- **`music`** — deduplicated music tracks with Spotify/ReccoBeats IDs and audio features
- **`downloads`** — tracks per-file download success/failure to avoid re-attempts

`_migrate_clips_table()` applies additive column migrations for pre-existing SQLite databases.

## Module structure

- `modules/database.py` — SQLAlchemy models, `init_db()`, `get_session()`, CSV loader
- `modules/parse.py` — HikerAPI calls to fill User + Clip rows
- `modules/download.py` — HTTP file downloads; sets `clip_disqualified` on video failure
- `modules/music.py` — ACRCloud fingerprinting + Spotify/ReccoBeats feature extraction
- `modules/speech.py` — Whisper transcription with hallucination filtering (logprob + compression ratio gates)
- `modules/captions.py` — Lingua language detection + TranslateGemma translation
- `modules/services.py` — `log()`, `SpotifyClient`, `ReccoBeatsClient`
- `modules/external/gemma_translate.py` — HF pipeline wrapper for `google/translategemma-4b-it`
- `modules/external/qwen3_vl_embedding.py` — Qwen3-VL video+text embedder (used by disabled `embed_clips()`)
- `modules/embeddings.py` — disabled embedding phase; writes to `embeddings` table

## Data flow and file paths

- Input: `data/data.csv` — one Instagram profile URL per line
- Downloaded files: `data/source/profile_pics/`, `data/source/thumbnails/`, `data/source/videos/`
- Database: `data/inst2vec.db` (default)

## Scripts

- `scripts/cleaner.sh` — deduplicates lines in `data/data.csv` (order-preserving)
