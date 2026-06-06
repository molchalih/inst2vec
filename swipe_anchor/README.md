# swipe_anchor

A companion app that collects an **external, bias-clean similarity anchor** for the
creator-embedding pipeline via crowd odd-one-out judgments, and exports it as a
held-out evaluation set (triplets) plus an optional learned target geometry.

It is an **isolated** top-level component: its own database, its own SQLAlchemy
`Base`/engine, and no imports of the pipeline (`core/`, `modules/`, `services/`).
The only coupling is a read-only export job that opens the pipeline DB.

## Layout

```
swipe_anchor/
  core/      pure, dependency-light logic (triplet derivation, bias-guard selection)
  db/        app-store ORM models + engine/session helpers
  backend/   FastAPI assignment + respond service (build_app factory + __main__)
  export/    Stage-3 hand-off bundle writer (triplets.jsonl + embedding.npy + meta)
  tests/     pytest suite (runs under the repo's uv environment)
  frontend/  mobile-first 3-card swipe UI (bun + Vite + React 19 + jotai + Zod + Tailwind)
```

The `frontend/` mirrors the atlas `frontend/` layering contract (one-way layers
`app → features → ui → interaction → state → data → core`, `index.ts` public
surfaces, tokens as the single source of truth, pure + fully-tested `core/`) but
is its own app with its own product aesthetic. It is DOM + `<video>`, so there is
no Pixi `render/` layer.

## Status (foundation slice)

Built and tested:

- **Schema** — the full app store (creators, digests, comparisons, annotators,
  assignments, responses, triplets, gold, consensus, reliability events).
- **Triplet rule** — one odd-one-out answer → two ordinal triplets.
- **Bias guard** — standardized-medoid + farthest-point representative-clip
  selection (modality-neutral; never the visually-biased centrality).
- **Backend** — `/next-batch` (Phase-1 random draw over eligible, issues
  assignments) and `/respond` (idempotent, emits two triplets, counts judgments,
  retires at quorum). Deeplink access-code auth + per-user activity logging.
- **Anchor export** — writes the Stage-3 bundle from the app store.

Deferred (later phases): the information-scored balancer + Dawid-Skene consensus
+ reliability/gold QC, the live t-STE/CKL embedding worker, the mobile frontend,
the pipeline→app digest export against real data, and the public licensing path.

## Run

```bash
# Backend (serving group provides FastAPI/uvicorn). Allow the dev frontend origin:
SWIPE_ANCHOR_CORS=http://localhost:5174 uv run --group serving python -m swipe_anchor.backend

# Backend tests:
uv run --group serving pytest swipe_anchor/tests/

# Frontend (mobile-first; talks to the backend at http://localhost:8100 by default):
cd swipe_anchor/frontend
bun install
bun run dev        # http://localhost:5174/swipe-anchor/
bun run test       # vitest (100% coverage on core/)
bun run build      # production bundle
```

Backend config is via environment variables — see `backend/__main__.py`. Frontend
config (`VITE_API_BASE_URL`) — see `frontend/src/app/config.ts`.
The backend returns bare creator cards until the pipeline→app digest export lands,
and `next-batch` is empty until comparisons are seeded.

## Access codes, identity & logging

The app is **invite-only via deeplink**. A person opens `…/?code=THEIRCODE`; the
code is saved on their device and sent as the `X-Access-Code` header on every
request. The backend uses the code as the `annotator_id`, so all of a person's
choices + timings (`reaction_time_ms`, per-card `card_dwell_ms`) link to them.
With no code in the URL or storage the frontend opens nothing and the backend
returns 401.

```bash
# Issue codes (the note is INTERNAL — who the person is; never shown/exported):
uv run python -m swipe_anchor.backend.codes add 48DHF63 --note "dasha, gym friend"
uv run python -m swipe_anchor.backend.codes list
uv run python -m swipe_anchor.backend.codes disable 48DHF63
```

Allowlist semantics: while the `access_codes` table is **empty** the backend runs
open (any non-empty code works, for bootstrap); once any row exists, only listed +
active codes are admitted (403 otherwise). Every issue/answer is written to the
activity log (`SWIPE_ANCHOR_LOG`, default `data/swipe_anchor.log`) and to stdout.

> Deployment note: the browser sends only `X-Access-Code` (no bearer secret is
> embedded in the bundle, by design). Do **not** set `SWIPE_ANCHOR_TOKEN` for a
> browser-facing deploy — it would 401 the web app. The access code is the auth.
