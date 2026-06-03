# Architecture overview

This is the entry point for the backend. It states what the system does, how the
tree is laid out, how the pipeline is structured, and what an "embedding case" is.
For the per-stage reference see [pipeline](pipeline.md); for the schema see the
[data model](data-model.md).

## What it does

`inst2vec` turns a CSV of Instagram Reels usernames into a flat **user → cluster**
table. It pulls each account's profile and Reels, downloads the videos, and embeds
their content across modalities — frames, audio, speech transcript, and caption —
with a vision-language embedding model. The per-clip vectors are mean-pooled into
one vector per user, per modality.

Those user vectors are then reduced with UMAP and clustered with HDBSCAN, grouping
creators by content similarity. The result is materialised as 2D layout coordinates,
cluster ellipses, and human-readable cluster labels, which the static **atlas viewer**
in `frontend/` renders as a pannable map where every dot is a creator, coloured by
cluster and sized by how central they are inside it.

## Top-level layout

| Path | Role |
|------|------|
| `core/` | Cross-cutting infrastructure shared by every stage: config, the two-database schema, ffmpeg, fingerprinting, logging/console, and vendored third-party model wrappers. |
| `modules/` | The pipeline stages — one subpackage per stage (`ingest`, `filter`, `mir`, `speech`, `captions`, `embeddings`, `clustering`, `labels`, `visualization`, plus `export` for the JSON writer). |
| `services/` | Standalone runtime services. `services/atlas_api/` is a read-only FastAPI service over the serving DB that backs the frontend viewer. |
| `scripts/` | Orchestration only — dataset analysis, cluster-parameter sweeps, and visualization JSON publishing. No business logic lives here. |
| `docs/` | Project logos, tracked under `docs/assets/`. The Quarto research paper and its paper-facing reporting tables/plots (`docs/reporting/`) are kept local-only. |
| `frontend/` | The static atlas viewer (TypeScript + Vite + jotai + Pixi). See [frontend](frontend.md). |

## Pipeline as a DAG

The pipeline is a sequence of idempotent stages run top-to-bottom from `main.py`.
Each stage is one public `run()` (or `run_<phase>()`) per subpackage with clear
inputs and outputs, and each is sealed by a **fingerprint** so reruns skip work
that is already complete and safely re-do work whose inputs changed. The stages form
a DAG by their upstream/downstream dependencies (declared as fingerprint dependency
edges); reordering them only triggers a noisy fingerprint reset on the next run, never
data loss.

See [pipeline](pipeline.md) for every stage's inputs, outputs, and entry point.

## Embedding cases

An **embedding case** is the stable identity of a complete embedding recipe — its
modality, provider, model family/version, prompt, and input-building logic together.
It is the single idempotence boundary the embeddings package recognises. Because the
case is part of identity, every embedding row carries `embedding_case` as part of its
composite key, so multiple cases coexist side by side in the same tables (see the
[embedding-case composite key](data-model.md#embedding-case-composite-key)).

The active set of cases for a run is resolved by
`modules/embeddings/cases.py::default_cases()`, which returns each case whose
`requires` gates all evaluate truthy against `settings.embeddings`. The registered
cases are `video`, `sandwich`, `spoken`, `textual`, and `auditory`:

- **`video`** — Qwen frames only.
- **`sandwich`** — Qwen frames combined with derived text (caption, transcript, and MIR descriptors).
- **`spoken`** — a text embedding over the speech transcript only.
- **`textual`** — a text embedding over the caption only.
- **`auditory`** — a raw-waveform MAEST acoustic embedding.
