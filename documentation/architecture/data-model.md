# Data model

The schema is split across two databases for PII isolation, plus a separate serving
DB the read API is built on. This document covers the two pipeline databases and the
embedding-case composite key. For the pipeline stages that read and write these tables,
see [pipeline](pipeline.md).

## Two databases (PII isolation)

The pipeline writes two databases, kept apart so personally identifiable information
never sits next to research data:

- **Main DB** (`DATABASE_URL`, default `sqlite:///data/inst2vec.db`) — research data
  only. Every row is keyed by **anonymous integer PKs**; there are no usernames or
  Instagram identifiers here.
- **Identity DB** (`IDENTITY_DB_URL`, default `sqlite:///data/identity_map.db`) — the
  PII lives here, as maps from Instagram API identifiers to the main DB's internal
  integer IDs.

`init_db(database_url, identity_db_url)` initialises both. `core/database/models.py`
owns the main-DB schema; `core/database/identity.py` owns the identity-DB CRUD.

## Main DB tables

The authoritative table set, matching the `__tablename__` declarations in
`core/database/models.py`:

| Table | What it holds |
|-------|---------------|
| `users` | One row per creator (anonymous integer PK) with eligibility / selection flags. |
| `user_stats` | Per-user play and posting-cadence statistics. |
| `clips` | One row per Reel: media URLs, caption fields, speech fields, and filter/eligibility flags. |
| `clip_filter_scratch` | Intermediate per-clip values used by the filter stage. |
| `audio_mir` | MAEST + EffNet MIR results per clip: genre / mood / instrument descriptors and flags. |
| `clip_embeddings` | One embedding vector per `(clip, embedding_case)`. |
| `user_embeddings` | One mean-pooled embedding vector per `(user, embedding_case)`. |
| `user_clusters` | Cluster assignment, 2D UMAP coordinates, and centrality per `(user, embedding_case)`. |
| `stage_state` | Fingerprint seals (data / config / dependency hashes) that make each stage idempotent. |
| `cluster_runs` | One row per searched UMAP + HDBSCAN parameter combination, with its scores. |
| `cluster_metrics` | Per-cluster quality metrics for the champion clustering, per case. |
| `visualizations` | Per-case visualization manifest row (label, size, source hash). |
| `visualization_users` | Per-case user atlas rows (2D position, cluster, centrality) for the viewer. |
| `visualization_clusters` | Per-case cluster ellipses (centre, radii, angle, label) for the viewer. |
| `clip_labels` | Per-clip, per-`label_case` Qwen-Instruct labels. |
| `cluster_labels` | Per-cluster, per-case Qwen-Instruct summary labels. |

## Identity DB tables

Two tables, owned by `core/database/identity.py`, mapping Instagram API identifiers to
the main DB's internal integer IDs:

| Table | What it holds |
|-------|---------------|
| `user_identities` | Instagram user pk ↔ internal `users.id`. |
| `clip_identities` | Instagram clip pk ↔ internal `clips.id`. |

## Embedding-case composite key

The `embedding_case` is a stable identity, not a transient run parameter, so it is part
of the **composite primary key** of both embedding tables:

- `clip_embeddings` is keyed by `(clip_id, embedding_case)`.
- `user_embeddings` is keyed by `(user_id, embedding_case)`.

Because the case is part of the key, every active case coexists in the same table — a
single clip carries one `clip_embeddings` row per case, and a single user one
`user_embeddings` row per case. The downstream `user_clusters`, `cluster_metrics`,
`visualization_*`, and `cluster_labels` tables carry `embedding_case` for the same
reason: the clustering, labelling, and visualization outputs are all scoped per case.
The active case set is resolved by `modules/embeddings/cases.py::default_cases()` (see
[embedding cases](overview.md#embedding-cases)).
