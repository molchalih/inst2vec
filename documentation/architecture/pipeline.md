# Pipeline

Stages run top-to-bottom from `main.py`. Each stage is one public `run()` (or
`run_<phase>()`) per subpackage, idempotent and sealed by a fingerprint. The
fingerprint dependency edges define the DAG (see `main.py` for the edge comment);
the order below is the exact `_stages()` sequence in `main.py`. For the schema each
stage reads and writes, see the [data model](data-model.md).

## Database

Initialises both databases. `init_db(database_url, identity_db_url)` creates the
main-DB schema (anonymous integer PKs) and the identity-DB schema (PII maps). No data
is read or written beyond table creation. Entry point: `_init_db_stage` in `main.py`,
wrapping `core.database.init_db`.

## Importing

Loads the seed usernames from the input CSV into the `users` table. Idempotent on
re-seed: existing users are not duplicated. Entry point: `modules.ingest.run_seed`.

## Profile Parsing

Pulls each seeded user's Instagram profile and Reels via HikerAPI, writing `users`
fields and `clips` rows (with the PII mapping recorded in the identity DB). Runs over a
`ThreadPoolExecutor` and is idempotent via the fingerprint seal. Entry point:
`modules.ingest.run_profiles`.

## Processing Dataset

Flags ineligible users and clips against the filter predicates, computes per-user play
statistics, and randomly selects a per-user clip pool by setting `is_selected`. Reads
`clips`/`users`, writes eligibility/selection flags and `user_stats` /
`clip_filter_scratch`. Entry point: `modules.filter.run` (split internally into
`predicates` / `preprocess` / `select` / `state` / `stats`).

## Download

Downloads the selected clips' MP4 videos and thumbnails to local storage and marks
`clips.is_downloaded`. Idempotent: already-downloaded files are skipped. Entry point:
`modules.ingest.run_download`.

## Audio extraction

Extracts MP3 audio from downloaded videos at the speech-targeted sample rate, for the
speech stage to consume. Entry point: `modules.ingest.run_audio`.

## MIR audio extraction

Extracts MP3 audio at the MIR-targeted sample rate, separately from the speech rate, so
the MIR models receive the input they expect. Entry point:
`modules.ingest.run_audio_mir`.

## MIR inference

Runs Essentia MAEST + EffNet-Discogs ONNX inference over the MIR-extracted audio,
producing genre / mood / instrument descriptors (and the raw MAEST vector consumed by
the `auditory` embedding case). Reads the MIR audio, writes `audio_mir` rows. Entry
point: `modules.mir.run_mir`.

## Speech transcription

Gates each clip's audio through a Silero VAD pre-gate, transcribes detected speech with
Whisper, translates with Gemma, and post-cleans with hallucination and low-quality
guards. Reads the speech-rate audio, writes the `speech_*` fields on `clips`. Entry
point: `modules.speech.run`.

## Captions translation

Cleans each clip's caption, detects its language, and translates it. Reads/writes the
`caption_*` fields on `clips`. Entry point: `modules.captions.run`.

## Clip Embeddings

Runs Qwen3-VL-Embedding-8B (and the MAEST provider for the `auditory` case) over each
selected clip for every active embedding case, writing one row per `(clip, case)` to
`clip_embeddings`. The active case set comes from
`modules/embeddings/cases.py::default_cases()`. Entry point:
`modules.embeddings.run_clip`.

## User Embeddings

Mean-pools each user's clip vectors per case into one user vector per case, writing
`user_embeddings` rows keyed by `(user, case)`. Entry point:
`modules.embeddings.run_users`.

## Cluster Search

Grid-searches over UMAP + HDBSCAN hyperparameters per case, recording each parameter
combination and its summary statistics in `cluster_runs`. Reads `user_embeddings`.
Entry point: `modules.clustering.run_search` →
`modules.clustering.search.run_cluster_search`.

## Cluster Validation

Scores the searched runs with DBCV + silhouette and detects the parameter plateau,
updating the validation columns on `cluster_runs`. Entry point:
`modules.clustering.run_validation` → `modules.clustering.validation.validate_clustering`.

## Clustering

Re-fits the champion configuration: a two-pass UMAP (nD → 2D) followed by HDBSCAN,
writing `user_clusters` rows (cluster id, 2D coordinates, and HDBSCAN soft-membership
**centrality**) plus per-cluster quality rows in `cluster_metrics`. Entry point:
`modules.clustering.run_assign` → `modules.clustering.assign.assign_clusters`.

## Visual Labels

Two-pass Qwen3-VL-Instruct labelling. The per-clip pass summarises each selected clip
per modality into `clip_labels`; the cluster pass summarises each cluster from its
member clips into `cluster_labels`. Case-scoped fingerprinting lets each modality
regenerate independently; config drift wipes the affected rows. Depends on the selected
clips and on clustering. Entry point: `modules.labels.run`.

## Visualization

Fingerprint-gated write of the per-case 2D layouts, cluster ellipses, and user/cluster
rows the frontend atlas reads (`visualizations`, `visualization_users`,
`visualization_clusters`). Opt-out is per case via `EmbeddingCaseSpec.expose_to_viewer`.
JSON publishing to the frontend is handled out-of-pipeline by
`scripts/publish_visualization.py` via `modules/export/`. Entry point:
`modules.visualization.run` → `modules.visualization.pipeline.run_visualization`.
