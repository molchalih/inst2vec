# Gemini multimodal embedding case

## Background

The embedding stage runs three local cases — `video`, `sandwich`, `audio`
— each backed by `LocalQwenProvider` over `Qwen3-VL-Embedding-8B`. The
case-spec architecture (`modules/embeddings/cases.py`) already supports
mixing provider backends: each `EmbeddingCaseSpec` binds to one
`provider_factory`, and `embedding_case` is the stable idempotence
boundary, so vectors from different providers coexist in
`clip_embeddings` and `user_embeddings` under different case names. A
placeholder `modules/embeddings/remote.py` exists for the first remote
provider.

Google's **Gemini Embedding 2** (`gemini-embedding-2-preview`) is a
natively multimodal embedding model. A single `embed_content` call can
interleave text, video, and audio parts and returns one aggregated
vector (default 3072-d, flexible 128–3072 via
`EmbedContentConfig(output_dimensionality=…)`). Per-request limits:
8192 text tokens, 120 s video, 80 s audio. The Google AI for Developers
surface (`generativelanguage.googleapis.com`) accepts inline bytes
(≤20 MB total request size) or Files API uploads (≤2 GB per file, 48 h
retention). It does **not** accept GCS / S3 URIs — that is a Vertex AI
feature, not used here.

## Goal

Add a fourth embedding case, `gemini_mm`, that for each clip embeds
**video bytes + audio bytes + text** in a single Gemini call, producing
one vector in a shared multimodal space. Existing cases, providers, the
fingerprint layer, and the runner's per-clip incremental dispatch are
unchanged.

A master switch `embeddings.gemini_enabled` gates the entire feature so
the codebase can ship the case while users who lack an API key (or
don't want to spend) keep running the three local cases unaffected.

## Non-goals

- Replacing or modifying any existing case.
- Vertex AI / GCS integration. The AI for Developers SDK is the only
  surface.
- Live API integration tests in the suite. Real provider I/O is
  exercised by an opt-in smoke script the developer runs manually.
- Batching multiple clips into one Gemini call. `batchEmbedContents`
  multimodal support is not yet GA at design time, and the runner's
  per-clip commit + fingerprint loop already amortizes work.
- Caching Files API URIs across runs. Files auto-expire at 48 h; per-call
  uploads keep the provider stateless and within free-tier limits.

## Pipeline shape

```
… → Download → AudioExtract (new) → Music → Speech → Captions
                                        → ClipEmbeddings (now 4 cases)
                                        → …
```

`AudioExtract` is a new stage owned by `modules/download.py` (helper
inside the same module per the design discussion — extraction is a
post-download local transcode, conceptually adjacent to fetching the
mp4). It is its own stage so it has its own `StageState` row and
fingerprint short-circuit.

## New module / file map

| File | Role |
|---|---|
| `modules/download.py` | Gains `extract_audio(video_path, audio_path, *, bitrate_kbps, sample_rate_hz) -> bool` and a stage entry `extract_audio_stage(settings) -> None` with its own fingerprint. Disabled-master-switch check at top of stage entry. |
| `modules/embeddings/gemini.py` | New. `GeminiMultimodalProvider` wraps `google.genai.Client`. Lazy-imports `google.genai` inside `__init__`. Owns Files API upload, length pre-checks, retry policy, single-vector return. |
| `modules/embeddings/remote.py` | Stub can be deleted, or kept as a one-line re-export of `GeminiMultimodalProvider`. Implementation note for the agent: delete it; nothing references it. |
| `modules/embeddings/cases.py` | Adds `GEMINI_MM_CASE`, `_gemini_factory`, `_gemini_payload`, the `case_config_identity` branch, and `default_cases(settings)` helper that switches `DEFAULT_CASES` on the master flag. |
| `modules/embeddings/text.py` | Adds `build_gemini_text(clip, _music_map) -> str | None`. |
| `modules/embeddings/state.py` | Extends `per_clip_source_hashes_and_aggregate` with a `"gemini_mm"` branch whose per-clip dependency row is `(clip_id, caption_translated_hash, transcript_translated_hash, video_stat, audio_stat)`. |
| `modules/embeddings/runner.py` | One-line tweak: when the provider raises a known `GeminiClipTooLongError`, log at warn level with the reason before falling through to the existing "skip and don't seal" path. No structural change. |
| `modules/config.py` | New fields on `Settings.paths` (`audio_dir`) and `Settings.embeddings` (master switch + Gemini + audio knobs). New `GeminiSecrets` model. `load_runtime_config()` returns `GeminiSecrets | None` based on the switch. |

## Case definition

In `modules/embeddings/cases.py`:

```python
GEMINI_MM_CASE = EmbeddingCaseSpec(
    name="gemini_mm",
    text_builder=build_gemini_text,
    requires_video=True,
    provider_factory=_gemini_factory,
    payload_builder=_gemini_payload,
    apply_video_token_fallback=False,
)
```

Registered in `CASE_REGISTRY`. `DEFAULT_CASES` becomes a function:

```python
def default_cases(settings) -> tuple[str, ...]:
    base = ("video", "sandwich", "audio")
    return base + ("gemini_mm",) if settings.embeddings.gemini_enabled else base
```

`embed_clip_embeddings(settings, cases=None)` calls `default_cases(settings)`
when `cases is None`. An explicit `cases=["gemini_mm"]` with
`gemini_enabled=False` raises `RuntimeError`.

`_gemini_payload`:

```python
def _gemini_payload(clip, text, video_path, fps, max_frames) -> dict:
    return {
        "video_path": video_path,
        "audio_path": _audio_path_for(clip.id),  # data/audio/{id}.mp3
        "text": text,
    }
```

`_gemini_factory` — note the **factory contract is widened** from
`Callable[[settings], Provider]` to
`Callable[[settings, secrets_bag], Provider]`. Existing factories
(`_local_qwen_video_factory`, `_local_qwen_text_factory`) gain a
second positional parameter they ignore (`_secrets = None` default).
The runner's `provider = spec.provider_factory(settings, secrets)`
call site changes accordingly, and `embed_clip_embeddings` takes a new
`secrets` argument plumbed from `main.py`.

```python
def _gemini_factory(settings, secrets) -> Provider:
    from modules.embeddings.gemini import GeminiMultimodalProvider
    return GeminiMultimodalProvider(
        api_key=secrets.gemini.api_key,
        model=settings.embeddings.gemini_model,
        output_dim=settings.embeddings.gemini_output_dim,
        max_video_seconds=settings.embeddings.gemini_max_video_seconds,
        max_audio_seconds=settings.embeddings.gemini_max_audio_seconds,
        request_timeout_s=settings.embeddings.gemini_request_timeout_s,
        max_retries=settings.embeddings.gemini_max_retries,
    )
```

`build_gemini_text` joins `clip.caption_translated` and
`clip.transcript_translated` with a `\n\n---\n\n` separator. Returns
`None` only if both fields are empty/None (clip non-embeddable; runner's
existing stale/fresh skip handling applies).

## Provider behavior

`GeminiMultimodalProvider.embed(payload) -> list[list[float]]`:

1. **Length gate.** ffprobe video duration and audio duration (cached
   per call). If either exceeds its configured cap, raise
   `GeminiClipTooLongError` — caught by the runner as a normal failure
   (stage stays unsealed, can be retried later if caps are raised).
2. **Files API upload.** `client.files.upload(file=video_path)` and
   `client.files.upload(file=audio_path)`. One inline retry on transient
   upload failure, then raise.
3. **Embed call.**

   ```python
   result = client.models.embed_content(
       model=self.model,
       contents=[
           text_string,
           types.Part.from_uri(file_uri=video_file.uri, mime_type="video/mp4"),
           types.Part.from_uri(file_uri=audio_file.uri, mime_type="audio/mpeg"),
       ],
       config=types.EmbedContentConfig(output_dimensionality=self.output_dim),
   )
   vector = result.embeddings[0].values
   ```
4. **Validate.** `len(vector) == self.output_dim` else raise
   `GeminiOutputDimMismatch` (config drift signal).
5. **Return** `[vector]` — single-element list matching the existing
   `provider.embed → out[0]` contract in `runner._embed_with_token_fallback`.

### Retry policy

`tenacity`-style exponential backoff on HTTP 429 / 5xx / timeout.
`gemini_max_retries` (default 5), backoff cap 60 s. Non-retriable 4xx
raise immediately. If `tenacity` is not already a transitive dep of
`google-genai`, vendor a ~30-line retry helper inside
`modules/embeddings/gemini.py` rather than adding a top-level dep.

### Observability

Per call, debug-level log: `bytes_uploaded`, `embed_seconds`, response
HTTP status. Enables post-batch cost sanity-checking.

## AudioExtract stage

Entry function in `modules/download.py`:

```python
def extract_audio_stage(settings) -> None:
    if not settings.embeddings.gemini_enabled:
        log("audio_extract", "disabled — skipping")
        return
    # ... fingerprint check, iterate downloaded clips, call extract_audio ...
```

Per-clip extraction:

```python
def extract_audio(video_path: str, audio_path: str,
                  *, bitrate_kbps: int, sample_rate_hz: int) -> bool:
    if (os.path.exists(audio_path)
            and os.path.getmtime(audio_path) >= os.path.getmtime(video_path)):
        return True  # cached and fresh
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vn",
           "-c:a", "libmp3lame", "-b:a", f"{bitrate_kbps}k",
           "-ar", str(sample_rate_hz), audio_path]
    return _run_ffmpeg(cmd, timeout=AUDIO_EXTRACT_TIMEOUT_S)
```

`_run_ffmpeg` already exists in `modules/speech/vad.py`. Promote it to
a shared helper (e.g. `modules/ffmpeg.py` exposing `run_ffmpeg`) and
have both `vad.py` and `download.py` import it. Avoids duplicating the
subprocess + timeout logic and keeps ffmpeg invocation conventions
consistent across stages.

Stage fingerprint (per the `modules.fingerprint` pattern):

- `STAGE = "audio_extract"`, no per-case dimension.
- `data = hash_rows((clip_id,) for clip_id in downloaded_ids)`.
- `config = hash_text(f"bitrate={audio_bitrate_kbps}|sr={audio_sample_rate_hz}|codec=libmp3lame")`.
- `dependency = hash_rows(video_file_stat(cid) for cid in downloaded_ids)` —
  reuse the same `video_file_stat` helper used by the `video` case.

No per-clip incremental gating inside the stage — ffmpeg is fast and a
full re-extract still completes in seconds on a thousand-clip dataset.
Filename-existence + mtime check inside `extract_audio` handles the
common rerun case anyway.

## Fingerprint integration for `gemini_mm`

### `case_config_identity` branch

Contributes:

```
provider=GeminiMultimodalProvider
model=<gemini_model>
output_dim=<gemini_output_dim>
audio_bitrate=<audio_bitrate_kbps>
audio_sample_rate=<audio_sample_rate_hz>
max_video_seconds=<gemini_max_video_seconds>
max_audio_seconds=<gemini_max_audio_seconds>
text_recipe=gemini_mm_v1
```

`TEXT_RECIPE_VERSIONS["gemini_mm"] = "gemini_mm_v1"`. Bump when
`build_gemini_text` changes semantics.

### Per-clip dependency row

In `state.per_clip_source_hashes_and_aggregate`, the `"gemini_mm"`
branch returns rows shaped:

```python
(
    clip.id,
    _hash_text_or_sentinel(clip.caption_translated),
    _hash_text_or_sentinel(clip.transcript_translated),
    _video_file_stat(clip.id),     # (size, mtime_ns) — already exists for "video"
    _audio_file_stat(clip.id),     # (size, mtime_ns) — new, mirrors video helper
)
```

This makes any of: caption edit, transcript edit, re-download, or
re-extract flip the per-clip hash → runner's existing diff loop
re-embeds that clip alone. Config drift on any of the
`case_config_identity` inputs flips `config_hash` → existing
`_wipe_case` re-embeds everyone.

## Configuration

`config.toml` additions:

```toml
[paths]
audio_dir = "data/audio"

[embeddings]
gemini_enabled           = false   # master switch
audio_bitrate_kbps       = 128
audio_sample_rate_hz     = 44100
gemini_model             = "gemini-embedding-2-preview"
gemini_output_dim        = 3072
gemini_max_video_seconds = 120
gemini_max_audio_seconds = 80
gemini_request_timeout_s = 60
gemini_max_retries       = 5
```

`.env.example` gains `GEMINI_API_KEY=`.

`pyproject.toml` declares `google-genai` under an extras group:

```toml
[project.optional-dependencies]
gemini = ["google-genai>=1.0"]
```

`.gitignore` gains `data/audio/`.

### Secrets

```python
class GeminiSecrets(BaseModel):
    api_key: str
```

`load_runtime_config()`:

- If `embeddings.gemini_enabled` is False: skip reading
  `GEMINI_API_KEY`; return `GeminiSecrets | None = None`.
- If True and the env var is missing: raise at startup with a clear
  message naming the env var and the config flag.

### Master-switch coverage

| Surface | When disabled |
|---|---|
| `extract_audio_stage` | Returns at entry; no ffmpeg, no `StageState` write. |
| `default_cases` | Omits `"gemini_mm"`. |
| Explicit `cases=["gemini_mm"]` | Runner raises `RuntimeError("gemini_mm requested but embeddings.gemini_enabled=false")`. |
| `GeminiSecrets` | Not loaded. |
| `google-genai` import | Never triggered (lazy inside provider `__init__`). Disabled installs do not need the package installed. |
| `main.py` | Stage call sites remain unconditional. Each stage short-circuits internally. No scattered `if gemini_enabled:`. |
| `CASE_REGISTRY` | Still contains `"gemini_mm"`. Registration is import-only and side-effect-free. |
| Existing `ClipEmbedding` rows for `embedding_case="gemini_mm"` | Untouched. Aggregation in `users.py` filters by case so they cannot leak into other cases' user vectors. Manual `DELETE` if cleanup desired — not the switch's job. |

## Failure handling

| Failure | Behavior |
|---|---|
| Video > `gemini_max_video_seconds` or audio > `gemini_max_audio_seconds` | `GeminiClipTooLongError` raised pre-upload. Runner counts as failure → stage unsealed → log at warn with reason. |
| Files API upload fails (5xx, timeout) | One inline retry, then raise. |
| `embed_content` 429 / 5xx / timeout | Exponential backoff up to `gemini_max_retries`. |
| `embed_content` non-429 4xx | Raise immediately (auth, malformed request — needs human attention). |
| Returned vector length ≠ `gemini_output_dim` | `GeminiOutputDimMismatch` — indicates config drift; raise. |
| Network unreachable mid-batch | Per-clip failure path is already correct: committed rows persist; stage stays unsealed; next run picks up the gap. |

## Testing

| Test file | Tests |
|---|---|
| `tests/test_audio_extract.py` | `test_extracts_mp3_from_mp4` (fixture mp4 → valid mp3 at configured bitrate/sr); `test_skip_when_audio_newer_than_video` (idempotence — second call no-op); `test_stage_fingerprint_seals` (stage seals on success; bumping `audio_bitrate_kbps` reseals on next run); `test_disabled_short_circuits` (`gemini_enabled=False` → stage returns without ffmpeg or `StageState` write). |
| `tests/test_embeddings_gemini_case.py` | `test_payload_shape` (`_gemini_payload` shape; `build_gemini_text` joins fields with documented separator); `test_provider_skips_oversize_video` (monkeypatched ffprobe → 150 s; provider raises without uploading); `test_runner_seals_when_all_clips_embed` (monkeypatched `embed` returns fixed 3072-d vector; stage seals); `test_runner_does_not_seal_on_failure` (one clip raises; stage unsealed; other clips' rows persist); `test_config_drift_wipes_case` (change `gemini_output_dim`, rerun → existing rows wiped, all re-embedded); `test_per_clip_diff_re_embeds_only_touched_clip` (edit one `caption_translated`, rerun → only that clip re-embedded); `test_disabled_excludes_from_defaults` (`default_cases` omits `gemini_mm` when off); `test_disabled_explicit_request_raises` (explicit cases=["gemini_mm"] with switch off raises). |
| `tests/test_config.py` | `test_secrets_optional_when_disabled` (load succeeds with `gemini_enabled=False` even if env var missing); `test_secrets_required_when_enabled` (load fails fast with clear message when enabled and env var missing). |
| `scripts/smoke_gemini_embed.py` | Opt-in. One clip → one Gemini call → prints first 10 dims and exits. Not in CI; manual smoke test for wiring. |

Fixture: `tests/fixtures/sample_5s.mp4` — a tiny ffmpeg-generated mp4
with a synthetic audio track. Used by audio-extract and provider
length-gate tests.

## Migration

None. All changes are additive:

- New config fields have defaults that make existing installs no-ops
  (`gemini_enabled = false`).
- New stage short-circuits when disabled.
- New case absent from `default_cases` when disabled.
- No schema migration for `ClipEmbedding` — `embedding_case` is already
  a free-text part of the composite key.

To enable on an existing install:

1. `pip install inst2vec[gemini]`
2. `GEMINI_API_KEY=...` in `.env`
3. Set `embeddings.gemini_enabled = true` in `config.toml`
4. `uv run python main.py` — AudioExtract runs once, then ClipEmbeddings
   embeds `gemini_mm` for all eligible clips.

## Out of scope (for follow-ups)

- A separate `gemini_mm` clustering run / visualization. The case
  produces a `(user_id, "gemini_mm")` row in `user_embeddings` the same
  way other cases do; cluster search and clustering pick it up
  automatically when configured to iterate that case. No code change
  required, just config.
- Vertex AI / GCS path. If a multi-region or enterprise tier becomes
  preferable, a second provider factory targeting Vertex would be added
  under a new case name (vectors from different providers cannot be
  pooled).
- Output-dimensionality experiments. The architecture supports
  switching `gemini_output_dim`; doing so wipes the case and rebuilds.
  Pick once and tune in a follow-up if recall benchmarks justify it.
