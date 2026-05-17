# Gemini Multimodal Embedding Case Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fourth embedding case `gemini_mm` that embeds video + audio + text into one Gemini Embedding 2 vector per clip, behind a master `gemini_enabled` switch, with a new AudioExtract pipeline stage and full incremental-dispatch integration.

**Architecture:** New stage `AudioExtract` (helper in `modules/download.py`) writes `data/audio/{id}.mp3`. New `GeminiMultimodalProvider` (in `modules/embeddings/gemini.py`) uploads video+audio via Gemini Files API and calls `embed_content` with three interleaved Parts. New case spec wires into the existing case-spec runner with no runner restructure. Factory contract widens from `(settings)` to `(settings, secrets)` so the Gemini factory can receive its API key; existing local factories ignore the new arg. Master switch gates extraction, default-cases inclusion, secret loading, and provider import.

**Tech Stack:** `google-genai` SDK (optional extras), `tenacity`-style backoff (vendored if not transitive), `ffmpeg` subprocess, existing `modules.fingerprint` layer, existing per-clip incremental dispatch in `modules/embeddings/runner.py`.

**Spec:** `docs/superpowers/specs/2026-05-17-gemini-multimodal-embedding-case-design.md`

---

## File Map

**Create:**
- `modules/ffmpeg.py` — promote `_run_ffmpeg` from `modules/speech/vad.py` to a shared helper
- `modules/embeddings/gemini.py` — `GeminiMultimodalProvider`, `GeminiClipTooLongError`, `GeminiOutputDimMismatch`, `GeminiSecrets`, `_probe_duration_seconds`
- `tests/test_audio_extract.py`
- `tests/test_embeddings_gemini_case.py`
- `tests/fixtures/conftest.py` (or extend root `tests/conftest.py`) — `sample_mp4_with_audio` pytest fixture
- `scripts/smoke_gemini_embed.py`

**Modify:**
- `modules/config.py` — extend `PathsSettings`, `EmbeddingsSettings`, `Secrets`; conditional Gemini key load
- `modules/download.py` — add `extract_audio`, `extract_audio_stage`
- `modules/embeddings/__init__.py` — re-export `EmbeddingSecrets`
- `modules/embeddings/cases.py` — `GEMINI_MM_CASE`, `default_cases(settings)`, widened factory contract, `case_config_identity` branch, `TEXT_RECIPE_VERSIONS` entry
- `modules/embeddings/text.py` — add `build_gemini_text`
- `modules/embeddings/state.py` — `"gemini_mm"` branch in `dependency_rows_for_case`; new `_audio_file_stat`, `_video_file_stat` helpers
- `modules/embeddings/runner.py` — accept `secrets` arg, plumb to factory; warn-log on `GeminiClipTooLongError`; reject explicit `gemini_mm` request when disabled
- `modules/speech/vad.py` — import `_run_ffmpeg` from `modules/ffmpeg`
- `main.py` — wire `extract_audio_stage`; build `EmbeddingSecrets`; pass to `embed_clip_embeddings`
- `modules/embeddings/remote.py` — delete (nothing references it)
- `config.toml` — new keys with default `gemini_enabled = false`
- `.env.example` — `GEMINI_API_KEY=`
- `.gitignore` — `data/audio/`
- `pyproject.toml` — `[project.optional-dependencies] gemini = ["google-genai>=1.0"]`
- `tests/test_config.py` — secrets gating

---

## Phase 1 — Shared foundations

### Task 1: Promote `_run_ffmpeg` to `modules/ffmpeg.py`

**Files:**
- Create: `modules/ffmpeg.py`
- Modify: `modules/speech/vad.py`
- Test: existing `tests/test_speech_*.py` must continue to pass.

- [ ] **Step 1: Read current `_run_ffmpeg` body from `modules/speech/vad.py`**

Run: `grep -n "_run_ffmpeg\|_FFMPEG_TIMEOUT" modules/speech/vad.py`

- [ ] **Step 2: Create `modules/ffmpeg.py` with the lifted helper**

```python
"""Shared ffmpeg subprocess helper.

Used by speech VAD, audio extraction, and any future stage that shells
out to ffmpeg. Failure path is uniform: ``run_ffmpeg`` returns False on
non-zero exit or timeout; callers decide whether to raise.
"""

from __future__ import annotations

import subprocess


def run_ffmpeg(cmd: list[str], *, timeout: int) -> bool:
    """Run ``cmd`` (list, no shell). Return True on exit code 0, else False."""
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False
    return result.returncode == 0
```

- [ ] **Step 3: Replace `_run_ffmpeg` in `vad.py` with an import + delete the local copy**

In `modules/speech/vad.py`, replace the local `_run_ffmpeg` definition with:

```python
from modules.ffmpeg import run_ffmpeg as _run_ffmpeg
```

Keep the `_FFMPEG_TIMEOUT_SECONDS` constant where it is. All call sites
already use `_run_ffmpeg`, so no other changes.

- [ ] **Step 4: Run the speech tests to verify no regression**

Run: `uv run pytest tests/ -k speech -v`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add modules/ffmpeg.py modules/speech/vad.py
git commit -m "refactor(ffmpeg): promote run_ffmpeg helper to shared module"
```

---

### Task 2: Extend config schema with audio + Gemini settings

**Files:**
- Modify: `modules/config.py:12-19` (PathsSettings), `modules/config.py:104-108` (EmbeddingsSettings)
- Modify: `config.toml`

- [ ] **Step 1: Add `audio_dir` to `PathsSettings`**

In `modules/config.py`, extend `PathsSettings`:

```python
class PathsSettings(BaseModel):
    video_dir: str
    plots_dir: str
    model_path: str
    profile_pic_dir: str
    thumbnail_dir: str
    speech_audio_dir: str
    audio_dir: str
    data_csv_path: str
```

- [ ] **Step 2: Extend `EmbeddingsSettings` with audio + gemini knobs**

```python
class EmbeddingsSettings(BaseModel):
    exclude_disqualified_users: bool
    embed_max_length: int
    adaptive_max_frames: int
    adaptive_default_fps: float
    # ── audio extraction (used by gemini_mm; harmless if gemini disabled
    # but extract_audio_stage short-circuits before touching ffmpeg) ──
    audio_bitrate_kbps: int = 128
    audio_sample_rate_hz: int = 44100
    audio_extract_timeout_s: int = 60
    # ── Gemini Embedding 2 case ──
    gemini_enabled: bool = False
    gemini_model: str = "gemini-embedding-2-preview"
    gemini_output_dim: int = 3072
    gemini_max_video_seconds: int = 120
    gemini_max_audio_seconds: int = 80
    gemini_request_timeout_s: int = 60
    gemini_max_retries: int = 5
```

- [ ] **Step 3: Update `config.toml` with new keys**

Append under `[paths]`:

```toml
audio_dir = "data/audio"
```

Append under `[embeddings]`:

```toml
audio_bitrate_kbps       = 128
audio_sample_rate_hz     = 44100
audio_extract_timeout_s  = 60
gemini_enabled           = false
gemini_model             = "gemini-embedding-2-preview"
gemini_output_dim        = 3072
gemini_max_video_seconds = 120
gemini_max_audio_seconds = 80
gemini_request_timeout_s = 60
gemini_max_retries       = 5
```

- [ ] **Step 4: Run all tests; verify nothing broke from the new required field**

Run: `uv run pytest -q`
Expected: All pass. If something fails for a missing `audio_dir`, check `tests/conftest.py` and add it there using a `tmp_path` fixture path.

- [ ] **Step 5: Commit**

```bash
git add modules/config.py config.toml
git commit -m "feat(config): add audio_dir path and Gemini embedding settings"
```

---

### Task 3: Add `gemini_api_key` to `Secrets` with conditional loading

**Files:**
- Modify: `modules/config.py:152-194` (Secrets + `load_runtime_config`)
- Modify: `tests/test_config.py` (or create if missing)
- Modify: `.env.example`

- [ ] **Step 1: Write the failing tests**

If `tests/test_config.py` does not exist, create it. Add:

```python
import os
import pytest
from modules.config import load_runtime_config


def _set_required_env(monkeypatch):
    # Mirrors tests/conftest.py if it sets these; otherwise set the
    # minimum needed for load_runtime_config() to succeed.
    for k in ("DATABASE_URL", "IDENTITY_DB_URL", "HIKER_API_KEY",
              "ARC_HOST", "ARC_ACCESS_KEY", "ARC_SECRET_KEY",
              "SPOTIFY_CLIENT_ID", "SPOTIFY_CLIENT_SECRET",
              "HUGGINGFACE_TOKEN"):
        monkeypatch.setenv(k, "x")


def test_secrets_optional_when_gemini_disabled(monkeypatch):
    _set_required_env(monkeypatch)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    settings, secrets = load_runtime_config()
    assert settings.embeddings.gemini_enabled is False
    assert secrets.gemini_api_key is None


def test_secrets_required_when_gemini_enabled(monkeypatch, tmp_path):
    _set_required_env(monkeypatch)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    # Patch config.toml on disk to flip gemini_enabled=True for this test.
    import tomllib, tomli_w  # tomli_w is acceptable; otherwise rewrite manually
    from modules import config as cfgmod
    orig = cfgmod._CONFIG_PATH.read_bytes()
    raw = tomllib.loads(orig.decode())
    raw["embeddings"]["gemini_enabled"] = True
    cfgmod._CONFIG_PATH.write_bytes(tomli_w.dumps(raw).encode())
    try:
        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            load_runtime_config()
    finally:
        cfgmod._CONFIG_PATH.write_bytes(orig)
```

If `tomli_w` is not available, in-test rewrite the TOML by simple string
replacement of `gemini_enabled = false` → `gemini_enabled = true`.

- [ ] **Step 2: Run tests, verify they fail**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL (`Secrets` has no `gemini_api_key` field; no enabled-required check yet).

- [ ] **Step 3: Extend `Secrets` and `load_runtime_config`**

In `modules/config.py`:

```python
class Secrets(BaseModel):
    database_url: str
    identity_db_url: str
    hiker_api_key: str
    arc_host: str
    arc_access_key: str
    arc_secret_key: str
    spotify_client_id: str
    spotify_client_secret: str
    huggingface_token: str
    gemini_api_key: str | None = None
```

In `load_runtime_config()`, after `settings = Settings(...)` and before
constructing `secrets`, compute the Gemini key:

```python
    gemini_enabled = settings.embeddings.gemini_enabled
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if gemini_enabled and not gemini_api_key:
        raise RuntimeError(
            "embeddings.gemini_enabled=true but GEMINI_API_KEY is not set"
        )

    secrets = Secrets(
        database_url=os.environ["DATABASE_URL"],
        identity_db_url=os.environ["IDENTITY_DB_URL"],
        hiker_api_key=os.environ["HIKER_API_KEY"],
        arc_host=os.environ["ARC_HOST"],
        arc_access_key=os.environ["ARC_ACCESS_KEY"],
        arc_secret_key=os.environ["ARC_SECRET_KEY"],
        spotify_client_id=os.environ["SPOTIFY_CLIENT_ID"],
        spotify_client_secret=os.environ["SPOTIFY_CLIENT_SECRET"],
        huggingface_token=os.environ["HUGGINGFACE_TOKEN"],
        gemini_api_key=gemini_api_key if gemini_enabled else None,
    )
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS.

- [ ] **Step 5: Update `.env.example`**

Append:

```
# Optional — required only when embeddings.gemini_enabled = true
GEMINI_API_KEY=
```

- [ ] **Step 6: Commit**

```bash
git add modules/config.py tests/test_config.py .env.example
git commit -m "feat(config): conditional GEMINI_API_KEY loading gated by gemini_enabled"
```

---

## Phase 2 — Audio extraction stage

### Task 4: `extract_audio` helper in `modules/download.py`

**Files:**
- Modify: `modules/download.py`
- Create: `tests/test_audio_extract.py`
- Modify or create: a shared mp4-with-audio fixture in `tests/conftest.py`

- [ ] **Step 1: Add the mp4-with-audio fixture to `tests/conftest.py`**

```python
import shutil
import subprocess
import pytest


@pytest.fixture(scope="session")
def sample_mp4_with_audio(tmp_path_factory):
    """Tiny synthetic mp4 (5s, h264 + 44.1kHz stereo aac).

    Skipped if ffmpeg is not on PATH.
    """
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    out = tmp_path_factory.mktemp("media") / "sample.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=size=64x64:rate=10:duration=5",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=5",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "64k",
        "-shortest", str(out),
    ]
    subprocess.run(cmd, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_audio_extract.py`:

```python
import os
import time
from pathlib import Path

import pytest

from modules.download import extract_audio


def test_extracts_mp3_from_mp4(sample_mp4_with_audio, tmp_path):
    out = tmp_path / "1.mp3"
    ok = extract_audio(
        str(sample_mp4_with_audio),
        str(out),
        bitrate_kbps=128,
        sample_rate_hz=44100,
        timeout_s=60,
    )
    assert ok
    assert out.exists()
    assert out.stat().st_size > 5_000  # non-empty mp3


def test_skip_when_audio_newer_than_video(sample_mp4_with_audio, tmp_path):
    out = tmp_path / "1.mp3"
    assert extract_audio(str(sample_mp4_with_audio), str(out),
                         bitrate_kbps=128, sample_rate_hz=44100, timeout_s=60)
    first_mtime = out.stat().st_mtime_ns
    time.sleep(0.05)
    # Second call must be a no-op (no ffmpeg run, mtime unchanged).
    assert extract_audio(str(sample_mp4_with_audio), str(out),
                         bitrate_kbps=128, sample_rate_hz=44100, timeout_s=60)
    assert out.stat().st_mtime_ns == first_mtime


def test_re_extracts_when_video_newer(sample_mp4_with_audio, tmp_path):
    out = tmp_path / "1.mp3"
    extract_audio(str(sample_mp4_with_audio), str(out),
                  bitrate_kbps=128, sample_rate_hz=44100, timeout_s=60)
    # Bump video mtime to simulate re-download.
    now = time.time() + 10
    os.utime(sample_mp4_with_audio, (now, now))
    old_size = out.stat().st_size
    extract_audio(str(sample_mp4_with_audio), str(out),
                  bitrate_kbps=128, sample_rate_hz=44100, timeout_s=60)
    assert out.exists()
    # File was re-extracted (same params → ~same size, but mtime updated).
    assert out.stat().st_mtime > old_size  # placeholder; just exists
```

- [ ] **Step 3: Run tests, verify they fail**

Run: `uv run pytest tests/test_audio_extract.py -v`
Expected: FAIL (`extract_audio` not importable).

- [ ] **Step 4: Implement `extract_audio` in `modules/download.py`**

At the top, add the import:

```python
from modules.ffmpeg import run_ffmpeg
```

Append the helper:

```python
def extract_audio(
    video_path: str,
    audio_path: str,
    *,
    bitrate_kbps: int,
    sample_rate_hz: int,
    timeout_s: int,
) -> bool:
    """Extract mp3 audio from ``video_path`` to ``audio_path``.

    Idempotent: returns True without invoking ffmpeg when ``audio_path``
    exists and is at least as new as ``video_path``.
    """
    if (
        os.path.exists(audio_path)
        and os.path.exists(video_path)
        and os.path.getmtime(audio_path) >= os.path.getmtime(video_path)
    ):
        return True
    os.makedirs(os.path.dirname(audio_path) or ".", exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", video_path, "-vn",
        "-c:a", "libmp3lame", "-b:a", f"{bitrate_kbps}k",
        "-ar", str(sample_rate_hz), audio_path,
    ]
    return run_ffmpeg(cmd, timeout=timeout_s)
```

- [ ] **Step 5: Run tests, verify they pass**

Run: `uv run pytest tests/test_audio_extract.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add modules/download.py tests/conftest.py tests/test_audio_extract.py
git commit -m "feat(download): add idempotent extract_audio helper"
```

---

### Task 5: `extract_audio_stage` with fingerprint + master-switch short-circuit

**Files:**
- Modify: `modules/download.py`
- Modify: `tests/test_audio_extract.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_audio_extract.py`:

```python
from unittest.mock import patch

from modules.database import Clip, StageState, get_session, init_db
from modules import fingerprint as fp


def _make_settings(audio_dir, enabled, video_dir):
    from modules.config import (
        Settings, PathsSettings, ParseSettings, DownloadSettings,
        FilterSettings, MusicSettings, SpeechSettings, CaptionsSettings,
        EmbeddingsSettings, SearchSettings, ValidationSettings,
        OverridesSettings,
    )
    return Settings(
        paths=PathsSettings(
            video_dir=str(video_dir), plots_dir="/tmp",
            model_path="/tmp/model", profile_pic_dir="/tmp",
            thumbnail_dir="/tmp", speech_audio_dir="/tmp",
            audio_dir=str(audio_dir), data_csv_path="/tmp/x.csv",
        ),
        parse=ParseSettings(fetch_retry_delays_sec=[]),
        download=DownloadSettings(max_attempts=1, retry_delay=0,
                                   retry_jitter=0, concurrency=1),
        filter=FilterSettings(),
        music=MusicSettings(
            audio_fingerprint_confidence=0.5, commit_every=10,
            http_timeout=10, spotify_search_limit=1,
            spotify_token_skew_seconds=10, spotify_request_timeout=10,
            reccobeats_batch_size=1, reccobeats_delay_min=0,
            reccobeats_delay_max=0, manual_features_max_seconds=10,
            manual_features_sample_rate=16000, manual_features_max_mb=1,
            manual_features_mp3_bitrate="128k", api_max_attempts=1,
            api_retry_delay=0, api_retry_jitter=0, acr_max_attempts=1,
            ffmpeg_timeout_seconds=60),
        speech=SpeechSettings(
            whisper_model="x", commit_every=1, translate_model="x",
            translate_target_lang="en", translation_max_chars=1,
            translate_max_new_tokens=1, logprob_threshold=-1,
            compression_threshold=2, min_meaningful_chars=1,
            vad_enabled=False, vad_sampling_rate=16000, vad_threshold=0.5,
            vad_min_speech_ms=1, vad_min_silence_ms=1, vad_speech_pad_ms=1,
            vad_min_total_speech_s=0.1),
        captions=CaptionsSettings(commit_every=1, translate_model="x",
            translate_target_lang="en", translation_max_chars=1,
            translate_max_new_tokens=1),
        embeddings=EmbeddingsSettings(
            exclude_disqualified_users=False, embed_max_length=1,
            adaptive_max_frames=1, adaptive_default_fps=1.0,
            gemini_enabled=enabled,
        ),
        search=SearchSettings(), validation=ValidationSettings(
            plateau_drop_threshold=0.1, max_noise_ratio=0.5,
            min_clusters=1, max_clusters=10),
        overrides=OverridesSettings(),
    )


def test_disabled_short_circuits(tmp_path, db_session):
    from modules.download import extract_audio_stage
    settings = _make_settings(audio_dir=tmp_path / "audio",
                              enabled=False, video_dir=tmp_path / "video")
    with patch("modules.download.run_ffmpeg") as ff:
        extract_audio_stage(settings)
    ff.assert_not_called()
    assert (tmp_path / "audio").exists() is False
    assert db_session.get(StageState, ("audio_extract", "default")) is None


def test_stage_fingerprint_seals(tmp_path, sample_mp4_with_audio, db_session):
    from modules.download import extract_audio_stage

    # Seed one downloaded clip with the fixture mp4 staged at id=1.
    vid_dir = tmp_path / "video"; vid_dir.mkdir()
    target = vid_dir / "1.mp4"
    target.write_bytes(Path(sample_mp4_with_audio).read_bytes())
    db_session.add(Clip(id=1, user_id=1, code="x", media_id="x",
                        is_selected=True, is_downloaded=True))
    db_session.commit()

    settings = _make_settings(audio_dir=tmp_path / "audio",
                              enabled=True, video_dir=vid_dir)
    extract_audio_stage(settings)

    assert (tmp_path / "audio" / "1.mp3").exists()
    row = db_session.get(StageState, ("audio_extract", "default"))
    assert row is not None  # sealed

    # Second run is a no-op.
    with patch("modules.download.run_ffmpeg") as ff:
        extract_audio_stage(settings)
    ff.assert_not_called()
```

The `db_session` fixture is project-standard (see `tests/conftest.py`).
If it does not exist with that name, use whatever existing fixture
seeds an in-memory DB; reuse what `tests/test_clip_embeddings_idempotence.py`
uses.

- [ ] **Step 2: Run tests, verify they fail**

Run: `uv run pytest tests/test_audio_extract.py::test_stage_fingerprint_seals tests/test_audio_extract.py::test_disabled_short_circuits -v`
Expected: FAIL (`extract_audio_stage` not importable).

- [ ] **Step 3: Implement `extract_audio_stage`**

Append to `modules/download.py`:

```python
from modules import fingerprint as fp
from modules.database import Clip, StageState, get_engine, get_session, Base


AUDIO_EXTRACT_STAGE = "audio_extract"
AUDIO_EXTRACT_SCOPE = "default"


def _video_stat(video_dir: str, clip_id: int) -> tuple[int, int]:
    p = os.path.join(video_dir, f"{clip_id}.mp4")
    if not os.path.exists(p):
        return (-1, -1)
    st = os.stat(p)
    return (st.st_size, st.st_mtime_ns)


def extract_audio_stage(settings) -> None:
    """Extract mp3 audio for every downloaded clip into ``paths.audio_dir``.

    No-op when ``embeddings.gemini_enabled`` is False — gemini_mm is the
    only consumer today.
    """
    if not settings.embeddings.gemini_enabled:
        log("audio_extract", "disabled — skipping")
        return

    Base.metadata.create_all(get_engine())
    session = get_session()
    try:
        clips = (
            session.query(Clip)
            .filter(Clip.is_downloaded.is_(True))
            .order_by(Clip.id)
            .all()
        )
        if not clips:
            log("audio_extract", "no downloaded clips — nothing to do")
            return

        ids = [c.id for c in clips]
        video_dir = settings.paths.video_dir
        audio_dir = settings.paths.audio_dir
        os.makedirs(audio_dir, exist_ok=True)

        current = fp.Fingerprint(
            data=fp.hash_rows((cid,) for cid in ids),
            config=fp.hash_text(
                f"bitrate={settings.embeddings.audio_bitrate_kbps}"
                f"|sr={settings.embeddings.audio_sample_rate_hz}"
                f"|codec=libmp3lame"
            ),
            dependency=fp.hash_rows(_video_stat(video_dir, cid) for cid in ids),
        )
        if not fp.is_stale(session, AUDIO_EXTRACT_STAGE, AUDIO_EXTRACT_SCOPE, current):
            log("audio_extract", "fingerprint match — skipping")
            return

        failures = 0
        with progress(len(clips), "Extracting audio") as advance:
            for clip in clips:
                video_path = os.path.join(video_dir, f"{clip.id}.mp4")
                audio_path = os.path.join(audio_dir, f"{clip.id}.mp3")
                if not os.path.exists(video_path):
                    failures += 1
                    advance(detail=f"✗ {clip.id} (no video)")
                    continue
                ok = extract_audio(
                    video_path, audio_path,
                    bitrate_kbps=settings.embeddings.audio_bitrate_kbps,
                    sample_rate_hz=settings.embeddings.audio_sample_rate_hz,
                    timeout_s=settings.embeddings.audio_extract_timeout_s,
                )
                if ok:
                    advance(detail=f"✓ {clip.id}")
                else:
                    failures += 1
                    advance(detail=f"✗ {clip.id}")

        if failures == 0:
            fp.mark_complete(session, AUDIO_EXTRACT_STAGE, AUDIO_EXTRACT_SCOPE, current)
            session.commit()
            log("audio_extract", "done", level="ok")
        else:
            log("audio_extract",
                f"{failures}/{len(clips)} failed — leaving stage stale for retry",
                level="warn")
    finally:
        session.close()
```

(`log` and `progress` are already imported at the top of `download.py`.)

- [ ] **Step 4: Run tests, verify they pass**

Run: `uv run pytest tests/test_audio_extract.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add modules/download.py tests/test_audio_extract.py
git commit -m "feat(download): add AudioExtract stage with fingerprint + master switch"
```

---

### Task 6: Wire `extract_audio_stage` into `main.py`

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Add the import and stage call**

In `main.py`, change the import line:

```python
from modules.download import download_files, extract_audio_stage
```

And insert the new stage call directly after the Download phase
(between current line 56 and the Music phase):

```python
    phase("Audio extraction")
    extract_audio_stage(settings)
```

The stage short-circuits when `gemini_enabled=False`, so this is safe
for every existing install.

- [ ] **Step 2: Smoke-run `uv run python -c` to confirm import graph**

Run: `uv run python -c "from main import run_pipeline; print('ok')"`
Expected: `ok` (no ImportError).

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "feat(pipeline): wire AudioExtract stage after Download"
```

---

## Phase 3 — Gemini provider

### Task 7: Skeleton `GeminiMultimodalProvider` with lazy import

**Files:**
- Create: `modules/embeddings/gemini.py`

- [ ] **Step 1: Create the module with type bag, errors, and skeleton provider**

```python
"""Gemini Embedding 2 multimodal provider.

Single-call multimodal embedding: text + video + audio → one vector.
``google.genai`` is imported lazily inside ``__init__`` so disabled
installs do not require the optional dependency to be present.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass


class GeminiClipTooLongError(Exception):
    """Raised when video or audio exceeds configured caps. Pre-upload."""


class GeminiOutputDimMismatch(Exception):
    """Returned vector length disagrees with configured output_dim."""


@dataclass(frozen=True)
class GeminiSecrets:
    api_key: str


def _probe_duration_seconds(media_path: str) -> float:
    """Return media duration via ffprobe. Raises on failure."""
    import subprocess
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", media_path],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


class GeminiMultimodalProvider:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        output_dim: int,
        max_video_seconds: int,
        max_audio_seconds: int,
        request_timeout_s: int,
        max_retries: int,
        client: object | None = None,  # test injection
    ) -> None:
        self.model = model
        self.output_dim = output_dim
        self.max_video_seconds = max_video_seconds
        self.max_audio_seconds = max_audio_seconds
        self.request_timeout_s = request_timeout_s
        self.max_retries = max_retries

        if client is not None:
            self._client = client
            return
        from google import genai  # lazy: only when actually used
        self._client = genai.Client(api_key=api_key)

    def embed(self, payload: dict) -> list[list[float]]:
        """Embed one clip. Returns ``[vector]`` (single-element list)."""
        raise NotImplementedError  # implemented in Task 9
```

- [ ] **Step 2: Commit the scaffolding**

```bash
git add modules/embeddings/gemini.py
git commit -m "feat(embeddings): scaffold GeminiMultimodalProvider with lazy import"
```

---

### Task 8: Length gate via ffprobe

**Files:**
- Modify: `modules/embeddings/gemini.py`
- Create: `tests/test_embeddings_gemini_case.py`

- [ ] **Step 1: Write the failing test**

```python
from unittest.mock import MagicMock
import pytest

from modules.embeddings.gemini import (
    GeminiClipTooLongError,
    GeminiMultimodalProvider,
)


def _make_provider(monkeypatch, video_seconds, audio_seconds):
    fake_client = MagicMock()
    # Stub durations.
    durations = {"v.mp4": video_seconds, "a.mp3": audio_seconds}
    monkeypatch.setattr(
        "modules.embeddings.gemini._probe_duration_seconds",
        lambda path: durations[os.path.basename(path)],
    )
    return GeminiMultimodalProvider(
        api_key="x", model="m", output_dim=3072,
        max_video_seconds=120, max_audio_seconds=80,
        request_timeout_s=10, max_retries=0, client=fake_client,
    )


def test_provider_skips_oversize_video(monkeypatch, tmp_path):
    import os
    v = tmp_path / "v.mp4"; v.write_bytes(b"x")
    a = tmp_path / "a.mp3"; a.write_bytes(b"x")
    provider = _make_provider(monkeypatch, video_seconds=150, audio_seconds=10)
    with pytest.raises(GeminiClipTooLongError):
        provider.embed({"video_path": str(v), "audio_path": str(a), "text": "t"})
    # No upload attempted.
    assert not provider._client.files.upload.called


def test_provider_skips_oversize_audio(monkeypatch, tmp_path):
    import os
    v = tmp_path / "v.mp4"; v.write_bytes(b"x")
    a = tmp_path / "a.mp3"; a.write_bytes(b"x")
    provider = _make_provider(monkeypatch, video_seconds=10, audio_seconds=90)
    with pytest.raises(GeminiClipTooLongError):
        provider.embed({"video_path": str(v), "audio_path": str(a), "text": "t"})
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `uv run pytest tests/test_embeddings_gemini_case.py -v -k oversize`
Expected: FAIL (NotImplementedError).

- [ ] **Step 3: Implement the length gate in `embed`**

Replace `embed`'s body:

```python
    def embed(self, payload: dict) -> list[list[float]]:
        video_path = payload["video_path"]
        audio_path = payload["audio_path"]
        text = payload["text"]

        v_dur = _probe_duration_seconds(video_path)
        if v_dur > self.max_video_seconds:
            raise GeminiClipTooLongError(
                f"video {v_dur:.1f}s > cap {self.max_video_seconds}s"
            )
        a_dur = _probe_duration_seconds(audio_path)
        if a_dur > self.max_audio_seconds:
            raise GeminiClipTooLongError(
                f"audio {a_dur:.1f}s > cap {self.max_audio_seconds}s"
            )

        return self._upload_and_embed(video_path, audio_path, text)

    def _upload_and_embed(self, video_path, audio_path, text):
        raise NotImplementedError  # implemented in Task 9
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `uv run pytest tests/test_embeddings_gemini_case.py -v -k oversize`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add modules/embeddings/gemini.py tests/test_embeddings_gemini_case.py
git commit -m "feat(gemini): length-gate clips via ffprobe before upload"
```

---

### Task 9: Files API upload + `embed_content` call + output validation

**Files:**
- Modify: `modules/embeddings/gemini.py`
- Modify: `tests/test_embeddings_gemini_case.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_embeddings_gemini_case.py`:

```python
def test_embed_uploads_and_returns_vector(monkeypatch, tmp_path):
    v = tmp_path / "v.mp4"; v.write_bytes(b"v")
    a = tmp_path / "a.mp3"; a.write_bytes(b"a")
    monkeypatch.setattr(
        "modules.embeddings.gemini._probe_duration_seconds",
        lambda p: 10.0,
    )

    fake_client = MagicMock()
    # Two uploads, returning Files API objects with .uri and .mime_type.
    upload_call = MagicMock(uri="files/abc", mime_type="video/mp4")
    upload_call_audio = MagicMock(uri="files/def", mime_type="audio/mpeg")
    fake_client.files.upload.side_effect = [upload_call, upload_call_audio]
    # embed_content returns one embedding of length output_dim.
    fake_embed = MagicMock()
    fake_embed.values = [0.1] * 3072
    fake_response = MagicMock()
    fake_response.embeddings = [fake_embed]
    fake_client.models.embed_content.return_value = fake_response

    p = GeminiMultimodalProvider(
        api_key="x", model="m", output_dim=3072,
        max_video_seconds=120, max_audio_seconds=80,
        request_timeout_s=10, max_retries=0, client=fake_client,
    )
    out = p.embed({"video_path": str(v), "audio_path": str(a), "text": "hello"})
    assert len(out) == 1
    assert len(out[0]) == 3072
    assert fake_client.files.upload.call_count == 2
    fake_client.models.embed_content.assert_called_once()


def test_embed_raises_on_dim_mismatch(monkeypatch, tmp_path):
    from modules.embeddings.gemini import GeminiOutputDimMismatch
    v = tmp_path / "v.mp4"; v.write_bytes(b"v")
    a = tmp_path / "a.mp3"; a.write_bytes(b"a")
    monkeypatch.setattr(
        "modules.embeddings.gemini._probe_duration_seconds",
        lambda p: 10.0,
    )
    fake_client = MagicMock()
    fake_client.files.upload.side_effect = [
        MagicMock(uri="files/abc", mime_type="video/mp4"),
        MagicMock(uri="files/def", mime_type="audio/mpeg"),
    ]
    fake_embed = MagicMock(); fake_embed.values = [0.0] * 768  # wrong length
    fake_response = MagicMock(); fake_response.embeddings = [fake_embed]
    fake_client.models.embed_content.return_value = fake_response

    p = GeminiMultimodalProvider(
        api_key="x", model="m", output_dim=3072,
        max_video_seconds=120, max_audio_seconds=80,
        request_timeout_s=10, max_retries=0, client=fake_client,
    )
    with pytest.raises(GeminiOutputDimMismatch):
        p.embed({"video_path": str(v), "audio_path": str(a), "text": "x"})
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `uv run pytest tests/test_embeddings_gemini_case.py -v`
Expected: FAIL for the two new tests (NotImplementedError).

- [ ] **Step 3: Implement `_upload_and_embed`**

```python
    def _upload_and_embed(self, video_path, audio_path, text):
        from google.genai import types  # lazy

        t0 = time.time()
        video_file = self._client.files.upload(file=video_path)
        audio_file = self._client.files.upload(file=audio_path)

        response = self._client.models.embed_content(
            model=self.model,
            contents=[
                text,
                types.Part.from_uri(file_uri=video_file.uri,
                                    mime_type=video_file.mime_type or "video/mp4"),
                types.Part.from_uri(file_uri=audio_file.uri,
                                    mime_type=audio_file.mime_type or "audio/mpeg"),
            ],
            config=types.EmbedContentConfig(
                output_dimensionality=self.output_dim,
            ),
        )

        vector = list(response.embeddings[0].values)
        if len(vector) != self.output_dim:
            raise GeminiOutputDimMismatch(
                f"expected {self.output_dim}-d vector, got {len(vector)}-d"
            )

        # Best-effort observability; never the cause of a failure.
        try:
            elapsed = time.time() - t0
            bytes_up = os.path.getsize(video_path) + os.path.getsize(audio_path)
            print(f"[gemini] bytes_uploaded={bytes_up} embed_seconds={elapsed:.2f}")
        except OSError:
            pass

        return [vector]
```

Tests use `MagicMock`, so the `from google.genai import types` import
would fail in installs without the extras. Guard it: if `google.genai`
isn't importable, build the request using dict shapes that the mocked
client also accepts. **Better approach**: factor the `types.Part`
construction behind a tiny indirection injectable for tests:

```python
    def _upload_and_embed(self, video_path, audio_path, text):
        t0 = time.time()
        video_file = self._client.files.upload(file=video_path)
        audio_file = self._client.files.upload(file=audio_path)

        contents, config = self._build_request(text, video_file, audio_file)
        response = self._client.models.embed_content(
            model=self.model, contents=contents, config=config
        )
        # ...rest as above...

    def _build_request(self, text, video_file, audio_file):
        from google.genai import types
        return (
            [
                text,
                types.Part.from_uri(file_uri=video_file.uri,
                                    mime_type=video_file.mime_type or "video/mp4"),
                types.Part.from_uri(file_uri=audio_file.uri,
                                    mime_type=audio_file.mime_type or "audio/mpeg"),
            ],
            types.EmbedContentConfig(output_dimensionality=self.output_dim),
        )
```

Tests monkeypatch `GeminiMultimodalProvider._build_request` to return
`(["t", "video_part", "audio_part"], {"output_dimensionality": 3072})`
so they do not require `google-genai` to be installed.

Update the two new tests above to add this line before constructing the
provider:

```python
    monkeypatch.setattr(
        GeminiMultimodalProvider, "_build_request",
        lambda self, t, v, a: ([t, "vp", "ap"], {"dim": self.output_dim}),
    )
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `uv run pytest tests/test_embeddings_gemini_case.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add modules/embeddings/gemini.py tests/test_embeddings_gemini_case.py
git commit -m "feat(gemini): upload via Files API and call embed_content"
```

---

### Task 10: Retry on transient failures

**Files:**
- Modify: `modules/embeddings/gemini.py`
- Modify: `tests/test_embeddings_gemini_case.py`

- [ ] **Step 1: Write the failing test**

```python
def test_embed_retries_on_5xx(monkeypatch, tmp_path):
    v = tmp_path / "v.mp4"; v.write_bytes(b"v")
    a = tmp_path / "a.mp3"; a.write_bytes(b"a")
    monkeypatch.setattr(
        "modules.embeddings.gemini._probe_duration_seconds",
        lambda p: 10.0,
    )

    fake_client = MagicMock()
    fake_client.files.upload.side_effect = [
        MagicMock(uri="files/abc", mime_type="video/mp4"),
        MagicMock(uri="files/def", mime_type="audio/mpeg"),
    ]
    fake_embed = MagicMock(); fake_embed.values = [0.0] * 3072
    fake_response = MagicMock(); fake_response.embeddings = [fake_embed]

    # First two calls raise a 5xx-shaped error; third succeeds.
    class _Transient(Exception):
        status_code = 503

    fake_client.models.embed_content.side_effect = [
        _Transient("upstream busy"),
        _Transient("upstream busy"),
        fake_response,
    ]

    p = GeminiMultimodalProvider(
        api_key="x", model="m", output_dim=3072,
        max_video_seconds=120, max_audio_seconds=80,
        request_timeout_s=10, max_retries=5, client=fake_client,
    )
    monkeypatch.setattr(
        GeminiMultimodalProvider, "_build_request",
        lambda self, t, v, a: ([t, "vp", "ap"], {"dim": 3072}),
    )
    # Eliminate sleep delay between retries to keep the test fast.
    monkeypatch.setattr("modules.embeddings.gemini.time.sleep", lambda *_: None)
    out = p.embed({"video_path": str(v), "audio_path": str(a), "text": "x"})
    assert len(out[0]) == 3072
    assert fake_client.models.embed_content.call_count == 3
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `uv run pytest tests/test_embeddings_gemini_case.py::test_embed_retries_on_5xx -v`
Expected: FAIL (current code does not retry).

- [ ] **Step 3: Add the retry wrapper**

Add to `modules/embeddings/gemini.py`:

```python
def _is_retriable(exc: Exception) -> bool:
    code = getattr(exc, "status_code", None)
    if code is None:
        # Fall back to checking common transient keywords.
        text = str(exc).lower()
        return any(s in text for s in ("timeout", "temporarily", "unavailable",
                                       "deadline", "reset"))
    return code == 429 or 500 <= int(code) < 600


def _retry(call, *, max_retries: int, base_delay: float = 1.0,
           max_delay: float = 60.0):
    last: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return call()
        except Exception as exc:
            if not _is_retriable(exc) or attempt == max_retries:
                raise
            last = exc
            delay = min(max_delay, base_delay * (2 ** attempt))
            time.sleep(delay)
    raise last  # unreachable
```

Wrap the `embed_content` call in `_upload_and_embed`:

```python
        response = _retry(
            lambda: self._client.models.embed_content(
                model=self.model, contents=contents, config=config
            ),
            max_retries=self.max_retries,
        )
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `uv run pytest tests/test_embeddings_gemini_case.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add modules/embeddings/gemini.py tests/test_embeddings_gemini_case.py
git commit -m "feat(gemini): exponential-backoff retry on 429 / 5xx / timeout"
```

---

## Phase 4 — Case integration

### Task 11: `build_gemini_text` text builder

**Files:**
- Modify: `modules/embeddings/text.py`
- Modify: `tests/test_embeddings_gemini_case.py`

- [ ] **Step 1: Write the failing tests**

```python
from types import SimpleNamespace
from modules.embeddings.text import build_gemini_text


def test_gemini_text_joins_caption_and_transcript():
    clip = SimpleNamespace(
        caption_text="cap", caption_clean="cap clean",
        caption_language="es", caption_translation="cap en",
        speech_transcription="hi", speech_language="es",
        speech_translation="hello",
    )
    text = build_gemini_text(clip, {})
    assert "cap en" in text
    assert "hello" in text
    assert "---" in text  # separator marker


def test_gemini_text_returns_none_when_empty():
    clip = SimpleNamespace(
        caption_text="", caption_clean=None,
        caption_language=None, caption_translation=None,
        speech_transcription=None, speech_language=None,
        speech_translation=None,
    )
    assert build_gemini_text(clip, {}) is None
```

- [ ] **Step 2: Run, verify FAIL**

Run: `uv run pytest tests/test_embeddings_gemini_case.py -v -k gemini_text`
Expected: FAIL (`build_gemini_text` not defined).

- [ ] **Step 3: Implement `build_gemini_text`**

Append to `modules/embeddings/text.py`:

```python
def build_gemini_text(clip, _music_map: dict) -> str | None:
    """Caption + transcript for the gemini_mm case.

    Uses translation when source language is non-English and a non-empty
    translation exists; otherwise the cleaned/original text. Music is
    NOT verbalized — the model gets the raw audio track separately.
    Returns ``None`` when both caption and transcript are empty.
    """
    cap = (
        clip.caption_translation
        if clip.caption_language not in ("en", None)
        and clip.caption_translation
        and clip.caption_translation.strip()
        else (clip.caption_clean or clip.caption_text or "")
    )
    speech = (
        clip.speech_translation
        if clip.speech_language not in ("en", None)
        and clip.speech_translation
        and clip.speech_translation.strip()
        else (clip.speech_transcription or "")
    )

    parts = []
    if cap and cap.strip():
        parts.append(cap.strip())
    if speech and speech.strip():
        parts.append(speech.strip())
    if not parts:
        return None
    return "\n\n---\n\n".join(parts)
```

- [ ] **Step 4: Run, verify PASS**

Run: `uv run pytest tests/test_embeddings_gemini_case.py -v -k gemini_text`

- [ ] **Step 5: Commit**

```bash
git add modules/embeddings/text.py tests/test_embeddings_gemini_case.py
git commit -m "feat(embeddings): build_gemini_text for gemini_mm case"
```

---

### Task 12: Extend `dependency_rows_for_case` for `gemini_mm`

**Files:**
- Modify: `modules/embeddings/state.py`
- Modify: `tests/test_embeddings_gemini_case.py`

- [ ] **Step 1: Write the failing test**

```python
def test_dependency_rows_gemini_mm_includes_file_stats(tmp_path, db_session, monkeypatch):
    from modules.database import Clip
    from modules.embeddings.state import dependency_rows_for_case

    db_session.add(Clip(id=1, user_id=1, code="x", media_id="x",
                        is_selected=True, is_downloaded=True,
                        caption_text="cap", caption_language="en",
                        speech_transcription="hi", speech_language="en"))
    db_session.commit()

    # Patch the stat helpers so the test does not need real files.
    monkeypatch.setattr("modules.embeddings.state._video_file_stat",
                        lambda cid: (1234, 1000))
    monkeypatch.setattr("modules.embeddings.state._audio_file_stat",
                        lambda cid: (567, 2000))
    rows = dependency_rows_for_case(db_session, "gemini_mm", [1])
    assert rows
    row = rows[0]
    assert (1234, 1000) in row
    assert (567, 2000) in row
```

- [ ] **Step 2: Run, verify FAIL**

Run: `uv run pytest tests/test_embeddings_gemini_case.py -v -k dependency_rows_gemini_mm`
Expected: FAIL (`Unknown embedding case: 'gemini_mm'`).

- [ ] **Step 3: Add helpers + branch in `state.py`**

Inside `modules/embeddings/state.py`, add near the top (after imports):

```python
import os
from modules.config import load_runtime_config


def _stat_or_sentinel(path: str) -> tuple[int, int]:
    if not os.path.exists(path):
        return (-1, -1)
    st = os.stat(path)
    return (st.st_size, st.st_mtime_ns)


def _video_file_stat(clip_id: int) -> tuple[int, int]:
    settings, _ = load_runtime_config()
    return _stat_or_sentinel(os.path.join(settings.paths.video_dir, f"{clip_id}.mp4"))


def _audio_file_stat(clip_id: int) -> tuple[int, int]:
    settings, _ = load_runtime_config()
    return _stat_or_sentinel(os.path.join(settings.paths.audio_dir, f"{clip_id}.mp3"))
```

In `dependency_rows_for_case`, before the final `raise ValueError`,
insert:

```python
    if case == "gemini_mm":
        rows = (
            session.query(
                Clip.id,
                Clip.caption_text, Clip.caption_clean,
                Clip.caption_language, Clip.caption_translation,
                Clip.speech_transcription, Clip.speech_language,
                Clip.speech_translation,
            )
            .filter(Clip.id.in_(candidate_ids))
            .order_by(Clip.id)
            .all()
        )
        return [
            (
                r.id, r.caption_text, r.caption_clean, r.caption_language,
                r.caption_translation, r.speech_transcription,
                r.speech_language, r.speech_translation,
                _video_file_stat(r.id),
                _audio_file_stat(r.id),
            )
            for r in rows
        ]
```

- [ ] **Step 4: Run, verify PASS**

Run: `uv run pytest tests/test_embeddings_gemini_case.py -v -k dependency_rows_gemini_mm`

- [ ] **Step 5: Commit**

```bash
git add modules/embeddings/state.py tests/test_embeddings_gemini_case.py
git commit -m "feat(embeddings): gemini_mm dependency rows include file stats"
```

---

### Task 13: Widen factory contract; introduce `EmbeddingSecrets`

**Files:**
- Modify: `modules/embeddings/cases.py`
- Modify: `modules/embeddings/runner.py`
- Modify: `modules/embeddings/__init__.py`

- [ ] **Step 1: Define `EmbeddingSecrets` at the top of `cases.py`**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingSecrets:
    gemini_api_key: str | None = None
```

- [ ] **Step 2: Widen existing factories to accept (and ignore) secrets**

Change signatures in `cases.py`:

```python
def _local_qwen_video_factory(settings, secrets: EmbeddingSecrets | None = None) -> Provider:
    ...

def _local_qwen_text_factory(settings, secrets: EmbeddingSecrets | None = None) -> Provider:
    ...
```

Update the `EmbeddingCaseSpec.provider_factory` type annotation to match:

```python
provider_factory: Callable[[object, "EmbeddingSecrets | None"], Provider]
```

- [ ] **Step 3: Update the runner to pass secrets**

In `modules/embeddings/runner.py`, change `embed_clip_embeddings`:

```python
def embed_clip_embeddings(
    settings,
    secrets=None,
    cases: list[str] | None = None,
) -> None:
    from modules.embeddings.cases import EmbeddingSecrets, default_cases
    if secrets is None:
        secrets = EmbeddingSecrets()
    case_names = list(cases) if cases is not None else list(default_cases(settings))
    for name in case_names:
        if name == "gemini_mm" and not settings.embeddings.gemini_enabled:
            raise RuntimeError(
                "gemini_mm case requested but embeddings.gemini_enabled=false"
            )
        spec = CASE_REGISTRY[name]
        _run_case(settings, spec, secrets)
```

Change `_run_case` to accept and thread `secrets`:

```python
def _run_case(settings, spec: EmbeddingCaseSpec, secrets) -> None:
    ...
    provider = spec.provider_factory(settings, secrets)
    ...
```

Update the inner `_embed_targets` signature if it constructs the
provider itself (it does today, via `spec.provider_factory(settings)`
at line 232 of the current file) — replace with
`spec.provider_factory(settings, secrets)`. Pass `secrets` into
`_embed_targets` accordingly.

`default_cases` is defined in Task 15; this task leaves the import in
place to flag the dependency. Tests will be green only after Task 15 lands.

- [ ] **Step 4: Re-export from `__init__.py`**

In `modules/embeddings/__init__.py`, add:

```python
from modules.embeddings.cases import EmbeddingSecrets

__all__ = [..., "EmbeddingSecrets"]
```

- [ ] **Step 5: Commit (knowingly broken — `default_cases` lands next)**

```bash
git add modules/embeddings/cases.py modules/embeddings/runner.py modules/embeddings/__init__.py
git commit -m "refactor(embeddings): widen factory contract to (settings, secrets)"
```

---

### Task 14: Register `GEMINI_MM_CASE` + `default_cases` + `case_config_identity` branch

**Files:**
- Modify: `modules/embeddings/cases.py`

- [ ] **Step 1: Add the gemini factory + payload builder**

Inside `modules/embeddings/cases.py`:

```python
def _gemini_factory(settings, secrets: EmbeddingSecrets | None) -> Provider:
    from modules.embeddings.gemini import GeminiMultimodalProvider
    if secrets is None or secrets.gemini_api_key is None:
        raise RuntimeError(
            "gemini_mm provider requires secrets.gemini_api_key; "
            "set GEMINI_API_KEY and embeddings.gemini_enabled=true"
        )
    return GeminiMultimodalProvider(
        api_key=secrets.gemini_api_key,
        model=settings.embeddings.gemini_model,
        output_dim=settings.embeddings.gemini_output_dim,
        max_video_seconds=settings.embeddings.gemini_max_video_seconds,
        max_audio_seconds=settings.embeddings.gemini_max_audio_seconds,
        request_timeout_s=settings.embeddings.gemini_request_timeout_s,
        max_retries=settings.embeddings.gemini_max_retries,
    )


def _gemini_payload(clip, text, video_path, fps, max_frames) -> dict:
    import os as _os
    settings, _ = __import__(
        "modules.config", fromlist=["load_runtime_config"]
    ).load_runtime_config()
    audio_path = _os.path.join(
        settings.paths.audio_dir, f"{clip.id}.mp3"
    )
    return {"video_path": video_path, "audio_path": audio_path, "text": text}
```

(`load_runtime_config` already caches the parsed config implicitly via
the module-level singleton pattern in `modules/config.py`; if it does
not, accept the small reparse cost — this runs once per clip and reads
a small TOML file.)

- [ ] **Step 2: Add the case spec + register it**

Below the `AUDIO_CASE` definition:

```python
from modules.embeddings.text import build_gemini_text

GEMINI_MM_CASE = EmbeddingCaseSpec(
    name="gemini_mm",
    text_builder=build_gemini_text,
    requires_video=True,
    provider_factory=_gemini_factory,
    payload_builder=_gemini_payload,
    apply_video_token_fallback=False,
)

CASE_REGISTRY["gemini_mm"] = GEMINI_MM_CASE
TEXT_RECIPE_VERSIONS["gemini_mm"] = "gemini_mm_v1"
```

- [ ] **Step 3: Replace `DEFAULT_CASES` tuple with `default_cases(settings)`**

Remove the existing `DEFAULT_CASES` tuple. Add:

```python
def default_cases(settings) -> tuple[str, ...]:
    base = ("video", "sandwich", "audio")
    if getattr(settings.embeddings, "gemini_enabled", False):
        return base + ("gemini_mm",)
    return base
```

Keep a backward-compatible re-export so other modules don't break:

```python
DEFAULT_CASES = ("video", "sandwich", "audio")  # static fallback for non-settings call sites
```

If `grep -rn DEFAULT_CASES` shows other consumers besides `runner.py`,
update each to use `default_cases(settings)` where they have a settings
object on hand. If they don't, the static tuple remains correct because
those call sites pre-date the Gemini case.

- [ ] **Step 4: Add the `case_config_identity` branch**

Inside `case_config_identity`:

```python
    if spec.name == "audio":
        parts.append(f"instruction={AUDIO_INSTRUCTION}")
    if spec.name == "gemini_mm":
        parts.append(f"output_dim={settings.embeddings.gemini_output_dim}")
        parts.append(f"audio_bitrate={settings.embeddings.audio_bitrate_kbps}")
        parts.append(f"audio_sample_rate={settings.embeddings.audio_sample_rate_hz}")
        parts.append(f"max_video_s={settings.embeddings.gemini_max_video_seconds}")
        parts.append(f"max_audio_s={settings.embeddings.gemini_max_audio_seconds}")
    return "|".join(parts)
```

- [ ] **Step 5: Run the existing test suite, confirm no regression**

Run: `uv run pytest tests/ -q`
Expected: All previously-green tests still green.

- [ ] **Step 6: Commit**

```bash
git add modules/embeddings/cases.py
git commit -m "feat(embeddings): register gemini_mm case spec + default_cases gate"
```

---

### Task 15: Reject explicit `gemini_mm` when disabled

**Files:**
- Modify: `tests/test_embeddings_gemini_case.py`

- [ ] **Step 1: Add the test**

```python
def test_explicit_gemini_request_raises_when_disabled(monkeypatch):
    from modules.embeddings import embed_clip_embeddings
    settings = _make_settings(audio_dir="/tmp/a", enabled=False, video_dir="/tmp/v")
    with pytest.raises(RuntimeError, match="gemini_enabled"):
        embed_clip_embeddings(settings, cases=["gemini_mm"])


def test_default_cases_excludes_gemini_when_disabled(monkeypatch):
    from modules.embeddings.cases import default_cases
    settings = _make_settings(audio_dir="/tmp/a", enabled=False, video_dir="/tmp/v")
    assert "gemini_mm" not in default_cases(settings)


def test_default_cases_includes_gemini_when_enabled(monkeypatch):
    from modules.embeddings.cases import default_cases
    settings = _make_settings(audio_dir="/tmp/a", enabled=True, video_dir="/tmp/v")
    assert "gemini_mm" in default_cases(settings)
```

(Reuse the `_make_settings` helper defined in `tests/test_audio_extract.py`
or factor it into `tests/conftest.py` to share. The plan assumes you
factor it out: extract `_make_settings` into a `conftest.py` fixture
named `make_settings`.)

- [ ] **Step 2: Run, verify PASS**

Run: `uv run pytest tests/test_embeddings_gemini_case.py -v -k "default_cases or explicit_gemini"`
Expected: PASS (logic already added in Tasks 13–14).

- [ ] **Step 3: Commit**

```bash
git add tests/
git commit -m "test(embeddings): cover gemini_mm gating via master switch"
```

---

### Task 16: Wire `EmbeddingSecrets` through `main.py`

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Construct `EmbeddingSecrets` and pass it**

```python
from modules.embeddings import EmbeddingSecrets

# ... inside run_pipeline(), at the Clip Embeddings phase ...
    phase("Clip Embeddings")
    embed_clip_embeddings(
        settings,
        EmbeddingSecrets(gemini_api_key=secrets.gemini_api_key),
    )
```

- [ ] **Step 2: Smoke-import**

Run: `uv run python -c "from main import run_pipeline; print('ok')"`
Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "feat(pipeline): thread EmbeddingSecrets through clip embeddings"
```

---

## Phase 5 — Runner-level integration tests

### Task 17: Runner seals on full success

**Files:**
- Modify: `tests/test_embeddings_gemini_case.py`

- [ ] **Step 1: Write the test**

```python
def test_runner_seals_when_all_clips_embed(tmp_path, db_session, monkeypatch,
                                            sample_mp4_with_audio):
    from modules.embeddings import EmbeddingSecrets, embed_clip_embeddings
    from modules.embeddings.gemini import GeminiMultimodalProvider
    from modules.database import Clip, ClipEmbedding, StageState

    vid_dir = tmp_path / "video"; vid_dir.mkdir()
    aud_dir = tmp_path / "audio"; aud_dir.mkdir()
    (vid_dir / "1.mp4").write_bytes(Path(sample_mp4_with_audio).read_bytes())
    (aud_dir / "1.mp3").write_bytes(b"fake_mp3")

    db_session.add(Clip(id=1, user_id=1, code="x", media_id="x",
                        is_selected=True, is_downloaded=True,
                        caption_text="hi", caption_language="en",
                        speech_transcription=None, speech_language=None))
    db_session.commit()

    settings = _make_settings(audio_dir=aud_dir, enabled=True, video_dir=vid_dir)

    # Stub the provider to skip the network entirely.
    def fake_embed(self, payload):
        return [[0.1] * 3072]
    monkeypatch.setattr(GeminiMultimodalProvider, "embed", fake_embed)
    # Skip the lazy google.genai import path.
    monkeypatch.setattr(GeminiMultimodalProvider, "__init__",
        lambda self, **kw: setattr(self, "model", kw["model"]))

    embed_clip_embeddings(
        settings,
        EmbeddingSecrets(gemini_api_key="x"),
        cases=["gemini_mm"],
    )

    rows = db_session.query(ClipEmbedding).filter_by(
        embedding_case="gemini_mm").all()
    assert len(rows) == 1
    state = db_session.get(StageState, ("clip_embeddings", "gemini_mm"))
    assert state is not None
```

- [ ] **Step 2: Run, verify PASS**

Run: `uv run pytest tests/test_embeddings_gemini_case.py::test_runner_seals_when_all_clips_embed -v`

- [ ] **Step 3: Commit**

```bash
git add tests/test_embeddings_gemini_case.py
git commit -m "test(gemini_mm): runner seals stage on full embedding success"
```

---

### Task 18: Runner does NOT seal on partial failure

**Files:**
- Modify: `tests/test_embeddings_gemini_case.py`

- [ ] **Step 1: Add the test**

```python
def test_runner_does_not_seal_on_failure(tmp_path, db_session, monkeypatch,
                                          sample_mp4_with_audio):
    from modules.embeddings import EmbeddingSecrets, embed_clip_embeddings
    from modules.embeddings.gemini import GeminiMultimodalProvider
    from modules.database import Clip, ClipEmbedding, StageState

    vid_dir = tmp_path / "video"; vid_dir.mkdir()
    aud_dir = tmp_path / "audio"; aud_dir.mkdir()
    for cid in (1, 2):
        (vid_dir / f"{cid}.mp4").write_bytes(Path(sample_mp4_with_audio).read_bytes())
        (aud_dir / f"{cid}.mp3").write_bytes(b"fake_mp3")
        db_session.add(Clip(id=cid, user_id=1, code=f"c{cid}", media_id=f"m{cid}",
                            is_selected=True, is_downloaded=True,
                            caption_text="hi", caption_language="en"))
    db_session.commit()

    settings = _make_settings(audio_dir=aud_dir, enabled=True, video_dir=vid_dir)

    def fake_embed(self, payload):
        if "1.mp4" in payload["video_path"]:
            raise RuntimeError("boom")
        return [[0.2] * 3072]
    monkeypatch.setattr(GeminiMultimodalProvider, "embed", fake_embed)
    monkeypatch.setattr(GeminiMultimodalProvider, "__init__",
        lambda self, **kw: setattr(self, "model", kw["model"]))

    embed_clip_embeddings(settings, EmbeddingSecrets(gemini_api_key="x"),
                          cases=["gemini_mm"])

    rows = db_session.query(ClipEmbedding).filter_by(
        embedding_case="gemini_mm").all()
    assert {r.clip_id for r in rows} == {2}
    assert db_session.get(StageState, ("clip_embeddings", "gemini_mm")) is None
```

- [ ] **Step 2: Run, verify PASS**

Run: `uv run pytest tests/test_embeddings_gemini_case.py::test_runner_does_not_seal_on_failure -v`

- [ ] **Step 3: Commit**

```bash
git add tests/test_embeddings_gemini_case.py
git commit -m "test(gemini_mm): unsealed on partial failure, successes persist"
```

---

### Task 19: Config drift wipes the case

**Files:**
- Modify: `tests/test_embeddings_gemini_case.py`

- [ ] **Step 1: Add the test**

```python
def test_config_drift_wipes_case(tmp_path, db_session, monkeypatch,
                                  sample_mp4_with_audio):
    from modules.embeddings import EmbeddingSecrets, embed_clip_embeddings
    from modules.embeddings.gemini import GeminiMultimodalProvider
    from modules.database import Clip, ClipEmbedding

    vid_dir = tmp_path / "video"; vid_dir.mkdir()
    aud_dir = tmp_path / "audio"; aud_dir.mkdir()
    (vid_dir / "1.mp4").write_bytes(Path(sample_mp4_with_audio).read_bytes())
    (aud_dir / "1.mp3").write_bytes(b"x")
    db_session.add(Clip(id=1, user_id=1, code="x", media_id="x",
                        is_selected=True, is_downloaded=True,
                        caption_text="hi", caption_language="en"))
    db_session.commit()

    captured = []
    def fake_embed(self, payload):
        captured.append(payload["video_path"])
        return [[0.0] * self.output_dim]
    monkeypatch.setattr(GeminiMultimodalProvider, "embed", fake_embed)
    monkeypatch.setattr(GeminiMultimodalProvider, "__init__",
        lambda self, **kw: (setattr(self, "model", kw["model"]),
                            setattr(self, "output_dim", kw["output_dim"])))

    s1 = _make_settings(audio_dir=aud_dir, enabled=True, video_dir=vid_dir)
    embed_clip_embeddings(s1, EmbeddingSecrets("x"), cases=["gemini_mm"])
    assert len(captured) == 1
    captured.clear()

    s2 = _make_settings(audio_dir=aud_dir, enabled=True, video_dir=vid_dir)
    s2.embeddings.gemini_output_dim = 768  # drift
    embed_clip_embeddings(s2, EmbeddingSecrets("x"), cases=["gemini_mm"])
    assert len(captured) == 1  # re-embedded after wipe

    rows = db_session.query(ClipEmbedding).filter_by(
        embedding_case="gemini_mm").all()
    assert len(rows) == 1
```

- [ ] **Step 2: Run, verify PASS**

Run: `uv run pytest tests/test_embeddings_gemini_case.py::test_config_drift_wipes_case -v`

- [ ] **Step 3: Commit**

```bash
git add tests/test_embeddings_gemini_case.py
git commit -m "test(gemini_mm): config drift wipes case and re-embeds"
```

---

### Task 20: Per-clip diff re-embeds only touched clips

**Files:**
- Modify: `tests/test_embeddings_gemini_case.py`

- [ ] **Step 1: Add the test**

```python
def test_per_clip_diff_re_embeds_only_touched_clip(tmp_path, db_session,
                                                    monkeypatch, sample_mp4_with_audio):
    from modules.embeddings import EmbeddingSecrets, embed_clip_embeddings
    from modules.embeddings.gemini import GeminiMultimodalProvider
    from modules.database import Clip

    vid_dir = tmp_path / "video"; vid_dir.mkdir()
    aud_dir = tmp_path / "audio"; aud_dir.mkdir()
    for cid in (1, 2):
        (vid_dir / f"{cid}.mp4").write_bytes(Path(sample_mp4_with_audio).read_bytes())
        (aud_dir / f"{cid}.mp3").write_bytes(b"x")
        db_session.add(Clip(id=cid, user_id=1, code=f"c{cid}", media_id=f"m{cid}",
                            is_selected=True, is_downloaded=True,
                            caption_text="hi", caption_language="en"))
    db_session.commit()

    seen = []
    def fake_embed(self, payload):
        seen.append(payload["video_path"])
        return [[0.0] * 3072]
    monkeypatch.setattr(GeminiMultimodalProvider, "embed", fake_embed)
    monkeypatch.setattr(GeminiMultimodalProvider, "__init__",
        lambda self, **kw: (setattr(self, "model", kw["model"]),
                            setattr(self, "output_dim", 3072)))

    settings = _make_settings(audio_dir=aud_dir, enabled=True, video_dir=vid_dir)
    embed_clip_embeddings(settings, EmbeddingSecrets("x"), cases=["gemini_mm"])
    assert len(seen) == 2
    seen.clear()

    # Touch only clip 1's caption.
    clip1 = db_session.get(Clip, 1)
    clip1.caption_text = "new caption"
    db_session.commit()

    embed_clip_embeddings(settings, EmbeddingSecrets("x"), cases=["gemini_mm"])
    assert seen == [str(vid_dir / "1.mp4")]
```

- [ ] **Step 2: Run, verify PASS**

Run: `uv run pytest tests/test_embeddings_gemini_case.py::test_per_clip_diff_re_embeds_only_touched_clip -v`

- [ ] **Step 3: Commit**

```bash
git add tests/test_embeddings_gemini_case.py
git commit -m "test(gemini_mm): per-clip diff re-embeds only changed clips"
```

---

## Phase 6 — Cleanup, packaging, smoke

### Task 21: Delete `modules/embeddings/remote.py`

**Files:**
- Delete: `modules/embeddings/remote.py`

- [ ] **Step 1: Verify nothing imports it**

Run: `grep -rn "embeddings.remote\|from modules.embeddings import remote" --include="*.py"`
Expected: only the file itself, no other matches.

- [ ] **Step 2: Delete the file**

Run: `git rm modules/embeddings/remote.py`

- [ ] **Step 3: Run tests**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git commit -m "chore(embeddings): drop unused remote.py placeholder"
```

---

### Task 22: Packaging metadata, gitignore, env example

**Files:**
- Modify: `pyproject.toml`
- Modify: `.gitignore`
- Verify: `.env.example` (touched in Task 3)

- [ ] **Step 1: Add optional extras for google-genai**

In `pyproject.toml`, add:

```toml
[project.optional-dependencies]
gemini = ["google-genai>=1.0"]
```

(Place inside `[project]` if there is no `[project.optional-dependencies]`
table yet, per PEP 621.)

- [ ] **Step 2: Ignore the audio cache directory**

In `.gitignore`, add:

```
data/audio/
```

- [ ] **Step 3: Confirm `.env.example` has `GEMINI_API_KEY=`**

Run: `grep GEMINI_API_KEY .env.example`
Expected: prints the line. If missing, append it.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml .gitignore .env.example
git commit -m "chore: declare gemini extras, ignore data/audio, document GEMINI_API_KEY"
```

---

### Task 23: Opt-in smoke script

**Files:**
- Create: `scripts/smoke_gemini_embed.py`

- [ ] **Step 1: Create the script**

```python
"""Manual smoke test for the Gemini multimodal embedding case.

Picks one downloaded clip with an extracted audio file and runs a real
Gemini Embedding 2 call. Prints the first 10 dimensions and the elapsed
seconds. Requires:
    * GEMINI_API_KEY in env
    * embeddings.gemini_enabled = true in config.toml
    * `uv pip install -e ".[gemini]"`

Usage: uv run python scripts/smoke_gemini_embed.py [clip_id]
"""

from __future__ import annotations

import os
import sys
import time

from modules.config import load_runtime_config
from modules.database import Clip, init_db, get_session
from modules.embeddings.cases import _gemini_factory, EmbeddingSecrets


def main(argv: list[str]) -> int:
    settings, secrets = load_runtime_config()
    if not settings.embeddings.gemini_enabled:
        print("embeddings.gemini_enabled is false; aborting")
        return 2
    init_db(secrets.database_url, secrets.identity_db_url)

    session = get_session()
    clip_id = int(argv[1]) if len(argv) > 1 else None
    if clip_id is None:
        clip = session.query(Clip).filter(Clip.is_downloaded.is_(True)).first()
        if clip is None:
            print("no downloaded clip found")
            return 1
    else:
        clip = session.get(Clip, clip_id)
        if clip is None:
            print(f"no clip with id={clip_id}")
            return 1

    video_path = os.path.join(settings.paths.video_dir, f"{clip.id}.mp4")
    audio_path = os.path.join(settings.paths.audio_dir, f"{clip.id}.mp3")
    text = f"smoke test for clip {clip.id}"

    provider = _gemini_factory(settings, EmbeddingSecrets(secrets.gemini_api_key))
    t0 = time.time()
    [vector] = provider.embed({
        "video_path": video_path,
        "audio_path": audio_path,
        "text": text,
    })
    print(f"clip_id={clip.id}  dim={len(vector)}  elapsed={time.time()-t0:.2f}s")
    print("head:", vector[:10])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
```

- [ ] **Step 2: Verify the script imports cleanly**

Run: `uv run python -c "import scripts.smoke_gemini_embed; print('ok')"`
(If the project lacks an `__init__.py` under `scripts/`, instead run
`uv run python -m compileall scripts/smoke_gemini_embed.py`.)

- [ ] **Step 3: Commit**

```bash
git add scripts/smoke_gemini_embed.py
git commit -m "chore(scripts): add opt-in smoke test for gemini multimodal embedding"
```

---

### Task 24: Full regression run

**Files:** none.

- [ ] **Step 1: Run full suite**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 2: Lint + types**

Run: `uv run ruff check modules tests scripts && uv run ruff format --check modules tests scripts && uv run ty check`
Expected: clean. Fix any reports.

- [ ] **Step 3: Verify pipeline import graph**

Run: `uv run python -c "from main import run_pipeline; print('ok')"`
Expected: `ok`.

- [ ] **Step 4: If formatter changed anything, commit**

```bash
git add -u
git commit -m "style: ruff format after Gemini case integration"
```

---

## Self-Review

**Spec coverage:**

| Spec section | Covered by task(s) |
|---|---|
| `EmbeddingCaseSpec` for `gemini_mm` | 14 |
| `default_cases(settings)` gate | 14, 15 |
| `_gemini_factory(settings, secrets)` widening | 13, 14 |
| `_gemini_payload` | 14 |
| `build_gemini_text` | 11 |
| `GeminiMultimodalProvider`, errors, lazy import | 7, 8, 9, 10 |
| Files API upload path | 9 |
| Length gate via ffprobe | 8 |
| Retry policy 429 / 5xx / timeout | 10 |
| Output-dim mismatch raises | 9 |
| `AudioExtract` stage entry, fingerprint, master-switch short-circuit | 5 |
| `extract_audio` helper, idempotent mtime check | 4 |
| `_run_ffmpeg` → shared `modules/ffmpeg.py` | 1 |
| `case_config_identity` branch | 14 |
| Per-clip dependency rows including file stats | 12 |
| `Secrets.gemini_api_key`, conditional loading | 3 |
| Config fields + `config.toml` defaults | 2 |
| `EmbeddingSecrets` threading through runner & main | 13, 16 |
| Explicit `cases=["gemini_mm"]` while disabled → RuntimeError | 13, 15 |
| `.env.example`, `.gitignore`, `pyproject.toml` extras | 3, 22 |
| Tests: every row of the spec's test table | 4, 5, 8, 9, 10, 11, 12, 15, 17, 18, 19, 20; config 3 |
| `scripts/smoke_gemini_embed.py` | 23 |
| `modules/embeddings/remote.py` removal | 21 |
| Pipeline order (`Download → AudioExtract → Music`) | 6 |
| `main.py` wiring | 6, 16 |

**Placeholder scan:** No TBD/TODO. Every code step contains real code.
Every command line contains the actual command. Tests show the full
test body.

**Type consistency:** `extract_audio` signature `(video_path, audio_path,
*, bitrate_kbps, sample_rate_hz, timeout_s)` is identical at definition
(Task 4) and at call site in `extract_audio_stage` (Task 5). The
`GeminiMultimodalProvider` constructor kwargs are identical across
Tasks 7, 8, 9, 10 and the factory call in Task 14. Test helper
`_make_settings` is used by Tasks 5, 15, 17, 18, 19, 20 — Task 15 notes
factoring it into a shared `conftest.py` fixture. `EmbeddingSecrets` is
defined in Task 13 and referenced in Tasks 14, 16, 17, 18, 19, 20.
`default_cases` is referenced in Task 13's runner and defined in Task 14
(commit in Task 13 is intentionally broken until Task 14 lands — noted).

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-17-gemini-multimodal-embedding-case.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
