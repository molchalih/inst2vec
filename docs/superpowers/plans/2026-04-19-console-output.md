# Console Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace all ad-hoc `print()` / `log()` calls across the pipeline with a unified, beautiful rich-powered console output — phase headers, colored log levels, and live progress bars — without touching any pipeline logic.

**Architecture:** A new `modules/console.py` owns all `rich` imports and exposes three functions: `phase()`, `log()`, and `progress()`. `services.py` re-exports `log` so existing callers need no import changes. All heavy loops are wrapped in `progress()` context managers. Phase boundaries are announced from `main.py` with `phase()`. Modules that only had raw `print()` get `log()` replacements.

**Tech Stack:** Python, `rich` (new dependency)

---

## File Map

| File | Change |
|---|---|
| `modules/console.py` | **Create** — all rich display logic |
| `requirements.txt` | Add `rich` |
| `modules/services.py` | Re-export `log` from `console` |
| `main.py` | Add `startup()` + `phase()` calls |
| `modules/parse.py` | `progress()` + `log()` |
| `modules/download.py` | `progress()` + `log()` |
| `modules/music.py` | `progress()` on fingerprint + upload loops |
| `modules/speech.py` | `progress()` on transcription + translation loops |
| `modules/captions.py` | `progress()` on detection + translation loops |
| `modules/embeddings.py` | `progress()` on all four embed loops |
| `modules/cluster_search.py` | `print()` → `log()` |
| `modules/clustering.py` | `print()` → `log()` |
| `modules/cluster_validation.py` | `print()` → `log()` |
| `modules/database.py` | `print()` → `log()` |
| `modules/finalize.py` | Already uses `log()` from services — no change needed |
| `modules/visualization.py` | `print()` → `log()` |

---

### Task 1: Create `modules/console.py` and add `rich` to requirements

**Files:**
- Create: `modules/console.py`
- Create: `tests/test_console.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_console.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import io
from modules.console import log, phase, progress, startup


def test_log_info_does_not_raise():
    log("test", "hello world")


def test_log_all_levels_do_not_raise():
    for level in ("info", "ok", "warn", "err"):
        log("test", f"level={level}", level=level)


def test_phase_does_not_raise():
    phase("Test Phase")


def test_startup_does_not_raise():
    startup("data/inst2vec.db")


def test_progress_advances_to_completion():
    with progress(3, "Testing") as advance:
        advance(detail="item 1")
        advance(detail="item 2")
        advance()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/molchalin/Projects/Code/inst2vec && source .venv/bin/activate && pytest tests/test_console.py -v
```

Expected: `ModuleNotFoundError: No module named 'modules.console'`

- [ ] **Step 3: Add `rich` to `requirements.txt`**

Open `requirements.txt` and add `rich` as a new line (top of file or with other deps — order doesn't matter).

- [ ] **Step 4: Install `rich`**

```bash
pip install rich
```

- [ ] **Step 5: Create `modules/console.py`**

```python
"""Unified console output for the inst2vec pipeline."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from typing import Callable, Generator, Literal

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.text import Text

_console = Console()

_LEVEL_STYLES: dict[str, str] = {
    "ok": "green",
    "warn": "yellow",
    "err": "red",
}


def startup(db_path: str = "") -> None:
    """Print a startup banner with timestamp and optional DB path."""
    _console.rule(style="dim")
    suffix = f"  {db_path}" if db_path else ""
    _console.print(f"  inst2vec  {datetime.now().strftime('%Y-%m-%d %H:%M')}{suffix}", style="bold")
    _console.rule(style="dim")
    _console.print()


def phase(name: str) -> None:
    """Print a bold section header ruling the full terminal width."""
    _console.print()
    _console.rule(name)
    _console.print()


def log(scope: str, msg: str, level: Literal["info", "ok", "warn", "err"] = "info") -> None:
    """Print a scoped log line. level controls the message text color."""
    style = _LEVEL_STYLES.get(level, "")
    line = Text()
    line.append(f"[{scope}]", style="dim")
    line.append(f" {msg}", style=style)
    _console.print(line)


@contextmanager
def progress(
    total: int, description: str
) -> Generator[Callable[[int, str], None], None, None]:
    """Context manager yielding advance(n=1, detail="").

    Renders a live rich progress bar for the duration of the block.
    detail is displayed inline after the bar, overwriting on each call.
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("  {task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TextColumn("[dim]{task.fields[detail]}[/dim]"),
        console=_console,
    ) as p:
        task_id = p.add_task(description, total=total, detail="")

        def advance(n: int = 1, detail: str = "") -> None:
            p.update(task_id, advance=n, detail=f"→ {detail}" if detail else "")

        yield advance
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
pytest tests/test_console.py -v
```

Expected: 5 tests PASSED.

- [ ] **Step 7: Commit**

```bash
git checkout -b feat/rich-console-output
git add modules/console.py tests/test_console.py requirements.txt
git commit -m "feat: add modules/console.py with rich phase/log/progress API"
```

---

### Task 2: Make `services.py` re-export `log` from `console`

**Files:**
- Modify: `modules/services.py`

All existing `from modules.services import log` callers continue to work unchanged after this — the signature `log(scope, msg, level="info")` is backward-compatible with the old `log(scope, msg)`.

- [ ] **Step 1: Update `modules/services.py`**

Replace the existing `log` function (lines 15–17):

```python
# ── logging ───────────────────────────────────────────────────────────────────

from modules.console import log  # noqa: F401 — re-exported for existing callers
```

Remove the old:
```python
def log(scope: str, msg: str) -> None:
    """Single output hook for all pipeline steps. Replace here for richer formatting."""
    print(f"[{scope}] {msg}", flush=True)
```

- [ ] **Step 2: Run all tests to verify nothing broke**

```bash
pytest tests/ -v
```

Expected: all existing tests pass (they don't exercise log output).

- [ ] **Step 3: Commit**

```bash
git add modules/services.py
git commit -m "refactor: services.py re-exports log from console"
```

---

### Task 3: Add startup banner and `phase()` calls in `main.py`

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Replace `main.py` entirely**

```python
import os

from modules.console import phase, startup
from modules.services import log
from modules.database import init_db, load_usernames_from_csv
from modules.parse import fetch_profiles
from modules.download import download_files
from modules.music import classify_music, extract_music_features
from modules.speech import classify_speech, translate_speech, clean_speech
from modules.captions import detect_caption_language, translate_captions, clean_captions
from modules.finalize import finalize_user_dataset
from modules.embeddings import embed_video_clips, embed_sandwich_clips, embed_audio_clips, embed_user_clips
from modules.cluster_search import run_cluster_search
from modules.cluster_validation import validate_clustering
from modules.clustering import cluster_users
from modules.visualization import plot_clusters

startup(os.environ.get("DATABASE_URL", "data/inst2vec.db"))

phase("Database")
init_db()
load_usernames_from_csv()

phase("Profile Parsing")
fetch_profiles()

phase("Dataset Filtering — Pass A")
finalize_user_dataset(pass_name="A")

phase("Download")
download_files()

phase("Music Classification")
classify_music()

phase("Music Feature Extraction")
extract_music_features()

phase("Speech")
classify_speech()
translate_speech()
clean_speech()

phase("Captions")
clean_captions()
detect_caption_language()
translate_captions()

phase("Dataset Filtering — Pass B")
finalize_user_dataset(pass_name="B")

phase("Video Embeddings")
embed_video_clips()
embed_sandwich_clips()
embed_audio_clips()

phase("User Embeddings")
embed_user_clips()

phase("Cluster Search")
run_cluster_search()

phase("Cluster Validation")
best_params = validate_clustering()

phase("Clustering")
for case, params in best_params.items():
    if params is None:
        log("cluster", f"{case}: no valid run — skipping", level="warn")
        continue
    cluster_users(case, **params)

phase("Visualization")
plot_clusters()
```

- [ ] **Step 2: Commit**

```bash
git add main.py
git commit -m "feat: add startup banner and phase headers to main.py"
```

---

### Task 4: Update `modules/parse.py`

**Files:**
- Modify: `modules/parse.py`

- [ ] **Step 1: Update `modules/parse.py`**

Add imports at top of file (after existing imports):
```python
from modules.console import progress
from modules.services import log

SCOPE = "fetch_profiles"
```

Replace `fetch_profiles()` entirely:

```python
def fetch_profiles():
    cl = Client(token=HIKER_TOKEN)
    session = get_session()

    failed_pks = session.query(Download.entity_pk).filter(
        Download.file_type == "profile_pic",
        Download.parse_available.is_(False),
    )
    users = (
        session.query(User)
        .filter(~User.pk.in_(failed_pks), (User.user_disqualified.is_(None)) | (User.user_disqualified == 0))
        .limit(BATCH_SIZE)
        .all()
    )

    parsed = skipped = failed = 0

    if not users:
        session.close()
        return

    log(SCOPE, f"{len(users)} users to process")
    with progress(len(users), "Fetching profiles") as advance:
        for user in users:
            if _is_parsed(user):
                skipped += 1
                advance()
                continue

            try:
                data = cl.user_by_username_v1(user.username)
                info = data.get("user", data)

                user.pk = info["pk"]
                user.full_name = info.get("full_name")
                user.profile_pic_url = info.get("profile_pic_url")
                user.profile_pic_url_hd = info.get("profile_pic_url_hd")
                user.following_count = info.get("following_count")
                user.city_name = info.get("city_name")

                clips_count = _fetch_clips(cl, user, session)
                parsed += 1
                advance(detail=f"{user.username} ({user.full_name}, {clips_count} clips)")

            except (ConnectError, TimeoutException) as e:
                session.merge(Download(entity_pk=user.pk, file_type="profile_pic", parse_available=False))
                failed += 1
                advance(detail=f"{user.username} — network error")

            except Exception as e:
                session.merge(Download(entity_pk=user.pk, file_type="profile_pic", parse_available=False))
                failed += 1
                advance(detail=f"{user.username} — error")

            time.sleep(0.3)

    session.commit()
    session.close()

    total = parsed + skipped + failed
    log(SCOPE, f"done — total: {total}, parsed: {parsed}, skipped: {skipped}, failed: {failed}", level="ok")
```

- [ ] **Step 2: Run all tests**

```bash
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add modules/parse.py
git commit -m "feat: add progress bar to fetch_profiles"
```

---

### Task 5: Update `modules/download.py`

**Files:**
- Modify: `modules/download.py`

- [ ] **Step 1: Update `modules/download.py`**

Add imports at top:
```python
from modules.console import progress
from modules.services import log

SCOPE = "download"
```

Remove the `print()` from `_try_download` (the `ok`/`FAILED` line):

```python
def _try_download(session, entity_pk, file_type, url):
    if session.query(Download).filter_by(entity_pk=entity_pk, file_type=file_type).first():
        return

    if not url:
        session.add(Download(entity_pk=entity_pk, file_type=file_type, success=False, parse_available=False))
        return

    ext = "mp4" if file_type == "video" else "jpg"
    path = os.path.join(DIRS[file_type], f"{entity_pk}.{ext}")

    if os.path.exists(path):
        session.add(Download(entity_pk=entity_pk, file_type=file_type, success=True, parse_available=True))
        return

    ok = _download(url, path)
    session.add(Download(
        entity_pk=entity_pk, file_type=file_type,
        success=ok, parse_available=ok,
    ))
```

Replace `download_files()` entirely:

```python
def download_files():
    for d in DIRS.values():
        os.makedirs(d, exist_ok=True)

    session = get_session()
    done_pks = session.query(Download.entity_pk).filter(Download.file_type == "profile_pic")
    users = (
        session.query(User)
        .filter(~User.pk.in_(done_pks), (User.user_disqualified.is_(None)) | (User.user_disqualified == 0))
        .limit(BATCH_SIZE)
        .all()
    )

    if not users:
        session.close()
        return

    log(SCOPE, f"{len(users)} users to download")
    with progress(len(users), "Downloading") as advance:
        for user in users:
            _try_download(session, user.pk, "profile_pic", user.profile_pic_url)
            for clip in user.clips[:MAX_CLIPS or None]:
                if clip.disqualified == 1:
                    continue
                _try_download(session, clip.pk, "thumbnail", clip.thumbnail_url)
                _try_download(session, clip.pk, "video", clip.video_url)
            session.commit()
            advance(detail=user.username)

    session.close()
```

- [ ] **Step 2: Run all tests**

```bash
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add modules/download.py
git commit -m "feat: add progress bar to download_files"
```

---

### Task 6: Update `modules/music.py`

**Files:**
- Modify: `modules/music.py`

- [ ] **Step 1: Add `progress` import**

At the top of `modules/music.py`, alongside the existing `from modules.services import ReccoBeatsClient, SpotifyClient, log`, add:

```python
from modules.console import progress
```

So the import line becomes:
```python
from modules.console import progress
from modules.services import ReccoBeatsClient, SpotifyClient, log
```

- [ ] **Step 2: Replace the fingerprinting loop in `classify_music()`**

Replace from `log(SCOPE_CLASSIFY, f"{len(clips)} clips to fingerprint")` through `log(SCOPE_CLASSIFY, f"done — {', '.join(parts)}")` with:

```python
    log(SCOPE_CLASSIFY, f"{len(clips)} clips to fingerprint")
    matched = no_match = missing = 0

    with progress(len(clips), "Fingerprinting") as advance:
        for i, clip in enumerate(clips, 1):
            path = VIDEO_DIR / f"{clip.pk}.mp4"
            if not path.exists():
                missing += 1
                advance()
                continue

            clip.music_id = None
            clip.music_confidence = None
            result = _fingerprint(acr, str(path))
            if result:
                artist, track, confidence = result
                music = _get_or_create_music(session, artist, track)
                clip.music_id = music.id
                clip.music_confidence = confidence
                clip.has_music = 1
                matched += 1
                advance(detail=f"{clip.pk}: {artist} – {track} ({confidence:.0%})")
            else:
                clip.has_music = 0
                no_match += 1
                advance()

            if i % COMMIT_EVERY == 0:
                session.commit()

    session.commit()
    session.close()
    parts = [f"{matched} matched", f"{no_match} no match"]
    if missing:
        parts.append(f"{missing} skipped (video not downloaded yet)")
    log(SCOPE_CLASSIFY, f"done — {', '.join(parts)}", level="ok")
```

- [ ] **Step 3: Replace the Spotify loop in `extract_music_features()`**

Replace from `log(SCOPE_FEATURES, f"spotify: resolving {total} tracks")` through `log(SCOPE_FEATURES, f"spotify: done — ...")` with:

```python
            log(SCOPE_FEATURES, f"spotify: resolving {total} tracks")
            found = 0
            with progress(total, "Spotify lookup") as advance:
                for i, row in enumerate(rows, 1):
                    sid = spotify.search_id(row.artist, row.track)
                    row.spotify_id = sid or _NO_MATCH
                    if sid:
                        found += 1
                    if i % COMMIT_EVERY == 0:
                        session.commit()
                    advance(detail=f"{row.artist} – {row.track}")
            session.commit()
            log(SCOPE_FEATURES, f"spotify: done — {found} found, {total - found} no match", level="ok")
```

- [ ] **Step 4: Replace the upload fallback loop in `extract_music_features()`**

Replace from `log(SCOPE_FEATURES, f"upload fallback: {total} tracks without catalog coverage")` through `log(SCOPE_FEATURES, f"upload fallback: done — ...")` with:

```python
            log(SCOPE_FEATURES, f"upload fallback: {total} tracks without catalog coverage")
            enriched = 0
            with progress(total, "Upload fallback") as advance:
                for i, row in enumerate(rows, 1):
                    video = _pick_video(session, row.id)
                    if not video:
                        row.has_features = "none"
                        advance(detail=f"{row.artist} – {row.track} (no video)")
                        continue

                    with tempfile.TemporaryDirectory(prefix="rb-audio-") as tmp:
                        audio = _extract_audio_sample(video, Path(tmp))
                        if not audio:
                            row.has_features = "none"
                            advance(detail=f"{row.artist} – {row.track} (audio extract failed)")
                            continue
                        feats = rb.upload_features(audio)

                    if feats and any(f in feats for f in UPLOAD_FIELDS):
                        for f in UPLOAD_FIELDS:
                            if f in feats:
                                setattr(row, f, feats[f])
                        row.has_features = "yes"
                        enriched += 1
                        advance(detail=f"{row.artist} – {row.track} (ok)", level="ok")
                    else:
                        row.has_features = "none"
                        advance(detail=f"{row.artist} – {row.track} (no features)")

                    if i % COMMIT_EVERY == 0:
                        session.commit()

            session.commit()
            log(SCOPE_FEATURES, f"upload fallback: done — {enriched}/{total} enriched", level="ok")
```

Note: `advance()` does not accept a `level` argument — remove the `level="ok"` from `advance(detail=..., level="ok")`. The correct line is simply `advance(detail=f"{row.artist} – {row.track} (ok)")`.

- [ ] **Step 5: Run all tests**

```bash
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add modules/music.py
git commit -m "feat: add progress bars to classify_music and extract_music_features"
```

---

### Task 7: Update `modules/speech.py`

**Files:**
- Modify: `modules/speech.py`

- [ ] **Step 1: Add `progress` import**

After the existing `from modules.services import log` in `modules/speech.py`, add:

```python
from modules.console import progress
```

- [ ] **Step 2: Replace the transcription loop in `classify_speech()`**

Replace from `log(SCOPE_CLASSIFY, f"{len(clips)} clips to transcribe")` through `log(SCOPE_CLASSIFY, f"done — {', '.join(parts)}")` with:

```python
    log(SCOPE_CLASSIFY, f"{len(clips)} clips to transcribe")
    model: Optional[whisper.Whisper] = None
    has_speech = no_speech = missing = 0

    with progress(len(clips), "Transcribing") as advance:
        for i, clip in enumerate(clips, 1):
            path = f"{VIDEO_DIR}/{clip.pk}.mp4"
            if not os.path.exists(path):
                missing += 1
                advance()
                continue

            if model is None:
                log(SCOPE_CLASSIFY, f"loading {WHISPER_MODEL}…")
                model = whisper.load_model(WHISPER_MODEL)
            try:
                text, language, conf, avg_logprob, compression_ratio = _transcribe(model, path)
            except Exception:
                text, language, conf, avg_logprob, compression_ratio = "", "", 0.0, 0.0, 0.0

            clip.speech_transcription = text
            clip.speech_language = language or None
            clip.speech_confidence = conf if text else None
            clip.speech_avg_logprob = avg_logprob if text else None
            clip.speech_compression_ratio = compression_ratio if text else None

            low_logprob = bool(text) and avg_logprob < LOGPROB_THRESHOLD
            high_compression = bool(text) and compression_ratio > COMPRESSION_THRESHOLD
            hallucination = low_logprob or high_compression
            meaningful = _has_meaningful_speech_text(text)

            if meaningful and not hallucination:
                clip.has_speech = 1
                has_speech += 1
                preview = text[:60] + ("…" if len(text) > 60 else "")
                advance(detail=f'{clip.pk}: "{preview}"')
            else:
                clip.has_speech = 0
                no_speech += 1
                advance()

            if i % COMMIT_EVERY == 0:
                session.commit()

    session.commit()
    session.close()
    parts = [f"{has_speech} with speech", f"{no_speech} silent"]
    if missing:
        parts.append(f"{missing} skipped (video not downloaded yet)")
    log(SCOPE_CLASSIFY, f"done — {', '.join(parts)}", level="ok")
```

- [ ] **Step 3: Replace the translation loop in `translate_speech()`**

Replace from `log(SCOPE_TRANSLATE, f"{total} clips to translate")` through `log(SCOPE_TRANSLATE, f"done — {translated}/{total} translated")` with:

```python
    log(SCOPE_TRANSLATE, f"{total} clips to translate")
    translator = GemmaTranslator(model_id=SPEECH_TRANSLATE_MODEL)
    log(SCOPE_TRANSLATE, f"loading {translator.model_id} on {translator.device}…")
    translated = 0

    with progress(total, "Translating speech") as advance:
        for i, clip in enumerate(clips, 1):
            source = (clip.speech_transcription or "").strip()[:SPEECH_TRANSLATION_MAX_CHARS]
            source_lang = (clip.speech_language or "").strip().replace("_", "-")
            if not source or not source_lang or source_lang.lower().startswith("en"):
                advance()
                continue

            try:
                translation = translator.translate_text(
                    text=source,
                    source_lang_code=source_lang,
                    target_lang_code=SPEECH_TRANSLATE_TARGET_LANG,
                    max_new_tokens=SPEECH_TRANSLATE_MAX_NEW_TOKENS,
                )
                if not translation:
                    advance()
                    continue
                clip.speech_translation = translation
                translated += 1
                src_preview = source[:45] + ("…" if len(source) > 45 else "")
                tr_preview = translation[:45] + ("…" if len(translation) > 45 else "")
                advance(detail=f'{clip.pk}: "{src_preview}" → "{tr_preview}"')
            except Exception:
                advance()
                continue

            if i % COMMIT_EVERY == 0:
                session.commit()

    session.commit()
    session.close()
    log(SCOPE_TRANSLATE, f"done — {translated}/{total} translated", level="ok")
```

- [ ] **Step 4: Run all tests**

```bash
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add modules/speech.py
git commit -m "feat: add progress bars to classify_speech and translate_speech"
```

---

### Task 8: Update `modules/captions.py`

**Files:**
- Modify: `modules/captions.py`

- [ ] **Step 1: Add `progress` import**

After the existing `from modules.services import log` in `modules/captions.py`, add:

```python
from modules.console import progress
```

- [ ] **Step 2: Replace the detection loop in `detect_caption_language()`**

Replace from `log(SCOPE_DETECT, f"{total} captions to detect")` through `log(SCOPE_DETECT, f"done — {detected}/{total} detected")` with:

```python
    log(SCOPE_DETECT, f"{total} captions to detect")
    detector = LanguageDetectorBuilder.from_all_languages().build()
    detected = 0

    with progress(total, "Detecting languages") as advance:
        for i, clip in enumerate(clips, 1):
            text = (clip.caption_text or "").strip()
            if not text:
                advance()
                continue
            lang = detector.detect_language_of(text)
            iso = getattr(lang, "iso_code_639_1", None) if lang else None
            if iso is None:
                advance()
                continue
            clip.caption_language = iso.name.lower()
            detected += 1
            advance(detail=f"{clip.pk}: {clip.caption_language}")

            if i % COMMIT_EVERY == 0:
                session.commit()

    session.commit()
    session.close()
    log(SCOPE_DETECT, f"done — {detected}/{total} detected", level="ok")
```

- [ ] **Step 3: Replace the translation loop in `translate_captions()`**

Replace from `log(SCOPE_TRANSLATE, f"{total} captions to translate")` through `log(SCOPE_TRANSLATE, f"done — {translated}/{total} translated")` with:

```python
    log(SCOPE_TRANSLATE, f"{total} captions to translate")
    translator = GemmaTranslator(model_id=CAPTION_TRANSLATE_MODEL)
    log(SCOPE_TRANSLATE, f"loading {translator.model_id} on {translator.device}…")
    translated = 0

    with progress(total, "Translating captions") as advance:
        for i, clip in enumerate(clips, 1):
            source = (clip.caption_text or "").strip()[:CAPTION_TRANSLATION_MAX_CHARS]
            source_lang = (clip.caption_language or "").strip().replace("_", "-")
            if not source or not source_lang or source_lang.lower().startswith("en"):
                advance()
                continue

            try:
                translation = translator.translate_text(
                    text=source,
                    source_lang_code=source_lang,
                    target_lang_code=CAPTION_TRANSLATE_TARGET_LANG,
                    max_new_tokens=CAPTION_TRANSLATE_MAX_NEW_TOKENS,
                )
                if not translation:
                    advance()
                    continue
                clip.caption_translation = translation
                translated += 1
                src_preview = source[:45] + ("…" if len(source) > 45 else "")
                tr_preview = translation[:45] + ("…" if len(translation) > 45 else "")
                advance(detail=f'{clip.pk}: "{src_preview}" → "{tr_preview}"')
            except Exception:
                advance()
                continue

            if i % COMMIT_EVERY == 0:
                session.commit()

    session.commit()
    session.close()
    log(SCOPE_TRANSLATE, f"done — {translated}/{total} translated", level="ok")
```

- [ ] **Step 4: Run all tests**

```bash
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add modules/captions.py
git commit -m "feat: add progress bars to detect_caption_language and translate_captions"
```

---

### Task 9: Update `modules/embeddings.py`

**Files:**
- Modify: `modules/embeddings.py`

- [ ] **Step 1: Add imports at the top of `modules/embeddings.py`**

After the existing imports, add:

```python
from modules.console import progress
from modules.services import log
```

- [ ] **Step 2: Replace the `embed_video_clips()` loop**

Replace the section from `if not todo:` through `print("[embed:video] done")` with:

```python
        if not todo:
            log("embed:video", "nothing to do")
            return

        log("embed:video", f"{len(todo)} clips to embed ({len(done_video)} already done)")
        from modules.external.qwen3_vl_embedding import Qwen3VLEmbedder
        model = Qwen3VLEmbedder(
            model_name_or_path=MODEL_PATH,
            max_length=EMBED_MAX_LENGTH,
            max_frames=ADAPTIVE_MAX_FRAMES,
            fps=ADAPTIVE_DEFAULT_FPS,
        )

        with progress(len(todo), "Embedding video") as advance:
            for i, clip in enumerate(todo, 1):
                path = _video_path(clip.pk)
                fps, max_frames, duration = _adaptive_video_sampling(path)
                frame_caps = _frame_retry_schedule(max_frames)
                embeddings = None
                last_error: Exception | None = None
                for attempt_idx, frame_cap in enumerate(frame_caps):
                    try:
                        embeddings = model.process([{"video": path, "fps": fps, "max_frames": frame_cap}])
                        break
                    except Exception as e:
                        last_error = e
                        if _is_token_mismatch_error(e) and attempt_idx < len(frame_caps) - 1:
                            continue
                        break
                if embeddings is None:
                    advance(detail=f"✗ {clip.pk}")
                    continue

                video_row = ClipEmbedding(
                    clip_pk=clip.pk,
                    embedding_case="video",
                    embedding=_to_bytes(embeddings[0]),
                )
                session.merge(video_row)
                session.commit()
                advance(detail=f"✓ {clip.pk}")

        log("embed:video", "done", level="ok")
```

- [ ] **Step 3: Replace the `embed_sandwich_clips()` loop**

Replace the section from `if not todo:` through `print("[embed:sandwich] done")` with:

```python
        if not todo:
            log("embed:sandwich", "nothing to do")
            return

        log("embed:sandwich", f"{len(todo)} clips to embed ({len(done_sandwich)} already done)")
        from modules.external.qwen3_vl_embedding import Qwen3VLEmbedder
        model = Qwen3VLEmbedder(
            model_name_or_path=MODEL_PATH,
            max_length=EMBED_MAX_LENGTH,
            max_frames=ADAPTIVE_MAX_FRAMES,
            fps=ADAPTIVE_DEFAULT_FPS,
        )

        with progress(len(todo), "Embedding sandwich") as advance:
            for i, (clip, text) in enumerate(todo, 1):
                path = _video_path(clip.pk)
                fps, max_frames, duration = _adaptive_video_sampling(path)
                frame_caps = _frame_retry_schedule(max_frames)
                embedding = None
                last_error: Exception | None = None
                for attempt_idx, frame_cap in enumerate(frame_caps):
                    try:
                        embeddings = model.process(
                            [{"video": path, "fps": fps, "max_frames": frame_cap, "text": text}]
                        )
                        embedding = embeddings[0]
                        break
                    except Exception as e:
                        last_error = e
                        if _is_token_mismatch_error(e) and attempt_idx < len(frame_caps) - 1:
                            continue
                        break

                if embedding is None:
                    advance(detail=f"✗ {clip.pk}")
                    continue

                sandwich_row = ClipEmbedding(
                    clip_pk=clip.pk,
                    embedding_case="sandwich",
                    embedding=_to_bytes(embedding),
                )
                session.merge(sandwich_row)
                session.commit()
                advance(detail=f"✓ {clip.pk}")

        log("embed:sandwich", "done", level="ok")
```

- [ ] **Step 4: Replace the `embed_audio_clips()` loop**

Replace the section from `if not todo:` through `print("[embed:audio] done")` with:

```python
        if not todo:
            log("embed:audio", "nothing to do")
            return

        log("embed:audio", f"{len(todo)} clips to embed ({len(done_audio)} already done)")
        model = Qwen3VLEmbedder(
            model_name_or_path=MODEL_PATH,
            max_length=EMBED_MAX_LENGTH,
        )

        with progress(len(todo), "Embedding audio") as advance:
            for i, (clip, text) in enumerate(todo, 1):
                try:
                    embeddings = model.process([{"text": text, "instruction": AUDIO_INSTRUCTION}])
                    embedding = embeddings[0]
                except Exception as e:
                    advance(detail=f"✗ {clip.pk}")
                    continue

                audio_row = ClipEmbedding(
                    clip_pk=clip.pk,
                    embedding_case="audio",
                    embedding=_to_bytes(embedding),
                )
                session.merge(audio_row)
                session.commit()
                advance(detail=f"✓ {clip.pk}")

        log("embed:audio", "done", level="ok")
```

- [ ] **Step 5: Replace the `embed_user_clips()` prints**

Replace all `print(f"[embed:user:{case}] ...")` calls with `log(f"embed:user:{case}", ...)`:

```python
            if not rows:
                log(f"embed:user:{case}", "nothing to do")
                continue

            aggregated = _aggregate_user_embeddings(rows)
            log(f"embed:user:{case}", f"{len(aggregated)} users to embed")

            for user_pk, mean_blob in aggregated.items():
                row = UserEmbedding(
                    user_pk=user_pk,
                    embedding_case=case,
                    embedding=mean_blob,
                )
                session.merge(row)
                session.commit()

            log(f"embed:user:{case}", "done", level="ok")
```

- [ ] **Step 6: Run all tests**

```bash
pytest tests/ -v
```

Expected: all tests pass (embeddings tests mock the model, so they don't hit the display code).

- [ ] **Step 7: Commit**

```bash
git add modules/embeddings.py
git commit -m "feat: add progress bars to all embed functions"
```

---

### Task 10: Replace raw `print()` with `log()` in remaining modules

**Files:**
- Modify: `modules/cluster_search.py`
- Modify: `modules/clustering.py`
- Modify: `modules/cluster_validation.py`
- Modify: `modules/database.py`
- Modify: `modules/visualization.py`

- [ ] **Step 1: Update `modules/cluster_search.py`**

Add import after existing imports:
```python
from modules.services import log
```

Replace all `print()` calls:
```python
# line 85: print(f"[cluster_search:{case}] no embeddings — skipping {len(case_combos)} combos")
log(f"cluster_search:{case}", f"no embeddings — skipping {len(case_combos)} combos", level="warn")

# line 103: print(f"[cluster_search:{case}] skipping — {exc}")
log(f"cluster_search:{case}", f"skipping — {exc}", level="warn")

# line 125: print(f"[cluster_search] done — {total_new} new, {total_skipped} skipped")
log("cluster_search", f"done — {total_new} new, {total_skipped} skipped", level="ok")
```

- [ ] **Step 2: Update `modules/clustering.py`**

Add import after existing imports:
```python
from modules.services import log
```

Replace all `print()` calls:
```python
# line 153: print(f"[cluster:{embedding_case}] nothing to do")
log(f"cluster:{embedding_case}", "nothing to do")

# line 156: print(f"[cluster:{embedding_case}] {matrix.shape[0]} users — running UMAP + HDBSCAN")
log(f"cluster:{embedding_case}", f"{matrix.shape[0]} users — running UMAP + HDBSCAN")

# line 160: print(f"[cluster:{embedding_case}] skipping — {exc}")
log(f"cluster:{embedding_case}", f"skipping — {exc}", level="warn")

# lines 179-182: print(f"[cluster:{embedding_case}] {result.n_clusters} clusters, ...")
log(
    f"cluster:{embedding_case}",
    f"{result.n_clusters} clusters, {result.noise_ratio:.1%} noise, sizes: {sizes_str}",
    level="ok",
)
```

- [ ] **Step 3: Update `modules/cluster_validation.py`**

Add import after existing imports:
```python
from modules.services import log
```

Replace all `print()` calls. The pattern is `print(f"[validate:{case}] <message>")` → `log(f"validate:{case}", "<message>")`. Apply level based on meaning:

```python
# filter line
log(f"validate:{case}", f"filter — {n_pass} passed, {len(rows) - n_pass} disqualified")

# score skip (warning — run disqualified)
log(f"validate:{case}", f"score skip id={row.id} — {exc}", level="warn")

# dbcv failed (error — computation failed)
log(f"validate:{case}", f"dbcv failed id={row.id} — disqualifying", level="err")

# scored progress line
log(f"validate:{case}", f"scored {i + 1}/{len(rows)} id={row.id} dbcv={dbcv_str} sil={sil_str}")

# composite done
log(f"validate:{case}", f"composite — updated {len(rows)} rows")

# bootstrap nothing to do
log(f"validate:{case}", "bootstrap — nothing to do")

# bootstrap skip
log(f"validate:{case}", f"bootstrap skip id={row.id} — {exc}", level="warn")

# bootstrap all-failed
log(f"validate:{case}", f"bootstrap all-failed id={row.id} — disqualifying", level="err")

# bootstrap scored line
log(f"validate:{case}", f"bootstrap {i + 1}/{len(rows)} id={row.id} stability={stability:.4f}")

# plateau nothing to do
log(f"validate:{case}", "plateau — nothing to do")

# plateau scored
log(f"validate:{case}", f"plateau — scored {len(top_rows)} top rows")

# override
log(f"validate:{case}", f"override — using run id={row.id} (forced via env var)")

# select no eligible
log(f"validate:{case}", "select — no eligible runs", level="warn")

# select result line (multi-line print → single log)
log(f"validate:{case}", f"select — run id={row.id} ...")  # keep the same message content

# starting / done
log(f"validate:{case}", "starting")
log(f"validate:{case}", "no embeddings — skipping", level="warn")
log(f"validate:{case}", "done", level="ok")
```

Apply these replacements to each `print()` in `cluster_validation.py` matching the pattern above.

- [ ] **Step 4: Update `modules/database.py`**

Find the `print(f"Loaded {loaded} usernames ...")` call (around line 308) and replace with:

```python
from modules.services import log  # add to imports at top of file
```

And replace:
```python
print(f"Loaded {loaded} usernames ({duplicates_in_csv} duplicates in csv, {already_in_db} already in db)")
```
with:
```python
log("database", f"loaded {loaded} usernames ({duplicates_in_csv} duplicates in csv, {already_in_db} already in db)")
```

- [ ] **Step 5: Update `modules/visualization.py`**

Add import:
```python
from modules.services import log
```

Replace:
```python
print(f"[viz] saved {path} ({n_clusters} clusters, {noise} noise points)")
```
with:
```python
log("viz", f"saved {path} ({n_clusters} clusters, {noise} noise points)", level="ok")
```

- [ ] **Step 6: Run all tests**

```bash
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add modules/cluster_search.py modules/clustering.py modules/cluster_validation.py modules/database.py modules/visualization.py
git commit -m "refactor: replace raw print() with log() in remaining modules"
```

---

## Verification

After all tasks are complete, verify the full test suite is green:

```bash
pytest tests/ -v
```

All tests should pass. Since the pipeline requires real credentials and data to run, the visual output can be verified by running the script with `python main.py` in a real environment and observing: startup banner, phase headers, live progress bars on heavy loops, colored done/warn/err summary lines.
