# Console Output Design

**Date:** 2026-04-19
**Scope:** Pure display refactor — no logic changes anywhere in the pipeline.

---

## Goal

Replace the current ad-hoc mix of `log()` calls and raw `print()` statements with a unified, beautiful console output using `rich`. The pipeline already has a clear phase structure and consistent progress patterns; this design makes that structure visible.

---

## API — `modules/console.py`

A single new module. Everything else imports display utilities from here. Three public functions:

### `phase(name: str) -> None`
Prints a bold ruled section header at full terminal width. Called once per pipeline phase from `main.py`.

```
─────────────────────────────────────────────
  Music Classification
─────────────────────────────────────────────
```

### `log(scope: str, msg: str, level: str = "info") -> None`
Prints a scoped log line. `level` controls color of the message text only:
- `"info"` — default terminal color
- `"ok"` — green (for done/summary lines)
- `"warn"` — yellow
- `"err"` — red

Scope label is always dim, no per-domain coloring.

```
[classify_music] 150 clips to fingerprint
[classify_music] done — 112 matched, 38 no match        ← green
[classify_music] upload failed: no video for track X    ← red
```

### `progress(total: int, description: str) -> ContextManager[Callable]`
Context manager. Renders a live rich progress bar for the duration of the block. Yields an `advance(n=1, detail="")` callable — callers call it once per item, optionally passing a short detail string shown below the bar.

```
  Transcribing  ━━━━━━━━━━━╸          72/150  48%  0:02:14
  → clip a3f9: "yeah so basically..."
```

Progress bar is white/default — no accent color.

---

## Color Scheme

Colors map to semantic meaning only. Three colors total:

| Meaning | Color |
|---|---|
| success / done summaries (`level="ok"`) | green |
| warnings (`level="warn"`) | yellow |
| errors (`level="err"`) | red |
| everything else | white / dim |

No per-domain colors. No emojis.

---

## Startup Banner

`main.py` prints a thin ruled header before the first phase with the current timestamp and database path. Plain text, no color beyond a rule line.

---

## Changes per file

### New file
- `modules/console.py` — all `rich` imports live here

### Signature-compatible re-export
- `modules/services.py` — `log()` becomes a re-export of `console.log` so existing `from modules.services import log` calls need no import change

### Phase headers only
- `main.py` — add `phase()` call before each pipeline step; add startup banner

### Heavy loops → `progress()` context manager
These files have loops over many items that take significant time. Replace the inner `log()` / `print()` per-item calls with a `progress()` block, using `advance(detail=...)` for per-item detail:

- `modules/speech.py` — transcription loop, translation loop
- `modules/captions.py` — translation loop
- `modules/music.py` — fingerprinting loop; ReccoBeats `on_batch` callbacks
- `modules/download.py` — download loop
- `modules/parse.py` — profile fetch loop (currently uses its own inline print style)
- `modules/embeddings.py` — all four embed loops

### Raw `print()` → `log()` (no progress bar)
Short loops or single-shot prints that don't benefit from a progress bar:

- `modules/cluster_search.py`
- `modules/clustering.py`
- `modules/cluster_validation.py`
- `modules/database.py`
- `modules/finalize.py`
- `modules/visualization.py`

---

## Constraints

- No changes to any module's logic, return values, DB writes, or control flow
- `rich` added to `requirements.txt`
- All display logic confined to `modules/console.py`; no `rich` imports elsewhere
- Output must degrade gracefully if stdout is not a TTY (rich handles this automatically)
