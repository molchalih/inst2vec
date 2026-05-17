"""Cross-cutting infrastructure shared by every pipeline stage.

Contains the configuration schema, console output helpers, stage-state
idempotence layer, ffmpeg subprocess shim, object-store client, database
schema, and vendored third-party model wrappers. Modules under `modules/`
import these as needed; nothing in `core/` should depend on `modules/`.
"""
