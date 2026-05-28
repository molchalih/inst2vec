"""One-shot data migration: rewrite legacy labels-stage state + backfill label_case.

Pre-multimodal-labels installations stored:
  * ``stage_state`` row at ``(labels, video)`` with the *legacy* hash shape
    (``data_hash`` and ``dependency_hash`` set to ``hash_text("")``, and
    ``config_hash`` derived from a flat ``labels.prompt`` key folded into
    ``stable_subset_payload``);
  * ``clip_labels`` rows where ``label_case`` is NULL or empty (rebuilt with
    a server-default of ``"video"`` by ``scripts/migrate_clip_labels_schema``
    in the same release, but some operators wrote raw NULLs that bypass that
    default).

The labels-stage runtime used to carry two compat helpers — one that rewrote
the legacy ``stage_state`` row into the new fingerprint shape on first run,
and one that backfilled ``label_case`` — so that operators could upgrade
without losing existing visual labels. Removing those helpers requires that
every environment first run this script (or have already executed an
equivalent in-stage migration via the old runtime).

Run on every environment BEFORE deploying a build that has the in-stage
compat helpers removed. Reads ``DATABASE_URL`` from the env (or from
``.env`` next to this script's repo root). Idempotent — re-running on a
migrated DB reports ``no-op`` for both steps.

Usage::

    uv run python scripts/migrate_labels_legacy.py --dry-run   # report only
    uv run python scripts/migrate_labels_legacy.py             # commit
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from sqlalchemy import or_, update
from sqlalchemy.orm import Session


def _load_env() -> None:
    """Minimal ``.env`` loader matching ``scripts/migrate_clip_labels_schema``.

    Avoids a hard dep on ``python-dotenv`` so this one-shot can run on any
    environment that has the project's `.env` checked in next to the repo
    root. Existing env vars take precedence.
    """
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env()

# Imports below depend on ``DATABASE_URL`` / ``IDENTITY_DB_URL`` being in
# the environment by the time ``load_runtime_config`` is invoked. Keep the
# imports above ``_load_env`` minimal so the .env load can populate the
# environment before any project module touches it.

from core import fingerprint as fp  # noqa: E402
from core.config import LabelsSettings, load_runtime_config  # noqa: E402
from core.database import ClipLabel, StageState, init_db  # noqa: E402
from core.database.engine import get_engine  # noqa: E402
from core.fingerprint import stable_subset_payload  # noqa: E402
from modules.labels.cases import VIDEO_CASE  # noqa: E402
from modules.labels.clip_pass import (  # noqa: E402
    _data_hash_for_video,
    _dependency_hash,
)
from modules.labels.state import (  # noqa: E402
    STAGE_LABELS,
    clip_labels_config_payload,
    clip_scope_for,
)

# ── embedded legacy config-payload computation ──────────────────────────────
#
# Reproduces ``modules.labels.state.legacy_video_clip_labels_config_payload``
# verbatim so this script keeps working after the in-stage compat helpers are
# removed. The legacy ``LabelsSettings`` snapshot carried a flat ``prompt``
# key; the post-migration shape replaces it with a per-case ``case_prompts``
# sub-table, so we wrap ``LabelsSettings`` in a tiny attribute-access shim
# that exposes ``prompt`` as ``case_prompts.get("video", "")`` to make
# ``stable_subset_payload`` reproduce the legacy bytes exactly.

_LEGACY_VIDEO_LABELS_CONFIG_FIELDS: tuple[str, ...] = (
    "prompt",
    "model_path",
    "frame_count",
    "max_new_tokens",
    "generation_seed",
    "min_tags_per_kind",
    "max_tags_per_kind",
    "min_tag_chars",
    "max_tag_chars",
    "min_sentence_chars",
    "max_sentence_chars",
)


class _LegacyLabelsView:
    def __init__(self, labels: LabelsSettings) -> None:
        self._labels = labels

    def __getattr__(self, name: str) -> object:
        if name == "prompt":
            return self._labels.case_prompts.get("video", "")
        return getattr(self._labels, name)


def _legacy_video_clip_labels_config_payload(labels: LabelsSettings) -> str:
    return stable_subset_payload(
        _LegacyLabelsView(labels), _LEGACY_VIDEO_LABELS_CONFIG_FIELDS
    )


# ── migration steps ─────────────────────────────────────────────────────────


def backfill_label_case_nulls(session: Session) -> int:
    """Set ``label_case='video'`` on rows where it's NULL or empty.

    Idempotent. Caller commits.
    """
    stmt = (
        update(ClipLabel)
        .where(or_(ClipLabel.label_case.is_(None), ClipLabel.label_case == ""))
        .values(label_case="video")
    )
    result = session.execute(stmt)
    return int(result.rowcount or 0)


def adopt_legacy_video_fingerprint(session: Session, *, settings) -> bool:
    """Rewrite the legacy ``stage_state`` row into the new fingerprint shape.

    Returns True iff a row was rewritten. No-op when:
      * no row exists at ``(labels, video)`` (first ever run);
      * the row is already in the new shape (migrated previously);
      * the row is in some other legacy variant (let normal drift wipe it).

    Caller commits.
    """
    stored = session.get(StageState, (STAGE_LABELS, clip_scope_for("video")))
    if stored is None:
        return False

    empty = fp.hash_text("")
    legacy_config_hash = fp.hash_text(
        _legacy_video_clip_labels_config_payload(settings.labels)
    )
    if (
        stored.data_hash != empty
        or stored.dependency_hash != empty
        or stored.config_hash != legacy_config_hash
    ):
        return False

    # Inline the video-case fingerprint compute (mirrors
    # ``modules.labels.clip_pass._current_fingerprint`` for the ``video``
    # case). Calling the lower-level helpers directly decouples this
    # one-shot from internal-API drift between the pre- and post-refactor
    # ``_current_fingerprint`` signatures.
    new_fingerprint = fp.Fingerprint(
        data=_data_hash_for_video(session, settings),
        config=fp.hash_text(clip_labels_config_payload(settings.labels, case="video")),
        dependency=_dependency_hash(session, spec=VIDEO_CASE),
    )
    fp.mark_complete(session, STAGE_LABELS, clip_scope_for("video"), new_fingerprint)
    return True


def run_migration(*, dry_run: bool) -> tuple[int, bool]:
    """Open a session, run both migration steps, optionally commit.

    Returns ``(nulls_backfilled, stage_state_rewritten)``.
    """
    settings, secrets = load_runtime_config()
    init_db(secrets.database_url, secrets.identity_db_url)

    with Session(get_engine()) as session:
        nulls_backfilled = backfill_label_case_nulls(session)
        stage_state_rewritten = adopt_legacy_video_fingerprint(
            session, settings=settings
        )
        if dry_run:
            session.rollback()
        else:
            session.commit()
    return nulls_backfilled, stage_state_rewritten


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "One-shot labels-stage legacy migration. Run once per "
            "environment before deploying a build with the in-stage compat "
            "helpers removed."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute counts but don't commit any changes.",
    )
    args = parser.parse_args(argv)

    nulls_backfilled, stage_state_rewritten = run_migration(dry_run=args.dry_run)

    print(f"label_case NULL backfill   : {nulls_backfilled} rows")
    print(f"legacy stage_state rewrite : {'yes' if stage_state_rewritten else 'no-op'}")
    print("(dry-run; rolled back)" if args.dry_run else "(committed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
