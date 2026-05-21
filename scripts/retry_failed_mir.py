"""Reset failed AudioMIR rows so they re-run on the next pipeline pass.

Usage:
    DATABASE_URL=sqlite:///data/inst2vec.db \
      uv run python scripts/retry_failed_mir.py
    DATABASE_URL=... \
      uv run python scripts/retry_failed_mir.py --error maest
"""

from __future__ import annotations

import argparse
import os
import sys

from sqlalchemy import create_engine, make_url
from sqlalchemy.orm import Session

VALID_ERRORS = {"maest", "effnet", "audio_load", "no_audio_file"}


def retry_failed(session: Session, *, error: str | None) -> int:
    """NULL is_mir_extracted / mir_error for failed rows; return count reset.

    Caller is responsible for committing.
    """
    from core.database import AudioMIR

    if error is not None and error not in VALID_ERRORS:
        raise ValueError(
            f"unknown error kind: {error!r} (valid: {sorted(VALID_ERRORS)})"
        )
    q = session.query(AudioMIR).filter(AudioMIR.is_mir_extracted.is_(False))
    if error is not None:
        q = q.filter(AudioMIR.mir_error == error)
    rows = q.all()
    for row in rows:
        row.is_mir_extracted = None
        row.mir_error = None
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--error",
        choices=sorted(VALID_ERRORS),
        default=None,
        help="restrict the reset to a single error kind",
    )
    args = parser.parse_args()

    url = os.environ.get("DATABASE_URL")
    if not url:
        print("ERROR: DATABASE_URL not set", file=sys.stderr)
        return 1
    safe_url = make_url(url).render_as_string(hide_password=True)
    print(
        f"Resetting failed MIR rows in {safe_url}"
        + (f" (error={args.error})" if args.error else "")
        + " ..."
    )
    engine = create_engine(url)
    with Session(engine) as session:
        n = retry_failed(session, error=args.error)
        session.commit()
    print(f"Done. {n} rows reset.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
