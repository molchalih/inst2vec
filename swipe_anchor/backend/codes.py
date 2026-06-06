"""Manage deeplink access codes (the auth/identity table).

    uv run python -m swipe_anchor.backend.codes add 48DHF63 --note "dasha, gym friend"
    uv run python -m swipe_anchor.backend.codes list
    uv run python -m swipe_anchor.backend.codes disable 48DHF63

The ``note`` is an INTERNAL annotation (who the person is / how you know them) —
it never leaves this DB and is never shown to the annotator.
"""

from __future__ import annotations

import argparse
import os
import sys

from sqlalchemy import select
from sqlalchemy.orm import Session

from swipe_anchor.db import create_app_engine
from swipe_anchor.db.models import AccessCode


def _engine():
    url = os.environ.get("APP_DATABASE_URL") or "sqlite:///data/swipe_anchor.db"
    return create_app_engine(url)


def _add(session: Session, code: str, note: str | None) -> None:
    row = session.get(AccessCode, code)
    if row is None:
        session.add(AccessCode(code=code, note=note, is_active=True))
    else:
        if note is not None:
            row.note = note
        row.is_active = True
    session.commit()
    print(f"saved code={code!r} active=True note={note!r}")


def _set_active(session: Session, code: str, active: bool) -> None:
    row = session.get(AccessCode, code)
    if row is None:
        print(f"unknown code {code!r}", file=sys.stderr)
        raise SystemExit(1)
    row.is_active = active
    session.commit()
    print(f"code={code!r} active={active}")


def _list(session: Session) -> None:
    rows = session.scalars(select(AccessCode).order_by(AccessCode.created_at)).all()
    if not rows:
        print("(no access codes — backend runs open: any non-empty code works)")
        return
    for r in rows:
        flag = "active" if r.is_active else "OFF   "
        print(f"{flag}  {r.code:<16}  {r.note or ''}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="swipe_anchor.backend.codes")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_add = sub.add_parser("add", help="add or update a code")
    p_add.add_argument("code")
    p_add.add_argument("--note", default=None, help="internal note: who this is")

    p_dis = sub.add_parser("disable", help="deactivate a code")
    p_dis.add_argument("code")
    p_en = sub.add_parser("enable", help="reactivate a code")
    p_en.add_argument("code")

    sub.add_parser("list", help="list codes + notes")

    args = parser.parse_args(argv)
    with Session(_engine()) as session:
        if args.cmd == "add":
            _add(session, args.code, args.note)
        elif args.cmd == "disable":
            _set_active(session, args.code, False)
        elif args.cmd == "enable":
            _set_active(session, args.code, True)
        elif args.cmd == "list":
            _list(session)
    return 0


if __name__ == "__main__":
    sys.exit(main())
