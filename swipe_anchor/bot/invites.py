"""File-backed store of admin-created invite links (the ``/start`` deeplink gate).

Admission policy lives with the bot (the backend only mints AccessCodes), so the
invite codes live here too. Persisted as JSON next to the bot's other state so
links survive restarts. The store is tiny and single-writer (one bot process),
so a plain load / mutate / atomic-replace is sufficient.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class Invite:
    code: str
    label: str
    created_at: str
    uses: int
    active: bool


class InviteStore:
    """A JSON-file map of ``code -> {label, created_at, uses, active}``."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._data: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        try:
            raw = self._path.read_text()
        except FileNotFoundError:
            self._data = {}
            return
        try:
            loaded = json.loads(raw)
        except (ValueError, TypeError):
            loaded = None
        self._data = loaded if isinstance(loaded, dict) else {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._data, indent=2, sort_keys=True))
        tmp.replace(self._path)

    def add(self, code: str, label: str = "", *, now: str | None = None) -> Invite:
        """Create (or overwrite) an active invite and persist it."""
        ts = now or datetime.now(UTC).isoformat(timespec="seconds")
        self._data[code] = {
            "label": label,
            "created_at": ts,
            "uses": 0,
            "active": True,
        }
        self._save()
        return self._as_invite(code)

    def is_valid(self, code: str | None) -> bool:
        """True iff ``code`` is a known, still-active invite."""
        if not code:
            return False
        rec = self._data.get(code)
        return bool(rec and rec.get("active"))

    def record_use(self, code: str) -> None:
        """Increment the join counter for an invite (no-op if unknown)."""
        rec = self._data.get(code)
        if rec is None:
            return
        rec["uses"] = int(rec.get("uses", 0)) + 1
        self._save()

    def deactivate(self, code: str) -> bool:
        """Revoke an invite; returns False if it was unknown or already inactive."""
        rec = self._data.get(code)
        if rec is None or not rec.get("active"):
            return False
        rec["active"] = False
        self._save()
        return True

    def active(self) -> list[Invite]:
        """All currently-active invites, ordered by code for stable display."""
        return [
            self._as_invite(code)
            for code, rec in sorted(self._data.items())
            if rec.get("active")
        ]

    def _as_invite(self, code: str) -> Invite:
        rec = self._data[code]
        return Invite(
            code=code,
            label=str(rec.get("label", "")),
            created_at=str(rec.get("created_at", "")),
            uses=int(rec.get("uses", 0)),
            active=bool(rec.get("active")),
        )
