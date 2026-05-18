"""Reclaim orphan identity rows that have no matching main-DB row.

See core/database/identity.py for the identity-first invariant.

Usage:
    uv run python scripts/sweep_identity_orphans.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.config import load_runtime_config
from core.console import log
from core.database import init_db
from core.database.identity import sweep_orphans

SCOPE = "identity-sweep"


def main() -> None:
    _, secrets = load_runtime_config()
    init_db(secrets.database_url, secrets.identity_db_url)
    result = sweep_orphans()
    log(
        SCOPE,
        f"users_swept={result['users_swept']} clips_swept={result['clips_swept']}",
    )


if __name__ == "__main__":
    main()
