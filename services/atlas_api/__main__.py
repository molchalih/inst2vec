"""Run the atlas read API.

    uv run python -m services.atlas_api

Reads the serving DB at ``SERVING_DATABASE_URL`` (required) and serves the
version-6 contract. ``ATLAS_API_TOKEN`` optionally gates every endpoint behind
a Bearer token; ``ATLAS_API_CORS_ORIGIN`` restricts browser access to the Pages
origin. Host/port come from ``ATLAS_API_HOST`` / ``ATLAS_API_PORT``.
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

from core.database import init_serving_db
from services.atlas_api.app import build_app


def _build_from_env():
    load_dotenv()
    serving_url = os.environ.get("SERVING_DATABASE_URL")
    if not serving_url:
        raise SystemExit(
            "SERVING_DATABASE_URL must be set to run the atlas API "
            "(point it at the serving DB the offload script wrote)."
        )
    init_serving_db(serving_url)
    return build_app(
        token=os.environ.get("ATLAS_API_TOKEN", ""),
        cors_origin=os.environ.get("ATLAS_API_CORS_ORIGIN", ""),
    )


def main() -> int:
    import uvicorn

    app = _build_from_env()
    uvicorn.run(
        app,
        host=os.environ.get("ATLAS_API_HOST", "0.0.0.0"),
        port=int(os.environ.get("ATLAS_API_PORT", "8000")),
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
