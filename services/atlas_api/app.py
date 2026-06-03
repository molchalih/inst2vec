"""Read-only FastAPI app serving the version-7 contract from the serving DB.

An app factory with an optional Bearer-token dependency (D7) and CORS
restricted to the Pages origin.
Handlers reconstruct each payload from the serving rows and return RAW bytes via
``Response`` (NOT ``JSONResponse``) using the exporter's exact serializer, so
endpoint bodies equal the static JSON files byte-for-byte.

Reads ONLY the serving DB — never the pipeline main or identity DBs.
"""

from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from sqlalchemy.orm import Session

from core.database import get_serving_session
from services.atlas_api import reconstruct
from services.atlas_api.serialize import to_bytes

_JSON = "application/json"

SessionFactory = Callable[[], AbstractContextManager[Session]]


def build_app(
    *,
    session_factory: SessionFactory = get_serving_session,
    token: str = "",
    cors_origin: str = "",
) -> FastAPI:
    """Build the atlas read API.

    ``session_factory`` is a context manager yielding a serving ``Session``
    (defaults to the global serving engine; tests inject their own). ``token``,
    when non-empty, gates every data endpoint behind ``Authorization: Bearer``.
    ``cors_origin``, when non-empty, allows that single browser origin.
    """
    app = FastAPI(title="inst2vec atlas api")

    if cors_origin:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[cors_origin],
            allow_methods=["GET"],
            allow_headers=["*"],
        )

    def _auth(authorization: str | None = Header(default=None)) -> None:
        if not token:
            return
        if authorization != f"Bearer {token}":
            raise HTTPException(status_code=401, detail="unauthorized")

    auth_dep = Annotated[None, Depends(_auth)]
    not_found = {404: {"description": "Resource not found"}}

    def _json(payload: dict) -> Response:
        return Response(to_bytes(payload), media_type=_JSON)

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/manifest.json", responses=not_found)
    def manifest(_: auth_dep) -> Response:
        with session_factory() as s:
            try:
                return _json(reconstruct.reconstruct_manifest(s))
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="no runs") from exc

    @app.get("/runs/{run_id}/users.json", responses=not_found)
    def users(run_id: str, _: auth_dep) -> Response:
        with session_factory() as s:
            try:
                return _json(reconstruct.reconstruct_users(s, run_id))
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="run not found") from exc

    @app.get("/runs/{run_id}/clusters.json", responses=not_found)
    def clusters(run_id: str, _: auth_dep) -> Response:
        with session_factory() as s:
            try:
                return _json(reconstruct.reconstruct_clusters(s, run_id))
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="run not found") from exc

    @app.get("/runs/{run_id}/users/{user_id}.json", responses=not_found)
    def user_detail(run_id: str, user_id: int, _: auth_dep) -> Response:
        with session_factory() as s:
            payload = reconstruct.reconstruct_creator_detail(s, run_id, user_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="creator not found")
        return _json(payload)

    @app.get("/runs/{run_id}/clusters-detail.json", responses=not_found)
    def clusters_detail(run_id: str, _: auth_dep) -> Response:
        with session_factory() as s:
            try:
                return _json(reconstruct.reconstruct_clusters_detail_bundle(s, run_id))
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="run not found") from exc

    @app.get("/runs/{run_id}/clusters/{cluster_id}.label.json", responses=not_found)
    def cluster_label(run_id: str, cluster_id: int, _: auth_dep) -> Response:
        with session_factory() as s:
            payload = reconstruct.reconstruct_cluster_label(s, run_id, cluster_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="cluster label not found")
        return _json(payload)

    return app
