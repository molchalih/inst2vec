"""Stage-scoped HTTP coordinator for distributed clip embedding.

``build_app`` returns a FastAPI app bound to a JobBroker. Endpoints are
trivial and non-blocking -- /complete and /fail only hand work to the
broker (whose completion queue the orchestrator's single-writer drain
loop consumes). ``serve_in_thread`` runs uvicorn in a background thread so
the embeddings stage can start it, drain, and shut it down.
"""

from __future__ import annotations

import threading
import time
from contextlib import asynccontextmanager

import numpy as np
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from modules.embeddings.broker import JobBroker


class LeaseRequest(BaseModel):
    served_only: bool = True


class CompleteRequest(BaseModel):
    lease_id: str
    embedding: list[float]


class FailRequest(BaseModel):
    lease_id: str
    error: str = ""


def build_app(broker: JobBroker, *, token: str, thread_tokens: int = 128) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Sync endpoints run in anyio's threadpool (default 40). With ~50 pods
        # bursting /complete, raise the limit so handlers (all O(1)) don't queue.
        import anyio.to_thread

        anyio.to_thread.current_default_thread_limiter().total_tokens = thread_tokens
        yield

    app = FastAPI(title="inst2vec embedding coordinator", lifespan=lifespan)

    def _auth(authorization: str | None = Header(default=None)) -> None:
        if not token:
            raise HTTPException(status_code=500, detail="coordinator token not set")
        if authorization != f"Bearer {token}":
            raise HTTPException(status_code=401, detail="unauthorized")

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    @app.post("/lease")
    def lease(req: LeaseRequest, _: None = Depends(_auth)):
        leased = broker.lease(served_only=req.served_only)
        if leased is not None:
            return {"lease_id": leased.lease_id, "job": leased.job}
        if broker.all_resolved():
            return JSONResponse(status_code=410, content={"status": "drained"})
        return Response(status_code=204)

    @app.post("/complete")
    def complete(req: CompleteRequest, _: None = Depends(_auth)) -> dict:
        blob = np.asarray(req.embedding, dtype=np.float32).tobytes()
        broker.complete(req.lease_id, blob)
        return {"ok": True}

    @app.post("/fail")
    def fail(req: FailRequest, _: None = Depends(_auth)) -> dict:
        broker.fail(req.lease_id, req.error)
        return {"ok": True}

    return app


class _ServerThread:
    def __init__(self, server: uvicorn.Server) -> None:
        self._server = server
        self._thread = threading.Thread(target=server.run, daemon=True)

    def start(self, *, startup_timeout_s: float = 10.0) -> None:
        self._thread.start()
        deadline = time.monotonic() + startup_timeout_s
        while not self._server.started:
            if not self._thread.is_alive():
                raise RuntimeError("coordinator server thread died on startup")
            if time.monotonic() > deadline:
                raise RuntimeError("timed out waiting for coordinator to start")
            time.sleep(0.01)

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10)


def serve_in_thread(app: FastAPI, *, host: str, port: int) -> _ServerThread:
    config = uvicorn.Config(app, host=host, port=port, log_level="warning", workers=1)
    server = uvicorn.Server(config)
    st = _ServerThread(server)
    st.start()
    return st
