"""The embedding worker — identical for the in-process local thread and a
remote pod. A worker leases a job, resolves the video path under its own
``video_root``, builds the case payload, embeds with token-budget frame
retry, and reports success/failure back through its job source.

Two job sources:
  * ``LocalJobSource`` — direct ``JobBroker`` calls; vector serialized to a
    float32 blob in-process and pushed to the broker completion queue.
  * ``HttpJobSource`` — POSTs /lease, /complete, /fail to the coordinator;
    treats HTTP 410 as the drain signal.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

from core.console import log
from core.log import _scope_var, item, warn
from modules.embeddings.broker import JobBroker, Leased
from modules.embeddings.cases import CASE_REGISTRY, EmbeddingCaseSpec
from modules.embeddings.sampling import frame_retry_schedule, is_token_mismatch_error
from modules.embeddings.vectors import to_bytes

DRAINED = "drained"
UNREACHABLE = "unreachable"


def embed_with_token_fallback(
    provider,
    spec: EmbeddingCaseSpec,
    *,
    clip_id: int,
    text: str | None,
    video_path: str | None,
    audio_path: str | None,
    fps: float | None,
    max_frames: int | None,
) -> bytes:
    """Embed one payload, retrying with smaller frame caps on a video
    token-budget mismatch for cases that opt in. Returns a float32 blob.
    Non-token errors and the final attempt's error propagate."""

    def _build(cap: int | None) -> dict:
        p = spec.payload_builder(None, text, video_path, audio_path, fps, cap)
        p["clip_id"] = clip_id
        p["case"] = spec.name
        return p

    if not spec.apply_video_token_fallback or max_frames is None:
        out = provider.embed(_build(max_frames))
        return to_bytes(out[0])

    caps = frame_retry_schedule(max_frames)
    for idx, cap in enumerate(caps):
        try:
            out = provider.embed(_build(cap))
        except Exception as e:
            if is_token_mismatch_error(e) and idx < len(caps) - 1:
                continue
            raise
        return to_bytes(out[0])
    raise RuntimeError("frame_retry_schedule exhausted")  # unreachable: caps non-empty


# ── job sources ────────────────────────────────────────────────────────────


class LocalJobSource:
    def __init__(self, broker: JobBroker) -> None:
        self._broker = broker

    def lease(self, *, served_only: bool):
        leased = self._broker.lease(served_only=served_only)
        if leased is not None:
            return leased
        return DRAINED if self._broker.all_resolved() else None

    def complete_blob(self, lease_id: str, blob: bytes) -> None:
        self._broker.complete(lease_id, blob)

    def fail(self, lease_id: str, error: str) -> None:
        self._broker.fail(lease_id, error)


class HttpJobSource:
    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        timeout_s: int,
        max_retries: int,
        _client: httpx.Client | None = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {token}"}
        self._timeout = timeout_s
        self._max_retries = max_retries
        self._client = _client or httpx.Client(timeout=timeout_s)

    def _post(self, path: str, body: dict) -> httpx.Response:
        attempts = 0
        while True:
            attempts += 1
            try:
                resp = self._client.post(
                    f"{self._base}{path}",
                    headers=self._headers,
                    json=body,
                    timeout=self._timeout,
                )
            except (httpx.TimeoutException, httpx.TransportError):
                if attempts > self._max_retries:
                    raise
                time.sleep(min(2 ** (attempts - 1), 30))
                continue
            if resp.status_code >= 500 and attempts <= self._max_retries:
                time.sleep(min(2 ** (attempts - 1), 30))
                continue
            return resp

    def lease(self, *, served_only: bool):
        try:
            resp = self._post("/lease", {"served_only": served_only})
        except (httpx.TimeoutException, httpx.TransportError):
            return UNREACHABLE
        if resp.status_code == 410:
            return DRAINED
        if resp.status_code == 204:
            return None
        if resp.status_code != 200:
            raise RuntimeError(
                f"coordinator /lease failed: {resp.status_code} {resp.text}"
            )
        data = resp.json()
        return Leased(lease_id=data["lease_id"], job=data["job"])

    def complete(self, lease_id: str, vector) -> None:
        resp = self._post(
            "/complete",
            {"lease_id": lease_id, "embedding": [float(x) for x in vector]},
        )
        if not resp.is_success:
            log(
                "embed:pod",
                "EXTRACT",
                f"lease_{lease_id}",
                "WARN",
                stats={"complete_status": resp.status_code},
            )

    def complete_blob(self, lease_id: str, blob: bytes) -> None:
        import numpy as np

        self.complete(lease_id, np.frombuffer(blob, dtype="<f4").tolist())

    def fail(self, lease_id: str, error: str) -> None:
        resp = self._post("/fail", {"lease_id": lease_id, "error": error})
        if not resp.is_success:
            log(
                "embed:pod",
                "EXTRACT",
                f"lease_{lease_id}",
                "WARN",
                stats={"fail_status": resp.status_code},
            )


# ── worker loop ──────────────────────────────────────────────────────────────


def _safe_join(root: str, key: str) -> str:
    # Jobs carry bare filenames ("{clip_id}.mp4"/".mp3") resolved against the
    # worker's own root. A pod ingests jobs over HTTP, so reject anything with
    # path separators, a parent ref, or a leading slash before joining: a
    # malformed/compromised coordinator must not be able to redirect a worker
    # outside its media root.
    if key != os.path.basename(key) or key in (".", ".."):
        raise ValueError(f"unsafe media key (must be a bare filename): {key!r}")
    return os.path.join(root, key)


def _resolve_video_path(video_root: str, video_key: str | None) -> str | None:
    # video_key is a bare filename ("{clip_id}.mp4") relative to the worker's
    # own video_root (local data dir or the pod's mounted volume). Returned
    # absolute: qwen-vl-utils forms a "file://" URI from the path, and a
    # relative root yields the malformed "file://data/..." (host=data) URI that
    # torchcodec cannot open. abspath also matches build_jobs_for_case's
    # existence check, which probes the absolute path.
    if video_key is None:
        return None
    return os.path.abspath(_safe_join(video_root, video_key))


def _resolve_audio_path(audio_root: str | None, audio_key: str | None) -> str | None:
    if audio_key is None or audio_root is None:
        return None
    return os.path.abspath(_safe_join(audio_root, audio_key))


def _process_one(
    source,
    provider,
    video_root: str,
    audio_root: str | None,
    leased: Leased,
) -> None:
    job = leased.job
    cid = job["clip_id"]
    with item("EXTRACT", f"clip_{cid}") as t:
        spec = CASE_REGISTRY[job["case"]]
        blob = embed_with_token_fallback(
            provider,
            spec,
            clip_id=cid,
            text=job["text"],
            video_path=_resolve_video_path(video_root, job["video_key"]),
            audio_path=_resolve_audio_path(audio_root, job.get("audio_key")),
            fps=job["fps"],
            max_frames=job["max_frames"],
        )
    if t.failed:
        source.fail(leased.lease_id, repr(t.exc))
        return
    source.complete_blob(leased.lease_id, blob)


def run_worker(
    source,
    *,
    provider,
    video_root: str,
    audio_root: str | None = None,
    inflight: int,
    served_only: bool,
    poll_idle_s: float = 0.5,
    unreachable_exit_s: float | None = None,
    log_tag: str = "embed:worker",
) -> None:
    """Drain the job source until it signals ``DRAINED``. Runs up to
    ``inflight`` embeds concurrently. If ``unreachable_exit_s`` is set (pods),
    a lane exits cleanly after that many seconds of continuous ``UNREACHABLE``
    leases — a crash backstop so a dead coordinator stops GPU billing."""

    def lane() -> None:
        token = _scope_var.set(log_tag)
        try:
            unreachable_since: float | None = None
            while True:
                leased = source.lease(served_only=served_only)
                if leased == DRAINED:
                    return
                if leased == UNREACHABLE:
                    now = time.monotonic()
                    if unreachable_since is None:
                        unreachable_since = now
                    elif (
                        unreachable_exit_s is not None
                        and now - unreachable_since > unreachable_exit_s
                    ):
                        warn("SCAN", "coordinator", stats={"status": "unreachable"})
                        return
                    time.sleep(poll_idle_s)
                    continue
                unreachable_since = None
                if leased is None:
                    time.sleep(poll_idle_s)
                    continue
                _process_one(source, provider, video_root, audio_root, leased)
        finally:
            _scope_var.reset(token)

    if inflight <= 1:
        lane()
        return
    with ThreadPoolExecutor(max_workers=inflight) as pool:
        futures = [pool.submit(lane) for _ in range(inflight)]
        for f in futures:
            f.result()
