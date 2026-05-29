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


def _build_payload(
    spec: EmbeddingCaseSpec, job: dict, video_root: str, audio_root: str | None
) -> dict:
    p = spec.payload_builder(
        None,
        job["text"],
        _resolve_video_path(video_root, job["video_key"]),
        _resolve_audio_path(audio_root, job.get("audio_key")),
        job["fps"],
        job["max_frames"],
    )
    p["clip_id"] = job["clip_id"]
    p["case"] = spec.name
    return p


def _process_group(
    source,
    provider,
    video_root: str,
    audio_root: str | None,
    leases: list[Leased],
) -> None:
    """One same-case batch. Falls back to per-clip ``_process_one`` if any
    clip in the batch needs the token-budget frame-retry path (variable
    max_frames across the batch would break the padded forward), if the
    backend has no ``embed_batch``, or if the batched forward itself
    raises (numerical / OOM / token-mismatch). Per-clip log lines are
    preserved so completion accounting and visible progress don't change.
    """
    if not leases:
        return
    if len(leases) == 1:
        _process_one(source, provider, video_root, audio_root, leases[0])
        return
    case = leases[0].job["case"]
    spec = CASE_REGISTRY[case]
    # Token-budget retry is per-clip; can't easily share a single padded
    # forward across clips that may need to shrink frames. Defer those.
    if spec.apply_video_token_fallback:
        for lz in leases:
            _process_one(source, provider, video_root, audio_root, lz)
        return
    payloads = [_build_payload(spec, lz.job, video_root, audio_root) for lz in leases]
    try:
        outs = provider.embed_batch(payloads)
    except Exception:
        # Any failure (OOM, kernel mismatch, etc.) — recover safely by
        # processing each lease individually; the per-clip path still
        # emits its own log line and marks fail/complete cleanly.
        for lz in leases:
            _process_one(source, provider, video_root, audio_root, lz)
        return
    # Emit per-clip success log lines for symmetry with the single path.
    for lz, vec in zip(leases, outs, strict=True):
        cid = lz.job["clip_id"]
        with item("EXTRACT", f"clip_{cid}"):
            blob = to_bytes(vec)
        source.complete_blob(lz.lease_id, blob)


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
    batch_size: int = 1,
    batch_fill_ms: int = 0,
    log_tag: str = "embed:worker",
) -> None:
    """Drain the job source until it signals ``DRAINED``. Runs up to
    ``inflight`` embeds concurrently. If ``unreachable_exit_s`` is set (pods),
    a lane exits cleanly after that many seconds of continuous ``UNREACHABLE``
    leases — a crash backstop so a dead coordinator stops GPU billing.

    When ``batch_size > 1``, each lane coalesces up to ``batch_size``
    same-case leases (waiting at most ``batch_fill_ms`` ms for the buffer
    to fill) and dispatches them as one padded GPU forward via
    ``provider.embed_batch``. Mixed-case leases are processed per-clip.
    """

    def _coalesce(initial: Leased) -> tuple[list[Leased], bool]:
        # Returns (lease_list, drained_seen). Stops at batch_size, deadline,
        # an empty lease, or a DRAINED signal; UNREACHABLE breaks the buffer
        # so the outer loop can run its idle/exit logic.
        batch = [initial]
        if batch_size <= 1:
            return batch, False
        deadline = time.monotonic() + (batch_fill_ms / 1000.0)
        case = initial.job["case"]
        while len(batch) < batch_size and time.monotonic() < deadline:
            more = source.lease(served_only=served_only)
            if more == DRAINED:
                return batch, True
            if more == UNREACHABLE or more is None:
                # No more immediately-available work; flush what we have.
                break
            # If a different case lands, we'd lose the batched path's
            # case-uniformity invariant. Put it back? The broker doesn't
            # expose return-to-queue, so process the buffer as-is and let
            # the mixed lease ride solo on the next outer iteration.
            if more.job["case"] != case:
                # Process current buffer first, then handle the off-case lease.
                _process_group(source, provider, video_root, audio_root, batch)
                _process_one(source, provider, video_root, audio_root, more)
                return [], False
            batch.append(more)
        return batch, False

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
                batch, drained = _coalesce(leased)
                if batch:
                    _process_group(source, provider, video_root, audio_root, batch)
                if drained:
                    return
        finally:
            _scope_var.reset(token)

    if inflight <= 1:
        lane()
        return
    with ThreadPoolExecutor(max_workers=inflight) as pool:
        futures = [pool.submit(lane) for _ in range(inflight)]
        for f in futures:
            f.result()
