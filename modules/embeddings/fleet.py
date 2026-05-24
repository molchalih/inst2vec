"""Lifecycle for the RunPod embedding pod fleet.

``pod_fleet(settings, secrets)`` is a context manager wrapped around the
``StageEmbedder`` block in the clip-embedding stage. When enabled it reconciles
orphan pods from a prior crashed run, then tops up to ``RUNPOD_POD_COUNT`` pods
in the background — re-fetching GPU availability every ``gpu_poll_interval_s``
and deploying any shortfall as stock appears, so the local worker starts at once
and a thin/empty datacenter keeps retrying. It persists pod IDs for crash
recovery and guarantees teardown on exit / atexit / SIGTERM. When disabled it is
a no-op so local-only runs are unchanged.
"""

from __future__ import annotations

import atexit
import json
import os
import signal
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager, suppress

from core.console import log
from core.runpod import PodSpec, RunPodClient


def read_reconcile(path: str) -> list[str]:
    try:
        with open(path) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    return [str(x) for x in data] if isinstance(data, list) else []


def write_reconcile(path: str, ids: list[str]) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(ids, f)
    os.replace(tmp, path)


def fleet_enabled(settings, secrets, count: int) -> bool:
    return bool(
        count > 0
        and getattr(secrets, "runpod_api_key", "")
        and getattr(secrets, "embedder_token", "")
        and getattr(secrets, "coordinator_public_host", "")
        and getattr(settings.storage, "bucket", "")
    )


def pod_count_from_env() -> int:
    try:
        return max(int(os.environ.get("RUNPOD_POD_COUNT", "0")), 0)
    except ValueError:
        return 0


def resolve_gpu_candidates(settings, client) -> tuple[str, ...]:
    """Ordered GPU types the fleet will try, cheapest-first.

    A pinned ``gpu_type_id`` wins outright. Otherwise query the volume's
    datacenter for in-stock types meeting the VRAM/RAM floors and under the
    price cap — giving deploy several fallbacks when stock is thin."""
    rp = settings.runpod
    if rp.gpu_type_id:
        return (rp.gpu_type_id,)
    offers = client.available_gpus(
        data_center_id=rp.data_center_id,
        min_vram_gb=rp.gpu_min_vram_gb,
        min_ram_gb=rp.gpu_min_ram_gb,
    )
    candidates = tuple(
        o.id
        for o in offers
        if o.price_hr is not None and o.price_hr <= rp.gpu_max_price_hr
    )
    log(
        "fleet",
        "SCAN",
        "gpu",
        "ok" if candidates else "none",
        stats={
            "dc": rp.data_center_id,
            "candidates": len(candidates),
            "cap": rp.gpu_max_price_hr,
        },
    )
    return candidates


def build_spec(settings, secrets, *, gpu_type_ids: tuple[str, ...]) -> PodSpec:
    rp = settings.runpod
    return PodSpec(
        image=rp.image,
        gpu_type_ids=gpu_type_ids,
        data_center_id=rp.data_center_id,
        network_volume_id=rp.network_volume_id,
        volume_mount_path=rp.volume_mount_path,
        container_disk_in_gb=rp.container_disk_in_gb,
        min_ram_gb=rp.gpu_min_ram_gb,
        template_id=rp.template_id,
        env={
            "ORCHESTRATOR_HOST": secrets.coordinator_public_host,
            "EMBEDDER_TOKEN": secrets.embedder_token,
            "MODEL_PATH": rp.pod_model_path,
            "VIDEO_ROOT": rp.pod_video_root,
            "HUGGINGFACE_TOKEN": getattr(secrets, "huggingface_token", ""),
        },
    )


class PodFleet:
    def __init__(
        self,
        *,
        client: RunPodClient,
        spec: PodSpec,
        count: int,
        reconcile_path: str,
        refill: Callable[[], tuple[str, ...]] | None = None,
        poll_s: float = 30.0,
    ) -> None:
        self._client = client
        self._spec = spec
        self._count = count
        self._path = reconcile_path
        # refill re-resolves GPU candidates on each attempt; when set, the fleet
        # tops up in the background and re-fetches availability every poll_s
        # until the target is met (so a thin/empty DC keeps retrying).
        self._refill = refill
        self._poll_s = poll_s
        # _ids: pods deployed for THIS run (the only ones counted toward _count).
        # _orphans: prior-run pods the reconcile sweep could not confirm dead —
        # tracked so teardown retries them, but NOT counted, since they point at a
        # stale coordinator host and cannot serve current work.
        self._lock = threading.Lock()
        self._ids: list[str] = []
        self._orphans: list[str] = []
        self._deploying = False
        self._started = False
        self._torn_down = False
        self._prev_sigterm = None
        self._stop: threading.Event | None = None
        self._thread: threading.Thread | None = None

    def _all_ids(self) -> list[str]:
        """Every pod we are responsible for: current-run pods + un-reaped orphans.
        The reconcile file always mirrors this, so neither writer can clobber the
        other's ids. Caller holds ``self._lock``."""
        return self._ids + self._orphans

    def __enter__(self) -> PodFleet:
        orphans = read_reconcile(self._path)
        if orphans:
            log("fleet", "SWEEP", "reconcile", "ok", stats={"orphans": len(orphans)})
            # Reaping a prior run's orphans STOPS billing, so it is unconditional.
            # Any we cannot confirm dead stay tracked (as orphans) for teardown.
            with self._lock:
                self._orphans.extend(self._client.terminate(orphans))
                write_reconcile(self._path, self._all_ids())
        # Arm teardown hooks BEFORE any deploy so a crash mid-deploy still reaps
        # what was created (atexit/SIGTERM within this process; the persisted ids
        # for the next run's reconcile).
        atexit.register(self._teardown)
        with suppress(
            ValueError
        ):  # off main thread (e.g. pytest) — atexit/__exit__ still cover it
            self._prev_sigterm = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGTERM, self._on_signal)
        return self

    def ensure_started(self) -> None:
        """Begin deploying pods; idempotent. Deferred until the stage finds
        remote-leaseable work so a sealed or all-local rerun never deploys a pod
        that would lease nothing and only be billed for the teardown grace."""
        with self._lock:
            if self._started or self._torn_down:
                return
            self._started = True
        if self._refill is None:
            # Eager one-shot deploy from the fixed candidate list (no retry).
            for _ in range(self._count):
                new = self._client.deploy(1, self._spec)
                with self._lock:
                    self._ids.extend(new)
                    write_reconcile(self._path, self._all_ids())
            log("fleet", "SEAL", "deploy", "ok", stats={"pods": len(self._ids)})
        else:
            # Background top-up: the local worker starts immediately while pods
            # join as stock appears. The thread stops on teardown / once full.
            self._stop = threading.Event()
            self._thread = threading.Thread(target=self._topup_loop, daemon=True)
            self._thread.start()
            log("fleet", "SEAL", "deploy", "async", stats={"target": self._count})

    def __exit__(self, *exc) -> bool:
        self._teardown()
        return False

    def stop_scaling(self) -> None:
        """Halt the background top-up so no new pods deploy once the producer is
        done. Idempotent and safe to call before ``_teardown`` (the stage calls
        it ahead of its drain grace). It does NOT reap — running pods drain on
        the coordinator's 410 and final teardown reaps any stragglers, including
        an in-flight deploy this join waits out."""
        if self._stop is not None:
            self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30)

    def _on_signal(self, *_a) -> None:
        self._teardown()
        raise SystemExit(143)

    def _topup_once(self) -> None:
        """Deploy the shortfall using freshly re-fetched GPU candidates. Only
        current-run pods count toward the target; un-reaped orphans do not."""
        with self._lock:
            if self._torn_down:
                return
            shortfall = self._count - len(self._ids)
            if shortfall <= 0:
                return
            self._deploying = True
        new: list[str] = []
        try:
            candidates = self._refill() if self._refill else self._spec.gpu_type_ids
            if candidates:
                self._spec.gpu_type_ids = candidates
                new = self._client.deploy(shortfall, self._spec)
        finally:
            with self._lock:
                # Record the result and clear _deploying together so teardown,
                # which waits for _deploying to fall, sees the new ids before it
                # snapshots — no pod can be created behind teardown's back.
                if new:
                    self._ids.extend(new)
                    write_reconcile(self._path, self._all_ids())
                self._deploying = False

    def _topup_loop(self) -> None:
        assert self._stop is not None
        while not self._stop.is_set():
            try:
                self._topup_once()
            except Exception as exc:  # never let the daemon thread die
                log("fleet", "SCAN", "topup", "WARN", stats={"err": repr(exc)})
            with self._lock:
                full = len(self._ids) >= self._count
            if full:
                log("fleet", "SEAL", "deploy", "ok", stats={"pods": len(self._ids)})
                return
            self._stop.wait(self._poll_s)

    def _teardown(self) -> None:
        with self._lock:
            if self._torn_down:
                return
            self._torn_down = True  # blocks _topup_once from STARTING new deploys
        if self._stop is not None:
            self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30)
        # Reap until the tracked set is stable: a deploy already in flight when we
        # set _torn_down still appends under the lock, so wait it out and snapshot
        # again rather than terminate a stale list and leak the late pod. Persist
        # only ids we could not confirm dead, for the next run's reconcile sweep.
        while True:
            with self._lock:
                to_kill = self._all_ids()
                deploying = self._deploying
            if not to_kill:
                if not deploying:
                    with self._lock:
                        write_reconcile(self._path, [])
                    break
            else:
                failed = self._client.terminate(to_kill)
                confirmed = set(to_kill) - set(failed)
                with self._lock:
                    self._ids = [i for i in self._ids if i not in confirmed]
                    self._orphans = [i for i in self._orphans if i not in confirmed]
                    write_reconcile(self._path, self._all_ids())
                    remaining = set(self._all_ids())
                    deploying = self._deploying
                if remaining <= set(failed) and not deploying:
                    break
            time.sleep(0.05)  # a deploy is mid-flight — let it land, then reap it
        if self._prev_sigterm is not None:
            with suppress(ValueError):
                signal.signal(signal.SIGTERM, self._prev_sigterm)


@contextmanager
def pod_fleet(settings, secrets):
    """Yield an active ``PodFleet`` when enabled, else ``None`` (no-op)."""
    count = pod_count_from_env()
    if not fleet_enabled(settings, secrets, count):
        log("fleet", "SKIP", "fleet", "disabled")
        yield None
        return
    client = RunPodClient(api_key=secrets.runpod_api_key)
    # The background top-up re-resolves candidates each poll, so the spec starts
    # with an empty list; refill fills it (and re-fetches when stock is thin).
    spec = build_spec(settings, secrets, gpu_type_ids=())
    fleet = PodFleet(
        client=client,
        spec=spec,
        count=count,
        reconcile_path=settings.runpod.reconcile_path,
        refill=lambda: resolve_gpu_candidates(settings, client),
        poll_s=settings.runpod.gpu_poll_interval_s,
    )
    with fleet:
        yield fleet
