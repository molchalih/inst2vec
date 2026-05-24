"""Thread-safe work broker shared by the in-process local worker and the
HTTP coordinator handlers.

One lock guards the available deque, the lease table, and the per-case
counters. Completions (success or terminal failure) are pushed onto a
``queue.Queue`` that the orchestrator's single-writer drain loop consumes.

A ``Job`` is a plain JSON-serializable dict so it crosses the wire to a
pod unchanged. ``attempts`` lives only in the broker-internal lease/record,
never in the dict sent to a worker.
"""

from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass

from modules.embeddings.cases import CASE_REGISTRY

Job = dict  # {clip_id, case, text, video_key, audio_key, fps, max_frames, remote_eligible}


def make_job(
    *,
    clip_id: int,
    case: str,
    text: str | None,
    video_key: str | None,
    fps: float | None,
    max_frames: int | None,
    remote_eligible: bool,
    audio_key: str | None = None,
) -> Job:
    return {
        "clip_id": clip_id,
        "case": case,
        "text": text,
        "video_key": video_key,
        "audio_key": audio_key,
        "fps": fps,
        "max_frames": max_frames,
        "remote_eligible": remote_eligible,
    }


@dataclass
class Leased:
    lease_id: str
    job: Job


@dataclass
class Completion:
    clip_id: int
    case: str
    blob: bytes | None
    ok: bool


@dataclass
class _Record:
    job: Job
    attempts: int = 0


@dataclass
class _Lease:
    record: _Record
    deadline: float


@dataclass
class _CaseCounter:
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    outstanding: int = 0


def _is_served_remotely(case: str) -> bool:
    return CASE_REGISTRY[case].served_remotely


class JobBroker:
    def __init__(self, *, lease_ttl_s: int, max_attempts: int) -> None:
        self._lease_ttl_s = lease_ttl_s
        self._max_attempts = max_attempts
        self._lock = threading.Lock()
        self._available: list[_Record] = []
        self._leases: dict[str, _Lease] = {}
        self._counters: dict[str, _CaseCounter] = {}
        self._producer_done = False
        self.completions: queue.Queue[Completion] = queue.Queue()

    # ── producer side ──────────────────────────────────────────────────────
    def add(self, job: Job) -> None:
        with self._lock:
            rec = _Record(job=job)
            self._available.append(rec)
            c = self._counters.setdefault(job["case"], _CaseCounter())
            c.total += 1
            c.outstanding += 1

    def producer_done(self) -> None:
        with self._lock:
            self._producer_done = True

    # ── worker side ────────────────────────────────────────────────────────
    def lease(self, *, served_only: bool) -> Leased | None:
        now = time.monotonic()
        with self._lock:
            for i, rec in enumerate(self._available):
                if served_only:
                    if not rec.job["remote_eligible"]:
                        continue
                    if not _is_served_remotely(rec.job["case"]):
                        continue
                del self._available[i]
                lease_id = uuid.uuid4().hex
                self._leases[lease_id] = _Lease(rec, now + self._lease_ttl_s)
                return Leased(lease_id=lease_id, job=rec.job)
            return None

    def complete(self, lease_id: str, blob: bytes) -> None:
        with self._lock:
            lease = self._leases.pop(lease_id, None)
            if lease is None:
                return
            case = lease.record.job["case"]
            c = self._counters[case]
            c.succeeded += 1
            c.outstanding -= 1
            self.completions.put(
                Completion(lease.record.job["clip_id"], case, blob, ok=True)
            )

    def fail(self, lease_id: str, error: str) -> None:
        del error  # logged by the worker; broker only counts
        with self._lock:
            lease = self._leases.pop(lease_id, None)
            if lease is None:
                return
            rec = lease.record
            rec.attempts += 1
            if rec.attempts < self._max_attempts:
                self._available.append(rec)  # requeue; still outstanding
                return
            case = rec.job["case"]
            c = self._counters[case]
            c.failed += 1
            c.outstanding -= 1
            self.completions.put(Completion(rec.job["clip_id"], case, None, ok=False))

    def reap_expired(self) -> None:
        now = time.monotonic()
        with self._lock:
            expired = [lid for lid, lz in self._leases.items() if lz.deadline <= now]
            for lid in expired:
                lease = self._leases.pop(lid)
                rec = lease.record
                rec.attempts += 1
                if rec.attempts < self._max_attempts:
                    self._available.append(rec)
                else:
                    case = rec.job["case"]
                    c = self._counters[case]
                    c.failed += 1
                    c.outstanding -= 1
                    self.completions.put(
                        Completion(rec.job["clip_id"], case, None, ok=False)
                    )

    # ── introspection (for drain loop + /lease drained signal) ──────────────
    def case_outstanding(self, case: str) -> int:
        with self._lock:
            c = self._counters.get(case)
            return c.outstanding if c else 0

    def case_failures(self, case: str) -> int:
        with self._lock:
            c = self._counters.get(case)
            return c.failed if c else 0

    def case_succeeded(self, case: str) -> int:
        with self._lock:
            c = self._counters.get(case)
            return c.succeeded if c else 0

    def all_resolved(self) -> bool:
        with self._lock:
            if not self._producer_done:
                return False
            return all(c.outstanding == 0 for c in self._counters.values())
