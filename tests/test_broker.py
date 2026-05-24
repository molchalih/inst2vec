from __future__ import annotations

import time

from modules.embeddings.broker import JobBroker, make_job


def _job(clip_id: int, case: str = "video", remote_eligible: bool = True) -> dict:
    return make_job(
        clip_id=clip_id,
        case=case,
        text=None,
        video_key=f"videos/{clip_id}.mp4",
        fps=2.0,
        max_frames=96,
        remote_eligible=remote_eligible,
    )


def test_lease_returns_job_then_none_when_empty():
    b = JobBroker(lease_ttl_s=600, max_attempts=3)
    b.add(_job(1))
    b.producer_done()
    leased = b.lease(served_only=True)
    assert leased is not None
    assert leased.job["clip_id"] == 1
    # second lease: nothing available, but a lease is outstanding -> None (not drained)
    assert b.lease(served_only=True) is None
    assert not b.all_resolved()


def test_complete_decrements_and_emits_completion():
    b = JobBroker(lease_ttl_s=600, max_attempts=3)
    b.add(_job(1))
    b.producer_done()
    leased = b.lease(served_only=True)
    b.complete(leased.lease_id, b"\x00\x00\x00\x00")
    item = b.completions.get_nowait()
    assert item.clip_id == 1 and item.ok and item.blob == b"\x00\x00\x00\x00"
    assert b.case_outstanding("video") == 0
    assert b.all_resolved()


def test_fail_requeues_until_max_attempts_then_terminal():
    b = JobBroker(lease_ttl_s=600, max_attempts=2)
    b.add(_job(1))
    b.producer_done()
    # attempt 1 -> fail -> requeued
    l1 = b.lease(served_only=True)
    b.fail(l1.lease_id, "boom")
    assert b.case_outstanding("video") == 1
    assert not b.all_resolved()
    # attempt 2 -> fail -> terminal
    l2 = b.lease(served_only=True)
    b.fail(l2.lease_id, "boom")
    item = b.completions.get_nowait()
    assert not item.ok and item.blob is None
    assert b.case_outstanding("video") == 0
    assert b.all_resolved()


def test_served_only_skips_non_remote_eligible():
    b = JobBroker(lease_ttl_s=600, max_attempts=3)
    b.add(_job(1, remote_eligible=False))
    b.producer_done()
    assert b.lease(served_only=True) is None  # pod can't take it
    leased = b.lease(served_only=False)  # local worker can
    assert leased.job["clip_id"] == 1


def test_served_only_skips_local_only_case():
    b = JobBroker(lease_ttl_s=600, max_attempts=3)
    b.add(_job(1, case="maest"))  # served_remotely=False in registry
    b.producer_done()
    assert b.lease(served_only=True) is None
    assert b.lease(served_only=False).job["case"] == "maest"


def test_reap_expired_requeues():
    b = JobBroker(lease_ttl_s=0, max_attempts=3)  # immediate expiry
    b.add(_job(1))
    b.producer_done()
    leased = b.lease(served_only=True)
    assert leased is not None
    time.sleep(0.01)
    b.reap_expired()
    assert b.case_outstanding("video") == 1
    assert b.lease(served_only=True) is not None  # available again


def test_unknown_lease_complete_is_noop():
    b = JobBroker(lease_ttl_s=600, max_attempts=3)
    b.add(_job(1))
    b.producer_done()
    b.lease(served_only=True)
    b.complete("does-not-exist", b"x")  # reaped/duplicate -> ignored
    assert b.completions.empty()
    assert b.case_outstanding("video") == 1


def test_case_failures_tracked():
    b = JobBroker(lease_ttl_s=600, max_attempts=1)
    b.add(_job(1))
    b.producer_done()
    leased = b.lease(served_only=True)
    b.fail(leased.lease_id, "boom")  # max_attempts=1 -> terminal immediately
    b.completions.get_nowait()
    assert b.case_failures("video") == 1
