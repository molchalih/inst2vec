"""Regression: per-clip EMB lines include wall-time stat."""

from __future__ import annotations

import time
from unittest.mock import patch


def test_dispatch_yields_clip_id_blob_and_time() -> None:
    from modules.embeddings.runner import _dispatch_embedding_jobs

    def _slow_embed(job):
        time.sleep(0.01)
        return job["clip_id"], b"\x00\x00\x00\x00"

    # The dispatcher must yield 3-tuples with a non-None elapsed time.
    jobs = [{"clip_id": 1}, {"clip_id": 2}]
    out = list(_dispatch_embedding_jobs(jobs, _slow_embed, inflight=1))
    assert len(out) == 2
    for clip_id, blob, elapsed in out:
        assert clip_id in (1, 2)
        assert blob == b"\x00\x00\x00\x00"
        assert elapsed is not None
        assert elapsed >= 0.005  # roughly 10ms minus jitter


def test_dispatch_failure_yields_none_blob_and_none_time() -> None:
    from modules.embeddings.runner import _dispatch_embedding_jobs

    def _explode(job):
        raise RuntimeError("boom")

    jobs = [{"clip_id": 7}]
    with patch("modules.embeddings.runner.log") as m_log:
        out = list(_dispatch_embedding_jobs(jobs, _explode, inflight=1))
    assert out == [(7, None, None)]
    m_log.assert_called_once()
