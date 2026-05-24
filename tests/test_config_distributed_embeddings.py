"""New distributed-embedding config fields load with sane defaults."""

from __future__ import annotations

from core.config import EmbeddingsSettings


def test_distributed_defaults():
    s = EmbeddingsSettings(
        exclude_disqualified_users=True,
        embed_max_length=32768,
        adaptive_max_frames=96,
        adaptive_default_fps=2.0,
    )
    assert s.coordinator_bind_host == "0.0.0.0"
    assert s.coordinator_bind_port > 0
    assert s.lease_ttl_s == 600
    assert s.max_attempts == 3
    assert s.worker_request_timeout_s == 120
    assert s.worker_max_retries == 3


def test_positive_validation_rejects_zero():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        EmbeddingsSettings(
            exclude_disqualified_users=True,
            embed_max_length=32768,
            adaptive_max_frames=96,
            adaptive_default_fps=2.0,
            lease_ttl_s=0,
        )
