"""Regression tests: a failed Qwen3-VL vision-processing step must NOT
silently produce a constant 'NULL' embedding. The exception must
propagate through the provider into the worker, which marks the clip as
failed (``source.fail``) so the broker counts it and the runner refuses
to seal.

See ``core/vendor/qwen3_vl_embedding.py::_preprocess_inputs`` and
``modules/embeddings/worker.py::_process_one``.
"""

from __future__ import annotations

import logging  # noqa: F401
from unittest.mock import MagicMock, patch

import pytest

from modules.embeddings.broker import JobBroker, make_job
from modules.embeddings.worker import (
    LocalJobSource,
    embed_with_token_fallback,
    run_worker,
)


def test_worker_converts_embed_exception_to_failure(monkeypatch):
    """A provider that raises must be reported as a terminal broker failure
    (not propagate, not silently succeed) and logged as a structured ERR
    line. This guarantees that when the vendor stops swallowing
    ``process_vision_info`` errors, the worker's machinery catches them.
    """

    class _FlakyProvider:
        def embed(self, payload):
            if payload["clip_id"] == 102:
                raise RuntimeError("vision processing failed: bad frame")
            return [[1.0, 2.0]]

    b = JobBroker(lease_ttl_s=600, max_attempts=1)
    for cid in (101, 102, 103):
        b.add(
            make_job(
                clip_id=cid,
                case="audio",
                text="x",
                video_key=None,
                fps=None,
                max_frames=None,
                remote_eligible=False,
            )
        )
    b.producer_done()

    log_calls: list[tuple[tuple, dict]] = []

    def _capture_log(*args, **kwargs) -> None:
        log_calls.append((args, kwargs))

    monkeypatch.setattr("modules.embeddings.worker.log", _capture_log)

    run_worker(
        LocalJobSource(b),
        provider=_FlakyProvider(),
        video_root="/x",
        inflight=1,
        served_only=False,
        poll_idle_s=0.01,
    )

    got = {}
    while not b.completions.empty():
        c = b.completions.get_nowait()
        got[c.clip_id] = c.ok
    assert got == {101: True, 102: False, 103: True}
    # The worker logs the failure as a structured ERR line so operators see
    # the real exception cause.
    assert any(
        len(args) >= 4
        and args[1] == "EMB"
        and args[2] == "clip_102"
        and args[3] == "ERR"
        and "vision processing failed" in str(kwargs.get("stats", {}).get("err", ""))
        for args, kwargs in log_calls
    )


def _make_embedder_without_loading_model():
    """Build a ``Qwen3VLEmbedder`` instance without contacting HuggingFace.

    The real ``__init__`` calls ``Qwen3VLForEmbedding.from_pretrained`` and
    ``Qwen3VLProcessor.from_pretrained``. We patch both factories with
    ``MagicMock`` chains so construction returns instantly.
    """
    from core.vendor import qwen3_vl_embedding as vendor

    with (
        patch.object(vendor, "Qwen3VLForEmbedding") as mock_model_cls,
        patch.object(vendor, "Qwen3VLProcessor") as mock_proc_cls,
    ):
        mock_model_cls.from_pretrained.return_value.to.return_value = MagicMock(
            name="model"
        )
        mock_proc_cls.from_pretrained.return_value = MagicMock(name="processor")
        emb = vendor.Qwen3VLEmbedder(
            model_name_or_path="unused",
            max_length=8,
        )
    return emb


def test_preprocess_inputs_propagates_vision_info_failure():
    """When ``process_vision_info`` raises, ``_preprocess_inputs`` must
    re-raise instead of substituting a synthetic 'NULL' conversation.

    Regression for the silent-failure bug: the upstream Qwen3-VL reference
    snippet swallows the exception and produces a constant embedding,
    which in a batch pipeline corrupts downstream clustering.
    """
    from core.vendor import qwen3_vl_embedding as vendor

    emb = _make_embedder_without_loading_model()
    conversations = [
        [
            {
                "role": "user",
                "content": [{"type": "video", "video": "file:///nonexistent.mp4"}],
            }
        ]
    ]

    with (
        patch.object(
            vendor,
            "process_vision_info",
            side_effect=RuntimeError("cannot read video: bad codec"),
        ),
        pytest.raises(RuntimeError, match="cannot read video"),
    ):
        emb._preprocess_inputs(conversations)


def test_embed_with_token_fallback_propagates_non_token_errors():
    """The worker's per-clip helper must NOT swallow provider exceptions
    that aren't token-budget mismatches — otherwise the worker's per-clip
    ERR log (the only place an operator sees the decode/codec cause) never
    fires for the real Qwen path.
    """
    from modules.embeddings.cases import CASE_REGISTRY

    class _RaisingProvider:
        def embed(self, payload):
            raise RuntimeError("process_vision_info: bad codec")

    # video case opts into token-fallback; non-token failure on the first
    # (and only) attempt must propagate, not be converted to None.
    with pytest.raises(RuntimeError, match="bad codec"):
        embed_with_token_fallback(
            _RaisingProvider(),
            CASE_REGISTRY["video"],
            clip_id=99,
            text=None,
            video_path="/v/99.mp4",
            audio_path=None,
            fps=1.0,
            max_frames=32,
        )

    # audio case has no token-fallback path; provider errors must
    # propagate verbatim.
    with pytest.raises(RuntimeError, match="bad codec"):
        embed_with_token_fallback(
            _RaisingProvider(),
            CASE_REGISTRY["audio"],
            clip_id=99,
            text="hello",
            video_path=None,
            audio_path=None,
            fps=None,
            max_frames=None,
        )
