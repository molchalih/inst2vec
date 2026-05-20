"""Regression tests: a failed Qwen3-VL vision-processing step must NOT
silently produce a constant 'NULL' embedding. The exception must
propagate through the provider into the runner's dispatcher, which is
already wired to mark the clip as failed (yield ``(clip_id, None)``)
and prevent the fingerprint seal.

See ``core/vendor/qwen3_vl_embedding.py::_preprocess_inputs`` and
``modules/embeddings/runner.py::_dispatch_embedding_jobs``.
"""

from __future__ import annotations

import logging  # noqa: F401
from unittest.mock import MagicMock, patch

import pytest

from modules.embeddings.runner import _dispatch_embedding_jobs


def test_dispatcher_converts_embed_exception_to_none_blob(monkeypatch):
    """A raising ``embed_fn`` must yield (clip_id, None) and log at warn,
    not propagate. This guarantees that when the vendor stops swallowing
    ``process_vision_info`` errors, the runner's existing machinery will
    catch them cleanly.
    """
    jobs = [{"clip_id": 101}, {"clip_id": 102}, {"clip_id": 103}]

    def embed_fn(job: dict) -> tuple[int, bytes | None]:
        if job["clip_id"] == 102:
            raise RuntimeError("vision processing failed: bad frame")
        return (job["clip_id"], b"\x00" * 16)

    log_calls: list[tuple[tuple, dict]] = []

    def _capture_log(*args, **kwargs) -> None:
        log_calls.append((args, kwargs))

    monkeypatch.setattr("modules.embeddings.runner.log", _capture_log)

    results = list(_dispatch_embedding_jobs(jobs, embed_fn, inflight=1))

    # Dispatcher now yields 3-tuples: (clip_id, blob_or_none, elapsed_or_none)
    assert [(r[0], r[1]) for r in results] == [
        (101, b"\x00" * 16),
        (102, None),
        (103, b"\x00" * 16),
    ]
    # Successful clips get a non-None elapsed; failed clip gets None.
    assert results[0][2] is not None
    assert results[1][2] is None
    assert results[2][2] is not None
    # The dispatcher logs the failure as a structured ERR worker line so
    # operators see the real exception cause.
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
    """The runner's per-clip helper must NOT swallow provider exceptions
    that aren't token-budget mismatches — otherwise the dispatcher's
    per-clip warn log (the only place an operator sees the decode/codec
    cause) never fires for the real Qwen path.
    """
    from types import SimpleNamespace

    from modules.embeddings.cases import CASE_REGISTRY
    from modules.embeddings.runner import _embed_with_token_fallback

    class _RaisingProvider:
        def embed(self, payload):
            raise RuntimeError("process_vision_info: bad codec")

    clip = SimpleNamespace(id=99)

    # video case opts into token-fallback; non-token failure on the first
    # (and only) attempt must propagate, not be converted to None.
    with pytest.raises(RuntimeError, match="bad codec"):
        _embed_with_token_fallback(
            _RaisingProvider(),
            CASE_REGISTRY["video"],
            clip,
            None,
            "/v/99.mp4",
            None,
            1.0,
            32,
        )

    # audio case has no token-fallback path; provider errors must
    # propagate verbatim.
    with pytest.raises(RuntimeError, match="bad codec"):
        _embed_with_token_fallback(
            _RaisingProvider(),
            CASE_REGISTRY["audio"],
            clip,
            "hello",
            None,
            None,
            None,
            None,
        )
