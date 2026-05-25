"""Aggregation + wiring tests for the ONNX MaestProvider.

``_aggregate_patches`` is tested as a pure function. The provider is tested
with ``MaestTokens`` and Essentia ``MonoLoader`` stubbed so no model loads.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from modules.embeddings.maest import _aggregate_patches


def test_aggregate_single_patch():
    """[1, T, 768] -> concat(CLS, DIST, mean(signal tokens)) -> (2304,)."""
    rng = np.random.default_rng(0)
    out = rng.standard_normal((1, 12, 768)).astype(np.float32)
    vec = _aggregate_patches(out)
    assert vec.shape == (2304,)
    expected = np.concatenate(
        [out[0, 0, :], out[0, 1, :], out[0, 2:, :].mean(axis=0)]
    ).astype(np.float32)
    np.testing.assert_allclose(vec, expected, atol=1e-6)


def test_aggregate_multi_patch_mean_pool():
    """N patches -> per-patch (2304,) -> mean across patches."""
    rng = np.random.default_rng(1)
    out = rng.standard_normal((3, 8, 768)).astype(np.float32)
    per_patch = [
        np.concatenate([out[i, 0, :], out[i, 1, :], out[i, 2:, :].mean(axis=0)])
        for i in range(out.shape[0])
    ]
    expected = np.stack(per_patch).mean(axis=0).astype(np.float32)
    np.testing.assert_allclose(_aggregate_patches(out), expected, atol=1e-6)


def _install_stubs(monkeypatch, *, tokens_output, audio):
    """Stub essentia.standard.MonoLoader and core.vendor.maest.MaestTokens."""
    essentia_mod = types.ModuleType("essentia")
    essentia_mod.log = types.SimpleNamespace(warningActive=True, infoActive=True)
    standard_mod = types.ModuleType("essentia.standard")

    class _FakeMonoLoader:
        def __init__(self, *, filename, sampleRate, resampleQuality):
            self.sample_rate = sampleRate

        def __call__(self):
            if isinstance(audio, Exception):
                raise audio
            return audio

    standard_mod.MonoLoader = _FakeMonoLoader
    monkeypatch.setitem(sys.modules, "essentia", essentia_mod)
    monkeypatch.setitem(sys.modules, "essentia.standard", standard_mod)

    import core.vendor.maest as vendor_maest

    class _FakeTokens:
        def __init__(self, onnx_path, **kwargs):
            self.onnx_path = onnx_path

        def tokens(self, audio):
            return tokens_output

    monkeypatch.setattr(vendor_maest, "MaestTokens", _FakeTokens)


def test_provider_wires_tokens_to_aggregation(monkeypatch):
    out = np.ones((2, 5, 768), dtype=np.float32)
    _install_stubs(monkeypatch, tokens_output=out, audio=np.ones(48000, np.float32))

    from modules.embeddings.maest import MaestProvider

    provider = MaestProvider(onnx_path="/fake/maest.onnx", sample_rate=16000)
    result = provider.embed({"audio_path": "/fake.mp3"})
    assert isinstance(result, list) and len(result) == 1
    vec = np.asarray(result[0], dtype=np.float32)
    assert vec.shape == (2304,)
    np.testing.assert_allclose(vec, _aggregate_patches(out), atol=1e-6)


def test_provider_decode_failure_raises(monkeypatch):
    _install_stubs(
        monkeypatch,
        tokens_output=np.ones((1, 3, 768), np.float32),
        audio=RuntimeError("decode failed"),
    )

    from modules.embeddings.maest import MaestProvider

    provider = MaestProvider(onnx_path="/fake/maest.onnx", sample_rate=16000)
    with pytest.raises(RuntimeError, match="decode failed"):
        provider.embed({"audio_path": "/fake.mp3"})


def test_provider_rejects_bad_sample_rate(monkeypatch):
    _install_stubs(
        monkeypatch,
        tokens_output=np.ones((1, 3, 768), np.float32),
        audio=np.ones(10, np.float32),
    )
    from modules.embeddings.maest import MaestProvider

    with pytest.raises(ValueError, match="sample_rate"):
        MaestProvider(onnx_path="/fake/maest.onnx", sample_rate=0)
