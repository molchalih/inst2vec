"""Aggregation / tiling tests for MaestProvider.

Stubs ``essentia.standard.TensorflowPredictMAEST`` and ``MonoLoader``
so no Essentia model is loaded.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest


def _install_essentia_stub(
    monkeypatch,
    *,
    predict_output: np.ndarray,
    audio: np.ndarray,
):
    """Install a minimal ``essentia`` / ``essentia.standard`` stub.

    ``TensorflowPredictMAEST(...)`` returns a callable that yields
    ``predict_output``. ``MonoLoader(...)`` returns a callable that
    yields ``audio``.
    """
    essentia_mod = types.ModuleType("essentia")
    essentia_mod.log = types.SimpleNamespace(warningActive=True, infoActive=True)
    standard_mod = types.ModuleType("essentia.standard")

    class _FakePredictor:
        def __init__(self, *, graphFilename, input, output):
            self.graph = graphFilename
            self.input = input
            self.output = output

        def __call__(self, _audio):
            return predict_output

    class _FakeMonoLoader:
        def __init__(self, *, filename, sampleRate, resampleQuality):
            self.filename = filename
            self.sample_rate = sampleRate

        def __call__(self):
            return audio

    standard_mod.TensorflowPredictMAEST = _FakePredictor
    standard_mod.MonoLoader = _FakeMonoLoader

    monkeypatch.setitem(sys.modules, "essentia", essentia_mod)
    monkeypatch.setitem(sys.modules, "essentia.standard", standard_mod)


def test_maest_single_patch_aggregation(monkeypatch):
    """Single patch: concat(CLS, DIST, mean(signal_tokens)) over 768-d."""
    rng = np.random.default_rng(0)
    out = rng.standard_normal((1, 1, 12, 768)).astype(np.float32)
    audio = np.ones(48000, dtype=np.float32)

    _install_essentia_stub(monkeypatch, predict_output=out, audio=audio)

    from modules.embeddings.maest import MaestProvider

    provider = MaestProvider(
        checkpoint_path="/fake/maest.pb",
        input_op="serving_default_melspectrogram",
        sample_rate=16000,
        min_samples=480000,
    )
    result = provider.embed({"audio_path": "/fake.mp3"})
    assert isinstance(result, list) and len(result) == 1
    vec = np.asarray(result[0], dtype=np.float32)
    assert vec.shape == (2304,)

    expected = np.concatenate(
        [out[0, 0, 0, :], out[0, 0, 1, :], out[0, 0, 2:, :].mean(axis=0)]
    )
    np.testing.assert_allclose(vec, expected.astype(np.float32), atol=1e-6)


def test_maest_multi_patch_mean_pool(monkeypatch):
    """N patches → per-patch (2304,) → mean across patches."""
    rng = np.random.default_rng(1)
    out = rng.standard_normal((2, 1, 8, 768)).astype(np.float32)
    audio = np.ones(960000, dtype=np.float32)  # > min_samples → no tiling

    _install_essentia_stub(monkeypatch, predict_output=out, audio=audio)

    from modules.embeddings.maest import MaestProvider

    provider = MaestProvider(
        checkpoint_path="/fake/maest.pb",
        input_op="serving_default_melspectrogram",
        sample_rate=16000,
        min_samples=480000,
    )
    vec = np.asarray(provider.embed({"audio_path": "/fake.mp3"})[0], dtype=np.float32)

    per_patch = []
    for i in range(out.shape[0]):
        per_patch.append(
            np.concatenate(
                [out[i, 0, 0, :], out[i, 0, 1, :], out[i, 0, 2:, :].mean(axis=0)]
            )
        )
    expected = np.stack(per_patch).mean(axis=0).astype(np.float32)
    np.testing.assert_allclose(vec, expected, atol=1e-6)


def test_maest_tiles_audio_shorter_than_min_samples(monkeypatch):
    """Audio shorter than ``min_samples`` is tiled before predict is called."""
    captured: dict[str, np.ndarray] = {}
    out = np.ones((1, 1, 3, 768), dtype=np.float32)
    short_audio = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    essentia_mod = types.ModuleType("essentia")
    essentia_mod.log = types.SimpleNamespace(warningActive=True, infoActive=True)
    standard_mod = types.ModuleType("essentia.standard")

    class _FakePredictor:
        def __init__(self, *, graphFilename, input, output):
            pass

        def __call__(self, audio):
            captured["audio"] = audio
            return out

    class _FakeMonoLoader:
        def __init__(self, *, filename, sampleRate, resampleQuality):
            pass

        def __call__(self):
            return short_audio

    standard_mod.TensorflowPredictMAEST = _FakePredictor
    standard_mod.MonoLoader = _FakeMonoLoader
    monkeypatch.setitem(sys.modules, "essentia", essentia_mod)
    monkeypatch.setitem(sys.modules, "essentia.standard", standard_mod)

    from modules.embeddings.maest import MaestProvider

    provider = MaestProvider(
        checkpoint_path="/fake/maest.pb",
        input_op="serving_default_melspectrogram",
        sample_rate=16000,
        min_samples=10,
    )
    provider.embed({"audio_path": "/fake.mp3"})

    assert captured["audio"].shape == (10,), (
        f"expected tiled to 10 samples, got {captured['audio'].shape}"
    )
    # Tiling repeats the input ([1,2,3,1,2,3,1,2,3,1]).
    np.testing.assert_array_equal(
        captured["audio"],
        np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1], dtype=np.float32),
    )


def test_maest_decode_failure_raises(monkeypatch):
    """MonoLoader errors propagate so the runner's safety net records the failure."""
    essentia_mod = types.ModuleType("essentia")
    essentia_mod.log = types.SimpleNamespace(warningActive=True, infoActive=True)
    standard_mod = types.ModuleType("essentia.standard")

    class _BoomLoader:
        def __init__(self, *, filename, sampleRate, resampleQuality):
            pass

        def __call__(self):
            raise RuntimeError("decode failed")

    class _FakePredictor:
        def __init__(self, *, graphFilename, input, output):
            pass

        def __call__(self, _audio):
            return np.zeros((1, 1, 3, 768), dtype=np.float32)

    standard_mod.TensorflowPredictMAEST = _FakePredictor
    standard_mod.MonoLoader = _BoomLoader
    monkeypatch.setitem(sys.modules, "essentia", essentia_mod)
    monkeypatch.setitem(sys.modules, "essentia.standard", standard_mod)

    from modules.embeddings.maest import MaestProvider

    provider = MaestProvider(
        checkpoint_path="/fake/maest.pb",
        input_op="serving_default_melspectrogram",
        sample_rate=16000,
        min_samples=10,
    )
    with pytest.raises(RuntimeError, match="decode failed"):
        provider.embed({"audio_path": "/fake.mp3"})
