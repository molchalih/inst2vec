import numpy as np
import pytest

import core.vendor.maest as maest_mod
from core.vendor.maest import MAEST


def test_maest_predict_means_over_patches(monkeypatch):
    # Stub mel: 3752 frames -> two 1876-frame patches
    monkeypatch.setattr(
        maest_mod, "musicnn_mel", lambda audio: np.ones((3752, 96), dtype=np.float32)
    )

    class _StubSession:
        def get_inputs(self):
            class _I:
                name = "serving_default_melspectrogram"
                shape = ("n", 1876, 96)

            return [_I()]

        def get_outputs(self):
            class _O:
                name = "preds"
                shape = ("n", 519)

            return [_O()]

        def run(self, names, feed):
            batch = next(iter(feed.values())).shape[0]
            # patch 0 -> 0.2 everywhere, patch 1 -> 0.4 -> mean 0.3
            base = np.full((batch, 519), 0.2, dtype=np.float32)
            if batch == 2:
                base[1] = 0.4
            return [base]

    monkeypatch.setattr(maest_mod, "make_session", lambda p: _StubSession())
    m = MAEST(pb=__import__("pathlib").Path("x.onnx"), patch_frames=1876)
    out = m.predict(np.zeros(48000, dtype=np.float32))
    assert out.shape == (519,)
    np.testing.assert_allclose(out, np.full(519, 0.3), rtol=1e-6)


def test_maest_selects_predictions_not_logits(monkeypatch):
    # The real ONNX graph exposes two 519-wide outputs named "logits" (linear)
    # and "activations" (the sigmoid predictions head). The wrapper must bind to
    # the predictions head, not the logits, to match the .pb reference path.
    class _Out:
        def __init__(self, name):
            self.name = name
            self.shape = ("batch_size", 519)

    class _StubSession:
        def get_inputs(self):
            class _I:
                name = "melspectrogram"
                shape = ("batch_size", 1876, 96)

            return [_I()]

        def get_outputs(self):
            return [_Out("logits"), _Out("activations")]

        def run(self, names, feed):  # pragma: no cover - not exercised here
            raise AssertionError("run should not be called")

    monkeypatch.setattr(maest_mod, "make_session", lambda p: _StubSession())
    m = MAEST(pb=__import__("pathlib").Path("x.onnx"), patch_frames=1876)
    assert m._output == "activations"


def test_maest_patches_tiles_normalizes_and_frames(monkeypatch):
    # Stub mel so we can assert the normalization + framing without Essentia.
    # 3752 frames -> two 1876-frame patches at hop 1875.
    monkeypatch.setattr(
        maest_mod,
        "musicnn_mel",
        lambda audio: np.full((3752, 96), 5.0, dtype=np.float32),
    )
    patches = maest_mod.maest_patches(
        np.zeros(48000, dtype=np.float32), patch_frames=1876, patch_hop=1875
    )
    assert patches.shape == (2, 1876, 96)
    # (5.0 - _MEL_MEAN) / (2 * _MEL_STD) applied to every cell.
    expected = (5.0 - maest_mod._MEL_MEAN) / (2.0 * maest_mod._MEL_STD)
    np.testing.assert_allclose(patches, expected, rtol=1e-6)


def _token_stub_session(output_names):
    class _O:
        def __init__(self, name):
            self.name = name
            self.shape = ("batch_size", "n_tokens", 768)

    class _I:
        name = "serving_default_melspectrogram"
        shape = ("batch_size", 1876, 96)

    class _StubSession:
        def __init__(self):
            self.requested = None

        def get_inputs(self):
            return [_I()]

        def get_outputs(self):
            return [_O(n) for n in output_names]

        def run(self, names, feed):
            self.requested = names
            patches = next(iter(feed.values()))
            # one fake [n_tokens=4, 768] block per input patch
            n = patches.shape[0]
            return [np.ones((n, 4, 768), dtype=np.float32)]

    return _StubSession()


# The real onnx graph names its outputs activations/logits + layer_0..11_tokens.
_ONNX_OUTPUT_NAMES = ["activations", "logits"] + [
    f"layer_{i}_tokens" for i in range(12)
]


def test_maest_tokens_binds_layer4_by_name(monkeypatch):
    monkeypatch.setattr(
        maest_mod, "make_session", lambda p: _token_stub_session(_ONNX_OUTPUT_NAMES)
    )
    m = maest_mod.MaestTokens(__import__("pathlib").Path("x.onnx"))
    # layer_4_tokens is the empirically-confirmed equivalent of the .pb's :7.
    assert m._output == "layer_4_tokens"


def test_maest_tokens_returns_raw_3d_output(monkeypatch):
    monkeypatch.setattr(
        maest_mod, "make_session", lambda p: _token_stub_session(_ONNX_OUTPUT_NAMES)
    )
    monkeypatch.setattr(
        maest_mod, "musicnn_mel", lambda audio: np.ones((3752, 96), dtype=np.float32)
    )
    m = maest_mod.MaestTokens(__import__("pathlib").Path("x.onnx"))
    out = m.tokens(np.zeros(48000, dtype=np.float32))
    assert out.shape == (2, 4, 768)  # two patches (3752 frames / 1876)
    assert m._session.requested == ["layer_4_tokens"]


def test_maest_tokens_raises_when_layer4_absent(monkeypatch):
    names = ["activations", "logits", "layer_0_tokens"]
    monkeypatch.setattr(maest_mod, "make_session", lambda p: _token_stub_session(names))
    with pytest.raises(ValueError, match="layer_4_tokens"):
        maest_mod.MaestTokens(__import__("pathlib").Path("x.onnx"))
