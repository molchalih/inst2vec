import numpy as np

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
