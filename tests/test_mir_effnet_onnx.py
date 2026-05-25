import numpy as np

import core.vendor.effnet as effnet_mod
from core.vendor.effnet import EffNet


def test_effnet_embed_returns_per_patch_vectors(monkeypatch):
    monkeypatch.setattr(
        effnet_mod, "musicnn_mel", lambda audio: np.ones((256, 96), dtype=np.float32)
    )  # 256 frames, patchSize 128 / patchHop 62 -> 3 overlapping patches (0,62,124)

    class _StubSession:
        def get_inputs(self):
            class _I:
                name = "serving_default_melspectrogram"
                shape = ("n", 128, 96)

            return [_I()]

        def get_outputs(self):
            class _O:
                name = "emb"
                shape = ("n", 1280)

            return [_O()]

        def run(self, names, feed):
            batch = next(iter(feed.values())).shape[0]
            return [np.arange(batch * 1280, dtype=np.float32).reshape(batch, 1280)]

    monkeypatch.setattr(effnet_mod, "make_session", lambda p: _StubSession())
    # Heads come from Essentia; inject a stub so no .pb / TF is needed here.
    monkeypatch.setattr(
        effnet_mod,
        "_build_essentia_heads",
        lambda heads: {"danceability": lambda emb: np.full((emb.shape[0], 2), 0.7)},
    )

    eff = EffNet(
        embed_pb=__import__("pathlib").Path("e.onnx"),
        heads={"danceability": (__import__("pathlib").Path("d.pb"), "model/Softmax")},
        patch_frames=128,
    )
    emb = eff.embed(np.zeros(48000, dtype=np.float32))
    assert emb.shape == (3, 1280)
    preds = eff.predict_all(emb)
    np.testing.assert_allclose(preds["danceability"], np.full(3, 0.7).mean())
