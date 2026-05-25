from core.vendor.onnx_session import pick_output_by_lastdim


class _FakeOut:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


def test_pick_output_by_lastdim_selects_matching():
    outs = [_FakeOut("logits", ["n", 519]), _FakeOut("emb", ["n", 768])]
    # Two have 519? here only one; ties broken by predicate
    assert pick_output_by_lastdim(outs, 768) == "emb"


def test_pick_output_by_lastdim_with_tiebreak_predicate():
    outs = [
        _FakeOut("StatefulPartitionedCall:13", ["n", 519]),  # logits (linear)
        _FakeOut("StatefulPartitionedCall:0", ["n", 519]),  # predictions (sigmoid)
    ]
    chosen = pick_output_by_lastdim(outs, 519, prefer=lambda o: o.name.endswith(":0"))
    assert chosen == "StatefulPartitionedCall:0"


def test_pick_output_by_lastdim_raises_when_absent():
    outs = [_FakeOut("emb", ["n", 1280])]
    try:
        pick_output_by_lastdim(outs, 999)
    except ValueError as e:
        assert "999" in str(e)
    else:
        raise AssertionError("expected ValueError")
