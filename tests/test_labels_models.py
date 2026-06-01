from core.config import LabelsSettings
from modules.labels.models import LabelsGenerator


class _Spy:
    def __init__(self):
        self.calls = []

    def run_many(self, paths, prompt, *, schema=None):
        self.calls.append(("run_many", list(paths), schema))
        return ["{}" for _ in paths]

    def run_text_batch(self, prompts, *, max_new_tokens, seeds, do_sample=False,
                       temperature=1.0, top_p=1.0, schema=None):
        self.calls.append(("run_text_batch", list(prompts), schema))
        return ["{}" for _ in prompts]


def test_wrapper_threads_schema():
    gen = LabelsGenerator.lazy(LabelsSettings())
    spy = _Spy()
    gen._impl = spy  # bypass _ensure_impl (no vLLM in CI)
    gen.run("/v.mp4", "p", schema={"k": 1})
    gen.run_text_batch(["a", "b"], max_new_tokens=8, seeds=[None, None], schema={"k": 2})
    assert spy.calls[0] == ("run_many", ["/v.mp4"], {"k": 1})
    assert spy.calls[1][0] == "run_text_batch"
    assert spy.calls[1][2] == {"k": 2}
    assert not hasattr(gen, "prepare_many")
