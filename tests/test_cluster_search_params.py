import inspect

from modules import cluster_search as cs_mod


def test_run_cluster_search_accepts_settings():
    sig = inspect.signature(cs_mod.run_cluster_search)
    assert "settings" in sig.parameters
