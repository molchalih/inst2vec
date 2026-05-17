import inspect

from modules.clustering import validation as cv_mod


def test_validate_clustering_accepts_settings():
    sig = inspect.signature(cv_mod.validate_clustering)
    assert "settings" in sig.parameters
