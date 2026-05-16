import inspect

from modules.clustering import validation as cv_mod


def test_validate_clustering_accepts_settings():
    sig = inspect.signature(cv_mod.validate_clustering)
    assert "settings" in sig.parameters


def test_phase_filter_accepts_settings():
    sig = inspect.signature(cv_mod._phase_filter)
    assert "settings" in sig.parameters


def test_phase_plateau_accepts_settings():
    sig = inspect.signature(cv_mod._phase_plateau)
    assert "settings" in sig.parameters


def test_select_best_accepts_settings():
    sig = inspect.signature(cv_mod._select_best)
    assert "settings" in sig.parameters


def test_compute_validation_config_hash_accepts_settings():
    sig = inspect.signature(cv_mod._compute_validation_config_hash)
    assert "settings" in sig.parameters
