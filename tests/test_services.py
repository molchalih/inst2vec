"""Tests for services module: SpotifyClient and ReccoBeatsClient constructors."""

import inspect

from modules.services import ReccoBeatsClient, SpotifyClient


def test_spotify_client_takes_credentials():
    """SpotifyClient.__init__ accepts client_id and client_secret."""
    sig = inspect.signature(SpotifyClient.__init__)
    assert "client_id" in sig.parameters
    assert "client_secret" in sig.parameters
    assert "token_skew" in sig.parameters
    assert "search_limit" in sig.parameters


def test_reccobeats_client_takes_batch_params():
    """ReccoBeatsClient.__init__ accepts batch configuration parameters."""
    sig = inspect.signature(ReccoBeatsClient.__init__)
    assert "batch" in sig.parameters
    assert "delay_min" in sig.parameters
    assert "delay_max" in sig.parameters
    assert "timeout" in sig.parameters
