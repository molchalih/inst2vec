import os
import sys

from IPython.display import Markdown

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def test_spotify_feature_metrics_to_markdown_renders_expected_rows():
    from docs.reporting.tables.spotify import (
        spotify_feature_metrics_to_markdown,
    )

    out = spotify_feature_metrics_to_markdown()

    assert out.startswith("| Metric | Value |")
    assert "| Acousticness | 0-1 |" in out
    assert "| Danceability | 0-1 |" in out
    assert "| Energy | 0-1 |" in out
    assert "| Instrumentalness | 0-1 |" in out
    assert "| Key | 0-11 |" in out
    assert "| Liveness | 0-1 |" in out
    assert "| Loudness | -60 to 0 dB |" in out
    assert "| Mode | 0 or 1 |" in out
    assert "| Speechiness | 0-1 |" in out
    assert "| Tempo | BPM (>0) |" in out
    assert "| Valence | 0-1 |" in out


def test_render_spotify_feature_metrics_returns_markdown_object():
    from docs.quarto_helpers import render_spotify_feature_metrics
    from docs.reporting.tables.spotify import (
        spotify_feature_metrics_to_markdown,
    )

    rendered = render_spotify_feature_metrics()

    assert isinstance(rendered, Markdown)
    assert rendered.data == spotify_feature_metrics_to_markdown()
