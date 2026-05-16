"""Static markdown table listing Spotify audio feature metrics."""

from __future__ import annotations

__all__ = ("spotify_feature_metrics_to_markdown",)


SPOTIFY_FEATURE_METRICS: tuple[tuple[str, str], ...] = (
    ("acousticness", "0-1"),
    ("danceability", "0-1"),
    ("energy", "0-1"),
    ("instrumentalness", "0-1"),
    ("key", "0-11"),
    ("liveness", "0-1"),
    ("loudness", "-60 to 0 dB"),
    ("mode", "0 or 1"),
    ("speechiness", "0-1"),
    ("tempo", "BPM (>0)"),
    ("valence", "0-1"),
)


def _display_name(metric: str) -> str:
    return metric.replace("_", " ").title()


def spotify_feature_metrics_to_markdown() -> str:
    lines = ["| Metric | Value |", "|---|---|"]
    for metric, value_range in SPOTIFY_FEATURE_METRICS:
        lines.append(f"| {_display_name(metric)} | {value_range} |")
    return "\n".join(lines)
