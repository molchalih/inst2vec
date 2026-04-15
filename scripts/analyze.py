#!/usr/bin/env python
"""Dataset analysis report: pipeline health + content statistics."""
import os
import sys
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import func
from modules.database import User, Clip, Music, Download, get_session


def _pct(n, total):
    """Format n/total as a percentage string, or 'N/A' if total is zero."""
    return f"{100 * n / total:.1f}%" if total else "N/A"


def _p90(values):
    """Return 90th percentile of a list of numbers."""
    if not values:
        return 0
    sorted_v = sorted(values)
    idx = min(int(0.9 * len(sorted_v)), len(sorted_v) - 1)
    return sorted_v[idx]


def _header(title, char="="):
    print(f"\n{title}")
    print(char * len(title))
