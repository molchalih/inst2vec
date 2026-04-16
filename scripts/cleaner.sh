#!/usr/bin/env bash

# remove duplicate lines from data/data.csv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CSV="${SCRIPT_DIR}/../data/data.csv"

if [[ ! -f "$CSV" ]]; then
  echo "error: $CSV not found" >&2
  exit 1
fi

tmp="$(mktemp "${CSV}.tmp.XXXXXX")"
trap 'rm -f "$tmp"' EXIT

# First occurrence of each line wins; order is preserved.
awk '!seen[$0]++' "$CSV" >"$tmp"
mv "$tmp" "$CSV"
trap - EXIT
