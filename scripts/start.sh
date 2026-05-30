#!/usr/bin/env bash
# Launch the inst2vec pipeline on a GPU machine.
#
# Reads .env for config (DATABASE_URL, HIKER_API_KEY, HUGGINGFACE_TOKEN, …).
set -euo pipefail
cd "$(dirname "$0")/.."

# Tee everything (stdout + stderr) to a timestamped log while still printing to
# the console. logs/ is gitignored.
mkdir -p logs
LOG_FILE="logs/$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[start] logging to $LOG_FILE"

# Load .env into this shell. python-dotenv also loads it at runtime and will
# not override what we export here.
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

echo "[start] syncing GPU deps (uv sync --group gpu)…"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}" MAX_JOBS="${MAX_JOBS:-1}" \
  uv sync --group gpu

echo "[start] running pipeline…"
uv run python main.py
