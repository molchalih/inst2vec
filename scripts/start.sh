#!/usr/bin/env bash
# Launch inst2vec on a GPU machine. Mirrors main.py's modes:
#   scripts/start.sh --orchestrator   (default) run the pipeline; auto-starts a
#                                     Cloudflare quick tunnel for the embedding
#                                     coordinator when pods are enabled.
#   scripts/start.sh --pod            run as an embedding pull-worker.
#
# Reads .env for config (RUNPOD_*, EMBEDDER_TOKEN, COORDINATOR_PUBLIC_HOST, …).
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:---orchestrator}"

# Load .env into this shell so tunnel/host vars are visible here. python-dotenv
# also loads it at runtime and will not override what we export below.
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

# --orchestrator imports modules.embeddings.coordinator (fastapi/uvicorn, the
# `embedder` group) whenever a coordinator/pods are in play, so sync it alongside
# the GPU group; it is tiny and harmless on the --pod path.
echo "[start] syncing GPU + embedder deps (uv sync --group gpu --group embedder)…"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}" MAX_JOBS="${MAX_JOBS:-1}" \
  uv sync --group gpu --group embedder

case "$MODE" in
  --pod)
    : "${ORCHESTRATOR_HOST:?set ORCHESTRATOR_HOST (e.g. https://x.trycloudflare.com)}"
    : "${EMBEDDER_TOKEN:?set EMBEDDER_TOKEN}"
    echo "[start] pod → $ORCHESTRATOR_HOST"
    exec uv run python main.py --pod --host="$ORCHESTRATOR_HOST" \
      --video-root="${VIDEO_ROOT:-/workspace/videos}"
    ;;

  --orchestrator)
    # Tunnel the SAME port the coordinator binds. The Python side reads
    # [embeddings].coordinator_bind_port from config.toml (no env override), so
    # read it from there too — otherwise changing the TOML would leave the tunnel
    # pointing at a stale port and auto-deployed pods would time out.
    PORT="$(uv run --no-sync python -c 'import tomllib; print(tomllib.load(open("config.toml","rb")).get("embeddings",{}).get("coordinator_bind_port",8765))')"
    if [ -n "${COORDINATOR_PUBLIC_HOST:-}" ]; then
      echo "[start] using preset COORDINATOR_PUBLIC_HOST=$COORDINATOR_PUBLIC_HOST"
    elif [ "${RUNPOD_POD_COUNT:-0}" = "0" ]; then
      echo "[start] RUNPOD_POD_COUNT=0 → local-only embedding, no tunnel needed"
    else
      command -v cloudflared >/dev/null || {
        echo "[start] cloudflared not found; install it or set COORDINATOR_PUBLIC_HOST" >&2
        exit 1
      }
      echo "[start] starting Cloudflare quick tunnel → http://localhost:$PORT …"
      TUNNEL_LOG="$(mktemp)"
      cloudflared tunnel --url "http://localhost:$PORT" --no-autoupdate \
        >"$TUNNEL_LOG" 2>&1 &
      TUNNEL_PID=$!
      trap 'kill "$TUNNEL_PID" 2>/dev/null || true; rm -f "$TUNNEL_LOG"' EXIT
      url=""
      for _ in $(seq 1 30); do
        url=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$TUNNEL_LOG" | head -1 || true)
        [ -n "$url" ] && break
        sleep 1
      done
      if [ -z "$url" ]; then
        echo "[start] tunnel URL not found after 30s:" >&2
        cat "$TUNNEL_LOG" >&2
        exit 1
      fi
      export COORDINATOR_PUBLIC_HOST="$url"
      echo "[start] coordinator public host: $COORDINATOR_PUBLIC_HOST"
    fi
    echo "[start] running pipeline…"
    uv run python main.py
    ;;

  *)
    echo "usage: $0 [--orchestrator|--pod]" >&2
    exit 2
    ;;
esac
