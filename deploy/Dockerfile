# Slim image for the read-only inst2vec serving API + the one-shot serving
# offload (compose image tag: inst2vec-serving-api:local).
#
# ENGINEERING NOTE — why we do NOT `uv sync` the project here.
# The project's [project].dependencies pull torch / vllm / essentia-tensorflow /
# onnxruntime-gpu / cuda libs — multiple GB, GPU-only, and totally unused by
# either the atlas API or the offload script. We instead install ONLY the real
# import closure of those two entrypoints and COPY just the source packages they
# touch, with PYTHONPATH=/app so absolute `core.*` / `modules.*` / `services.*`
# imports resolve.
#
# The closure was traced by hand (see RUNBOOK "Slim image" section):
#   services.atlas_api  -> fastapi, sqlalchemy, pydantic, (psycopg driver)
#   services.atlas_api/__main__ -> uvicorn[standard], python-dotenv
#   scripts.offload_serving -> + numpy (viz builders), rich (core.console),
#                              alembic (schema), python-dotenv
# No heavy/GPU package appears anywhere in that closure. The build verifies this
# by importing both entrypoints with NO torch present (see the RUN check below).
#
# `modules/` is copied wholesale, BUT the package __init__.py files of the
# touched subpackages are eager re-export shells: importing e.g.
# `modules.visualization.contract` first runs `modules/visualization/__init__.py`,
# which imports `.pipeline` -> the heavy model runners (silero_vad, essentia,
# vllm, torch). To keep the image slim WITHOUT editing the public repo, we
# OVERWRITE those __init__.py files INSIDE THE IMAGE with empty stubs (build
# artifact only — the repo on disk is byte-untouched). The atlas/offload code
# only ever imports specific LEAF modules, never the package re-exports:
#   visualization: contract, schema, export, compute, cluster_label_render
#   labels:        cases, validation, inputs
#   embeddings:    cases, providers, text
#   captions/mir/speech: state
# None of those leaves import any heavy/GPU dep (verified by the RUN check).

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# --- Minimal runtime closure (pinned to the project's lower bounds) ----------
# Keep these in sync with pyproject.toml's [project].dependencies for the names
# that overlap; everything here is pure-runtime, no compiler/GPU toolchain.
RUN pip install --no-cache-dir \
        "fastapi>=0.110" \
        "uvicorn[standard]>=0.30" \
        "sqlalchemy>=2.0.41" \
        "psycopg[binary]>=3.2" \
        "pydantic>=2.0" \
        "python-dotenv>=1.2.2" \
        "alembic>=1.13" \
        "numpy>=1.26" \
        "rich>=15.0.0"

# --- Source: only the packages in the import closure -------------------------
# COPY (not bind) so the image is self-contained on the target host.
# `services` and `scripts` are namespace dirs in the repo (no __init__.py);
# `services.atlas_api` resolves as a namespace package because its own
# atlas_api/__init__.py exists, and offload_serving.py inserts the repo root on
# sys.path itself. We mirror that layout exactly.
COPY core/ /app/core/
COPY modules/ /app/modules/
COPY services/atlas_api/ /app/services/atlas_api/
COPY scripts/offload_serving.py /app/scripts/offload_serving.py
COPY config.toml /app/config.toml
COPY alembic.ini /app/alembic.ini

# Neutralise the eager package re-exports so importing a leaf module does NOT
# drag in the heavy pipeline runners. Empty __init__ keeps each dir a package;
# the leaf modules import each other by explicit path, so nothing is lost.
RUN set -e; for pkg in visualization labels embeddings captions mir speech; do \
        : > "/app/modules/$pkg/__init__.py"; \
    done

# --- Build-time proof: the closure imports cleanly with NO torch installed ---
# Fails the build if either entrypoint accidentally drags in a heavy dep.
RUN python -c "import importlib.util; \
assert importlib.util.find_spec('torch') is None, 'torch leaked into the slim image'; \
import services.atlas_api.app; \
import services.atlas_api.reconstruct; \
import scripts.offload_serving; \
print('OK: atlas_api + offload import closure resolves, torch absent')"

# Default command runs the API. The offload service overrides `command:`.
EXPOSE 8000
CMD ["python", "-m", "services.atlas_api"]
