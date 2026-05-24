# `devel` (not `runtime`): flash-attn ships only as an sdist in uv.lock and
# is marked `no-build-isolation-package`, so `uv sync --group gpu` compiles
# it against the system CUDA + Python. That needs nvcc and Python.h, neither
# of which exist on the runtime image.
#
# The CUDA major.minor here MUST match the CUDA that torch is built against,
# or flash-attn's build aborts on a version mismatch. torch>=2.11 resolves to
# the cu130 wheel (its bundled nvidia-*-cu13 deps), so the base is CUDA 13.0.
FROM nvidia/cuda:13.0.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON=python3.12

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common gnupg ca-certificates \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev python3-pip curl git \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY core/ ./core/
COPY modules/ ./modules/
COPY services/embedder/ ./services/embedder/
# Embedding tunables (max length, frame caps, fps) must match the
# orchestrator, so the pod reads them from the same config.toml via
# load_pod_config — DB/ingest secrets are not required.
COPY config.toml ./

# flash-attn has no prebuilt wheel for the cu130/torch2.11 combo, so uv builds
# it from source here. Two knobs keep that compile from OOMing a small builder:
#  - MAX_JOBS caps parallel nvcc jobs (each can eat several GB; the default is
#    one per core, which blows past a 16GB VM). Raise it if you have RAM.
#  - FLASH_ATTN_CUDA_ARCHS trims the built kernels to the fleet's GPUs:
#    80 (Ampere, e.g. A100/A30), 89 (Ada, e.g. L4/RTX 4090), 90 (Hopper, e.g.
#    H100), 120 (Blackwell, e.g. RTX PRO 6000). Ada is the cheapest 24GB stock
#    under our price cap, so it gets native sm_89 kernels rather than leaning on
#    sm_80's forward minor-version compat (which still covers sm_86 consumer
#    Ampere). The upstream default also builds 100 (B100/B200), never deployed.
ARG MAX_JOBS=2
ARG FLASH_ATTN_CUDA_ARCHS="80;89;90;120"
ENV MAX_JOBS=${MAX_JOBS} \
    FLASH_ATTN_CUDA_ARCHS=${FLASH_ATTN_CUDA_ARCHS}

# Install only what the embedder needs: the GPU build group plus the
# embedder group. We deliberately skip the project's CPU-side deps
# (sqlalchemy, hikerapi, spotify…) — they aren't imported by the service.
# --no-install-project: don't build the inst2vec wheel here; the service
# imports core/ and modules/ from /app via PYTHONPATH (WORKDIR), so we don't
# need to ship README.md, scripts/, and other build-time inputs.
RUN uv sync --group gpu --group embedder --no-dev --no-install-project

ENV HF_HOME=/workspace/hf-cache \
    MODEL_PATH=/workspace/models/Qwen3-VL-Embedding-8B \
    VIDEO_ROOT=/workspace/videos

CMD ["uv", "run", "python", "-m", "services.embedder"]
