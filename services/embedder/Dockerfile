# `devel` (not `runtime`): flash-attn ships only as an sdist in uv.lock and
# is marked `no-build-isolation-package`, so `uv sync --group gpu` compiles
# it against the system CUDA + Python. That needs nvcc and Python.h, neither
# of which exist on the runtime image.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

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
COPY modules/ ./modules/
COPY services/embedder/ ./services/embedder/

# Install only what the embedder needs: the GPU build group plus the
# embedder group. We deliberately skip the project's CPU-side deps
# (sqlalchemy, hikerapi, spotify…) — they aren't imported by the service.
RUN uv sync --group gpu --group embedder --no-dev

ENV HF_HOME=/workspace/hf-cache \
    MODEL_PATH=/workspace/models/Qwen3-VL-Embedding-8B \
    EMBEDDER_HOST=0.0.0.0 \
    EMBEDDER_PORT=8000

EXPOSE 8000

CMD ["uv", "run", "python", "-m", "services.embedder"]
