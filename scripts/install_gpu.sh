#!/bin/bash
MAX_JOBS=1 \
FLASH_ATTENTION_FORCE_BUILD=1 \
TORCH_CUDA_ARCH_LIST="8.9" \
uv sync --group gpu
