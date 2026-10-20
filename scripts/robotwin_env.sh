#!/usr/bin/env bash
# Source this file from any directory before RoboTwin commands.
VLA_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_HOME="$VLA_ROOT/tools/cuda-12.8"
export PATH="$VLA_ROOT/.venv-robotwin/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
export XDG_CACHE_HOME="$VLA_ROOT/.cache"
export TORCH_EXTENSIONS_DIR="$VLA_ROOT/.cache/torch_extensions"
export UV_CACHE_DIR="$VLA_ROOT/.cache/uv"
export TORCH_CUDA_ARCH_LIST=12.0
export MAX_JOBS=2
