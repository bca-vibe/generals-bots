#!/bin/sh
# Reproducible training environment for givemeanode CUDA images.
#
# Bundled mode is the most portable, but downloads roughly 3.1 GiB of CUDA
# wheels. It therefore requires an explicit acknowledgement. Local mode reuses
# CUDA/cuDNN/NCCL from the image and is intended for Dockerfile.givemeanode.
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PYTHON_VERSION=${PYTHON_VERSION:-3.11.15}
JAX_VERSION=${JAX_VERSION:-0.10.2}
VENV_PATH=${VENV_PATH:-"$ROOT_DIR/.venv"}
UV_CACHE_DIR=${UV_CACHE_DIR:-"$ROOT_DIR/.cache/uv"}
CUDA_MODE=${CUDA_MODE:-}
EXPECTED_GPU_COUNT=${EXPECTED_GPU_COUNT:-}

if [ "$CUDA_MODE" != "bundled" ] && [ "$CUDA_MODE" != "local" ]; then
    echo "Set CUDA_MODE to either 'bundled' or 'local'." >&2
    exit 2
fi
if [ "$CUDA_MODE" = "bundled" ] && [ "${ALLOW_BUNDLED_CUDA_DOWNLOAD:-}" != "1" ]; then
    echo "Bundled CUDA installs roughly 3.1 GiB of wheels." >&2
    echo "Set ALLOW_BUNDLED_CUDA_DOWNLOAD=1 to acknowledge that download." >&2
    exit 2
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required but was not found on PATH." >&2
    exit 2
fi

mkdir -p "$UV_CACHE_DIR"
export UV_CACHE_DIR
uv python install "$PYTHON_VERSION"

if [ -x "$VENV_PATH/bin/python" ]; then
    ACTUAL_PYTHON=$(
        "$VENV_PATH/bin/python" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))'
    )
    if [ "$ACTUAL_PYTHON" != "$PYTHON_VERSION" ]; then
        echo "Refusing to reuse $VENV_PATH: Python $ACTUAL_PYTHON != $PYTHON_VERSION." >&2
        echo "Choose a new VENV_PATH or remove the incorrect environment explicitly." >&2
        exit 2
    fi
else
    uv venv --python "$PYTHON_VERSION" "$VENV_PATH"
fi

# Install the exact lock without installing the local project. This works for
# minimal recovery contexts where README.md was intentionally omitted.
UV_PROJECT_ENVIRONMENT="$VENV_PATH" uv sync \
    --locked \
    --no-install-project \
    --extra train \
    --extra tracking \
    --python "$VENV_PATH/bin/python"

if [ "$CUDA_MODE" = "bundled" ]; then
    JAX_EXTRA="cuda12"
else
    if ! ldconfig -p 2>/dev/null | grep -q 'libcudnn\.so'; then
        echo "CUDA_MODE=local requires cuDNN in the base image; none was found." >&2
        exit 2
    fi
    JAX_EXTRA="cuda12-local"
fi

uv pip install --python "$VENV_PATH/bin/python" "jax[$JAX_EXTRA]==$JAX_VERSION"

VERIFY_ARGS="--python 3.11 --jax $JAX_VERSION --jaxlib $JAX_VERSION"
if [ -n "$EXPECTED_GPU_COUNT" ]; then
    VERIFY_ARGS="$VERIFY_ARGS --devices $EXPECTED_GPU_COUNT"
fi
# Intentional word splitting: VERIFY_ARGS is assembled only from validated
# constants and an integer supplied by the caller.
# shellcheck disable=SC2086
"$VENV_PATH/bin/python" "$ROOT_DIR/tools/verify_training_runtime.py" $VERIFY_ARGS
