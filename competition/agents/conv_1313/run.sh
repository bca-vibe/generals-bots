#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 inter_op_parallelism_threads=1"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export JAX_NUM_THREADS=1
export JAX_COMPILATION_CACHE_DIR="$PWD/.jax_cache"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0

exec python -u main.py
