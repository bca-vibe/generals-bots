#!/usr/bin/env bash
set -euo pipefail

readonly RUN_NAME="castle_ppo_a100_flex_from_012200_20260806"
readonly RUN_DIR="/workspace/runs/${RUN_NAME}"
readonly CONFIG="generals/training/configs/castle_ppo_a100_flex_from_12200.toml"
readonly HF_REPO="bca-vibe/generals-bot"
readonly HF_REVISION="main"
readonly HF_PARENT_PREFIX="runs/castle_counterfactual_anneal_long_from_004400_20260805/checkpoints"
readonly HF_OUTPUT_PREFIX="runs/${RUN_NAME}"
readonly ANCHOR_ITERATION="012000"
readonly RESUME_ITERATION="012200"
readonly ANCHOR_SHA256="fda4b60acd4be7f97598df3dcb0afd795d8169856944a5f8582fdda0c4e9d2fb"
readonly RESUME_SHA256="017837d8e268d7e18ce3ceb24684b84c89fc7a698dfde29dcdebcc6ccd5ddddb"
readonly STOP_FILE="${RUN_DIR}/.publisher_stop"

if [[ -z "${HF_TOKEN:-}" || -z "${WANDB_API_KEY:-}" ]]; then
  echo "HF_TOKEN and WANDB_API_KEY are required" >&2
  exit 2
fi

mkdir -p "${RUN_DIR}" /workspace/.cache/jax /workspace/.cache/huggingface
rm -f "${STOP_FILE}"

python - "${HF_REPO}" "${HF_REVISION}" "${HF_PARENT_PREFIX}" "${RUN_DIR}" <<'PY'
import hashlib
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

repo, revision, prefix, output = sys.argv[1:]
output = Path(output)
items = (
    ("012000", "fda4b60acd4be7f97598df3dcb0afd795d8169856944a5f8582fdda0c4e9d2fb"),
    ("012200", "017837d8e268d7e18ce3ceb24684b84c89fc7a698dfde29dcdebcc6ccd5ddddb"),
)
for iteration, expected in items:
    destination = output / f"checkpoint_{iteration}.eqx"
    if not destination.is_file():
        source = hf_hub_download(
            repo_id=repo,
            filename=f"{prefix}/iteration_{iteration}/training_checkpoint.eqx",
            revision=revision,
        )
        shutil.copy2(source, destination)
    hasher = hashlib.sha256()
    with destination.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    if digest != expected:
        raise SystemExit(f"{destination}: {digest} != {expected}")
PY

python -u tools/gcp_hf_publisher.py \
  --run-dir "${RUN_DIR}" \
  --repo-id "${HF_REPO}" \
  --revision "${HF_REVISION}" \
  --path-prefix "${HF_OUTPUT_PREFIX}" \
  --stop-file "${STOP_FILE}" &
publisher_pid=$!

cleanup() {
  touch "${STOP_FILE}"
  wait "${publisher_pid}" || true
}
trap cleanup EXIT

CUDA_VISIBLE_DEVICES="" \
TRAIN_CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
TRAIN_EXPECTED_JAX_DEVICE_COUNT="8" \
PYTHONPATH="/workspace" \
JAX_COMPILATION_CACHE_DIR="/workspace/.cache/jax" \
HF_HOME="/workspace/.cache/huggingface" \
PYTHONUNBUFFERED="1" \
python -u -m generals.training.continuation_supervisor \
  --config "${CONFIG}" \
  --resume "${RUN_DIR}/checkpoint_${RESUME_ITERATION}.eqx" \
  --duration-hours 15 \
  --hf-root "${HF_REPO}@${HF_REVISION}:${HF_OUTPUT_PREFIX}"
