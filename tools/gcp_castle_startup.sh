#!/usr/bin/env bash
set -euo pipefail

exec >>/var/log/castle-training-startup.log 2>&1

readonly PROJECT_ID="project-8cd09d98-d730-4216-844"
readonly REGION="us-central1"
readonly IMAGE_TAG="$(curl --fail --silent \
  -H 'Metadata-Flavor: Google' \
  'http://metadata.google.internal/computeMetadata/v1/instance/attributes/castle-image-tag')"
readonly IMAGE="us-central1-docker.pkg.dev/${PROJECT_ID}/generals-training/castle-trainer:${IMAGE_TAG}"
readonly CONTAINER_NAME="castle-ppo-a100-flex-12200"
readonly RUNS_ROOT="/opt/castle-runs"

finish() {
  status=$?
  printf '%s\n' "${status}" >"${RUNS_ROOT}/last_container_exit_status"
  sync
  shutdown -h now || true
}
mkdir -p "${RUNS_ROOT}"
trap finish EXIT

systemctl start docker
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

export HF_TOKEN
export WANDB_API_KEY
HF_TOKEN="$(gcloud secrets versions access latest --secret=castle-hf-token --project="${PROJECT_ID}")"
WANDB_API_KEY="$(gcloud secrets versions access latest --secret=castle-wandb-api-key --project="${PROJECT_ID}")"

docker pull "${IMAGE}"
docker run --rm \
  --name "${CONTAINER_NAME}" \
  --gpus all \
  --ipc host \
  --shm-size 64g \
  --ulimit memlock=-1 \
  --env HF_TOKEN \
  --env WANDB_API_KEY \
  --env WANDB_MODE=online \
  --volume "${RUNS_ROOT}:/workspace/runs" \
  "${IMAGE}" \
  bash tools/run_gcp_castle_continuation.sh
