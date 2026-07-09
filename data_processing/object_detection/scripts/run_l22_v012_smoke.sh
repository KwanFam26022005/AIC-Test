#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OBJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${OBJECT_DIR}/../.." && pwd)"

ENV_DIR="${OBJECT_DETECTION_ENV_DIR:-/tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/aic-object-detection-gpu}"
TEST_FRAMES="${TEST_FRAMES:-${PROJECT_ROOT}/keyframe_test/L22_V012}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/object_detection}"
CONFIG_PATH="${CONFIG_PATH:-${OBJECT_DIR}/configs/object_detection_a5000.yaml}"
VIDEO_ID="${VIDEO_ID:-L22_V012}"
LIMIT="${LIMIT:-20}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
RAM_CHECKPOINT_PATH="${RAM_CHECKPOINT_PATH:-/tmp2/maitanha/vgu/ttn/AIC-Khoa/models/ram_plus_swin_large_14m.pth}"
HF_HOME="${HF_HOME:-/tmp2/maitanha/vgu/ttn/AIC-Khoa/hf_cache}"

export CUDA_VISIBLE_DEVICES
export RAM_CHECKPOINT_PATH
export HF_HOME

if [[ ! -d "${ENV_DIR}" ]]; then
  echo "Object detection env not found: ${ENV_DIR}" >&2
  echo "Create it first with:" >&2
  echo "  bash ${OBJECT_DIR}/scripts/setup_object_detection_env.sh" >&2
  exit 1
fi

if [[ ! -d "${TEST_FRAMES}" ]]; then
  echo "Test frames directory not found: ${TEST_FRAMES}" >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_DIR}"

mkdir -p "${OUTPUT_DIR}" "${OUTPUT_DIR}/summaries"

echo "Project root: ${PROJECT_ROOT}"
echo "Object dir:    ${OBJECT_DIR}"
echo "Env:           ${ENV_DIR}"
echo "Frames:        ${TEST_FRAMES}"
echo "Output dir:    ${OUTPUT_DIR}"
echo "GPU:           CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

python "${OBJECT_DIR}/ram_gdino_pipeline.py" \
  --config "${CONFIG_PATH}" \
  --input "${TEST_FRAMES}" \
  --output "${OUTPUT_DIR}/${VIDEO_ID}_objects.jsonl" \
  --video-id "${VIDEO_ID}" \
  --limit "${LIMIT}" \
  --summary-output "${OUTPUT_DIR}/summaries/${VIDEO_ID}_summary.json" \
  "$@"
