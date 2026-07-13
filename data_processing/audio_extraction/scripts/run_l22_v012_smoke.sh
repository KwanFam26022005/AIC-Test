#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUDIO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${AUDIO_DIR}/../.." && pwd)"

AUDIO_ENV_DIR="${AUDIO_ENV_DIR:-${PROJECT_ROOT}/.venv_audio}"
TEST_VIDEO="${TEST_VIDEO:-/tmp2/maitanha/vgu/ttn/data/AIC2025/videos/Videos_L22/L22_V012.mp4}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/audio/L22_V012}"
CONFIG_PATH="${CONFIG_PATH:-${AUDIO_DIR}/configs/audio_pipeline_a5000.yaml}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

if [[ ! -d "${AUDIO_ENV_DIR}" ]]; then
  echo "Audio env not found: ${AUDIO_ENV_DIR}" >&2
  echo "Create it first with:" >&2
  echo "  bash ${AUDIO_DIR}/scripts/setup_audio_env.sh" >&2
  exit 1
fi

if [[ ! -f "${TEST_VIDEO}" ]]; then
  echo "Test video not found: ${TEST_VIDEO}" >&2
  exit 1
fi

source "${AUDIO_ENV_DIR}/bin/activate"

echo "Project root: ${PROJECT_ROOT}"
echo "Audio env:    ${AUDIO_ENV_DIR}"
echo "Test video:   ${TEST_VIDEO}"
echo "Output dir:   ${OUTPUT_DIR}"
echo "GPU:          CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

python "${AUDIO_DIR}/run_audio_pipeline.py" \
  --video "${TEST_VIDEO}" \
  --video_id L22_V012 \
  --output_dir "${OUTPUT_DIR}" \
  --config "${CONFIG_PATH}" \
  "$@"

