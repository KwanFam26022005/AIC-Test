#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OBJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${OBJECT_DIR}/../.." && pwd)"

ENV_ROOT="${OBJECT_DETECTION_ENV_ROOT:-/tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs}"
ENV_NAME="${OBJECT_DETECTION_ENV_NAME:-aic-object-detection-gpu}"
ENV_DIR="${OBJECT_DETECTION_ENV_DIR:-${ENV_ROOT}/${ENV_NAME}}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found on PATH." >&2
  exit 1
fi

mkdir -p "${ENV_ROOT}"

if [[ ! -d "${ENV_DIR}" ]]; then
  conda create --prefix "${ENV_DIR}" "python=${PYTHON_VERSION}" -y
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_DIR}"

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision --index-url "${TORCH_INDEX_URL}"
python -m pip install -r "${OBJECT_DIR}/requirements.txt"

cat <<EOF

Object detection env is ready.
Project root: ${PROJECT_ROOT}
Object dir:    ${OBJECT_DIR}
Env:           ${ENV_DIR}

Activate it with:
  source "\$(conda info --base)/etc/profile.d/conda.sh"
  conda activate ${ENV_DIR}
EOF
