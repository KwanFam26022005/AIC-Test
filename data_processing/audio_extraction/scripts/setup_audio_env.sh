#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUDIO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${AUDIO_DIR}/../.." && pwd)"

AUDIO_ENV_DIR="${AUDIO_ENV_DIR:-${PROJECT_ROOT}/.venv_audio}"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python3.10 >/dev/null 2>&1; then
    PYTHON_BIN="python3.10"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "Could not find python/python3 on PATH." >&2
    exit 1
  fi
fi

echo "Project root: ${PROJECT_ROOT}"
echo "Audio env:    ${AUDIO_ENV_DIR}"
echo "Python:       $(${PYTHON_BIN} --version)"

"${PYTHON_BIN}" -m venv "${AUDIO_ENV_DIR}"
source "${AUDIO_ENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "${AUDIO_DIR}/requirements.txt"

echo
echo "Audio env is ready."
echo "Activate it with:"
echo "  source ${AUDIO_ENV_DIR}/bin/activate"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo
  echo "Warning: ffmpeg was not found on PATH."
fi

if ! command -v ffprobe >/dev/null 2>&1; then
  echo
  echo "Warning: ffprobe was not found on PATH."
fi

