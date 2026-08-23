#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_DIR="${INPUT_DIR:-${PROJECT_ROOT}/vendor/RoboTwin/data/stack_bowls_three}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/data/lerobot}"
GROOT_ROOT="${GROOT_ROOT:-${PROJECT_ROOT}/vendor/Isaac-GR00T-N1.7}"
GROOT_PYTHON="${GROOT_PYTHON:-/home/ubuntu/env_groot/bin/python}"
PROJECT_PYTHON="${PROJECT_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"

demo_count="$(find "${INPUT_DIR}/data" -maxdepth 1 -name 'demo_*.hdf5' -type f | wc -l)"
if [[ "${demo_count}" != "100" ]]; then
  echo "Expected 100 source demonstrations, found ${demo_count} under ${INPUT_DIR}/data" >&2
  exit 1
fi

prepare_split() {
  local name="$1"
  local start="$2"
  local count="$3"
  local dataset="${OUTPUT_ROOT}/stack_bowls_three-${name}"
  local manifest="${PROJECT_ROOT}/results/manifests/conversion-${name}.json"

  if [[ ! -d "${dataset}" ]]; then
    "${PROJECT_PYTHON}" -m twingr00t_lab.conversion \
      --input-dir "${INPUT_DIR}" \
      --output-dir "${dataset}" \
      --fps 31.25 \
      --start-episode "${start}" \
      --limit "${count}" \
      --manifest "${manifest}"
  else
    echo "[prepare] dataset exists, skipping conversion: ${dataset}"
  fi

  if [[ ! -f "${dataset}/meta/stats.json" || ! -f "${dataset}/meta/relative_stats.json" ]]; then
    (
      cd "${GROOT_ROOT}"
      "${GROOT_PYTHON}" gr00t/data/stats.py \
        --dataset-path "${dataset}" \
        --embodiment-tag NEW_EMBODIMENT \
        --modality-config-path "${PROJECT_ROOT}/configs/groot_robotwin_aloha.py"
    )
  else
    echo "[prepare] statistics exist, skipping: ${dataset}"
  fi
}

prepare_split train 0 80
prepare_split val 80 10
prepare_split test 90 10

echo "[prepare] all leakage-safe splits and relative statistics are ready"
