#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_DIR="${INPUT_DIR:-${PROJECT_ROOT}/data/replayed/blocks_ranking_rgb}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/data/lerobot}"
GROOT_ROOT="${GROOT_ROOT:-${PROJECT_ROOT}/vendor/Isaac-GR00T-N1.7}"
CONVERT_PYTHON="${CONVERT_PYTHON:-/home/ubuntu/env_convert/bin/python}"
GROOT_PYTHON="${GROOT_PYTHON:-/home/ubuntu/env_groot/bin/python}"
FPS="${FPS:-50}"

for executable in "${CONVERT_PYTHON}" "${GROOT_PYTHON}"; do
  if [[ ! -x "${executable}" ]]; then
    echo "Missing Python interpreter: ${executable}" >&2
    exit 1
  fi
done
if [[ ! -f "${GROOT_ROOT}/gr00t/data/stats.py" ]]; then
  echo "Missing pinned GR00T N1.7 source: ${GROOT_ROOT}" >&2
  exit 1
fi

mapfile -t demos < <(find "${INPUT_DIR}/data" -maxdepth 1 -type f -name 'demo_*.hdf5' | sort -V)
total="${#demos[@]}"
if (( total < 10 )); then
  echo "Need at least 10 successful, audited replays for an 80/10/10 split; found ${total}" >&2
  exit 1
fi

train_count=$((total * 8 / 10))
val_count=$((total / 10))
test_count=$((total - train_count - val_count))

prepare_split() {
  local name="$1"
  local start="$2"
  local count="$3"
  local dataset="${OUTPUT_ROOT}/blocks_ranking_rgb-${name}"
  local manifest="${PROJECT_ROOT}/results/manifests/blocks_ranking_rgb-${name}.json"

  if [[ -e "${dataset}" ]]; then
    echo "Refusing to reuse an existing split: ${dataset}" >&2
    exit 1
  fi

  "${CONVERT_PYTHON}" -m twingr00t_lab.conversion \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${dataset}" \
    --fps "${FPS}" \
    --start-episode "${start}" \
    --limit "${count}" \
    --manifest "${manifest}"

  (
    cd "${GROOT_ROOT}"
    "${GROOT_PYTHON}" gr00t/data/stats.py \
      --dataset-path "${dataset}" \
      --embodiment-tag NEW_EMBODIMENT \
      --modality-config-path "${PROJECT_ROOT}/configs/groot_robotwin_aloha.py"
  )
}

prepare_split train 0 "${train_count}"
prepare_split val "${train_count}" "${val_count}"
prepare_split test "$((train_count + val_count))" "${test_count}"

echo "Prepared ${train_count}/${val_count}/${test_count} train/val/test episodes at ${FPS} fps"
