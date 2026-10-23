#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GROOT_ROOT="${GROOT_ROOT:-${PROJECT_ROOT}/vendor/Isaac-GR00T-N1.7}"
GROOT_PYTHON="${GROOT_PYTHON:-/home/ubuntu/env_groot/bin/python}"
DATASET_PATH="${DATASET_PATH:-${PROJECT_ROOT}/data/lerobot/stack_bowls_three}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/artifacts/checkpoints/stack_bowls_three}"
MAX_STEPS="${MAX_STEPS:-2000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
NUM_GPUS="${NUM_GPUS:-1}"

if [[ ! -x "${GROOT_PYTHON}" ]]; then
  echo "GR00T Python is missing: ${GROOT_PYTHON}" >&2
  exit 1
fi
if [[ ! -d "${GROOT_ROOT}/gr00t" ]]; then
  echo "GR00T source is missing: ${GROOT_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${DATASET_PATH}/meta/stats.json" || ! -f "${DATASET_PATH}/meta/relative_stats.json" ]]; then
  echo "Dataset statistics are missing under: ${DATASET_PATH}" >&2
  exit 1
fi

min_vram_mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | head -1 | tr -d ' ')"
if (( min_vram_mib < 40000 )) && [[ "${ALLOW_UNSUPPORTED_VRAM:-0}" != "1" ]]; then
  cat >&2 <<EOF
The official GR00T N1.7 fine-tuning minimum is 40 GB VRAM per GPU.
This host reports ${min_vram_mib} MiB per GPU. Two 24 GB cards used with DDP do
not pool into one 48 GB device. Use a 48/80 GB training instance, or set
ALLOW_UNSUPPORTED_VRAM=1 only for an explicitly unsupported OOM probe.
EOF
  exit 2
fi

free_vram_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1 | tr -d ' ')"
min_free_vram_mib="${MIN_FREE_VRAM_MIB:-36000}"
if (( free_vram_mib < min_free_vram_mib )); then
  echo "Only ${free_vram_mib} MiB GPU memory is free; need ${min_free_vram_mib} MiB before training." >&2
  echo "Existing GPU processes will not be stopped." >&2
  exit 2
fi

cd "${GROOT_ROOT}"
common_args=(
  gr00t/experiment/launch_finetune.py
  --base-model-path nvidia/GR00T-N1.7-3B
  --dataset-path "${DATASET_PATH}"
  --embodiment-tag NEW_EMBODIMENT
  --modality-config-path "${PROJECT_ROOT}/configs/groot_robotwin_aloha.py"
  --num-gpus "${NUM_GPUS}"
  --output-dir "${OUTPUT_DIR}"
  --max-steps "${MAX_STEPS}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-1}"
  --save-steps "${SAVE_STEPS:-500}"
  --save-total-limit "${SAVE_TOTAL_LIMIT:-3}"
  --dataloader-num-workers "${DATALOADER_NUM_WORKERS:-4}"
  --state-dropout-prob "${STATE_DROPOUT_PROB:-0.2}"
)

if (( NUM_GPUS == 1 )); then
  exec "${GROOT_PYTHON}" "${common_args[@]}"
fi

TORCHRUN="$(dirname "${GROOT_PYTHON}")/torchrun"
exec "${TORCHRUN}" --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT:-29500}" "${common_args[@]}"
