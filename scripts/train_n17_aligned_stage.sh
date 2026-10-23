#!/usr/bin/env bash
# Rebuild a post-step recording as next-action supervision, then post-train.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
groot_root="${GROOT_ROOT:-/home/ubuntu/mycode/robotwin-gr00t-lab/vendor/Isaac-GR00T-N1.7-src}"
dataset="${DATASET_PATH:-${project_root}/data/lerobot/blocks_ranking_rgb-aligned-v1}"
output="${OUTPUT_DIR:-${project_root}/artifacts/checkpoints/n17-aligned-v1}"
base="${BASE_MODEL_PATH:-${project_root}/artifacts/checkpoints/blocks_ranking_rgb-n17-rgb-v1/checkpoint-500}"
export PYTHONPATH="${project_root}/src:${groot_root}${PYTHONPATH:+:${PYTHONPATH}}"
export NO_ALBUMENTATIONS_UPDATE=1
export TOKENIZERS_PARALLELISM=false
cd "${project_root}"
if [[ ! -d "${dataset}" ]]; then
  /home/ubuntu/env_convert/bin/python -m twingr00t_lab.conversion \
    --input-dir data/replayed/blocks_ranking_rgb-vision-batch \
    --output-dir "${dataset}" --start-episode 0 --limit 9 \
    --fps 62.5 --correct-legacy-jpeg-bgr --observation-action-offset 1 \
    --manifest results/manifests/blocks_ranking_rgb-aligned-v1.json
fi
(
  cd "${groot_root}"
  /home/ubuntu/env_groot/bin/python gr00t/data/stats.py \
    --dataset-path "${dataset}" --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path "${project_root}/configs/groot_robotwin_aloha.py"
)
/home/ubuntu/env_groot/bin/python scripts/launch_aloha_n17_lowmem.py \
  --dataset-path "${dataset}" --output-dir "${output}" \
  --base-model-path "${base}" \
  --modality-config-path configs/groot_robotwin_aloha.py \
  --max-steps "${MAX_STEPS:-200}" --save-steps "${SAVE_STEPS:-100}" \
  --save-total-limit "${SAVE_TOTAL_LIMIT:-2}" --batch-size 1 --gradient-accumulation-steps 8 \
  --optimizer adafactor --min-free-mib 22000 \
  --state-dropout-prob 0 --learning-rate "${LEARNING_RATE:-0.00005}"
