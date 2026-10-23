#!/usr/bin/env bash
# Auditable single-checkpoint search across distinct factory teacher initial states.
# Teacher actions are read only by the recorder's reset/teleport code; the policy
# produces every executed action. A success is still only one selected case.
set -u

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root" || exit 1

: "${MODEL_PATH:?set MODEL_PATH to a complete N1.7 checkpoint}"
: "${ACTION_STATS:?set ACTION_STATS to the matching train statistics}"
: "${DATASET_DIR:?set DATASET_DIR to the factory HDF5 set}"
: "${RUN_LABEL:?set a unique RUN_LABEL}"

first=${FIRST_EPISODE:-0}
last=${LAST_EPISODE:-8}
if (( first < 0 || last < first )); then
  echo "Invalid episode range: $first..$last" >&2
  exit 2
fi

mkdir -p results/n17_factory results/logs
for (( episode=first; episode<=last; episode++ )); do
  demo="$DATASET_DIR/data/demo_${episode}.hdf5"
  stem="results/n17_factory/${RUN_LABEL}_demo${episode}"
  if [[ ! -f "$demo" ]]; then
    echo "Missing source initial-state demo: $demo" >&2
    exit 2
  fi
  if [[ -e "${stem}.mp4" || -e "${stem}.json" ]]; then
    echo "Refusing to overwrite existing evidence: $stem" >&2
    exit 2
  fi
  echo "[search] episode=$episode start=$(date -u +%FT%TZ) checkpoint=$MODEL_PATH"
  MODEL_PATH="$MODEL_PATH" ACTION_STATS="$ACTION_STATS" HELDOUT_DEMO="$demo" \
    OUTPUT_VIDEO="${project_root}/${stem}.mp4" INITIAL_STATE_SPLIT=train \
    POLICY_PORT="${POLICY_PORT:-5570}" POLICY_SEED="${POLICY_SEED:-42}" SIM_SEED="${SIM_SEED:-42}" \
    SCENE_STYLE=factory MAX_STEPS="${MAX_STEPS:-1500}" EXECUTE_STEPS="${EXECUTE_STEPS:-8}" \
    ACTION_ALPHA="${ACTION_ALPHA:-0.2}" RECORD_EVERY=2 \
    bash scripts/record_aloha_n17_factory.sh > "results/logs/${RUN_LABEL}_demo${episode}.log" 2>&1
  record_status=$?
  if [[ ! -f "${stem}.json" ]]; then
    echo "[search] episode=$episode infrastructure/error exit=$record_status; no manifest" >&2
    exit 1
  fi
  status=$(/home/ubuntu/env_convert/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "${stem}.json")
  if [[ "$status" == success && "$record_status" -eq 0 ]]; then
    PYTHONPATH=src /home/ubuntu/env_convert/bin/python scripts/verify_n17_recording.py \
      "${stem}.json" > "${stem}.verify.json" || exit 1
    echo "[search] episode=$episode VERIFIED_SUCCESS; selected-case result only"
    exit 0
  fi
  if [[ "$status" != task_not_completed ]]; then
    echo "[search] episode=$episode unexpected status=$status exit=$record_status" >&2
    exit 1
  fi
  PYTHONPATH=src /home/ubuntu/env_convert/bin/python scripts/verify_n17_recording.py \
    "${stem}.json" --allow-failed > "${stem}.verify.json" || exit 1
  echo "[search] episode=$episode task_not_completed"
done
echo "[search] no verified success in episodes $first..$last" >&2
exit 3
