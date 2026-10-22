#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
groot_root="${GROOT_ROOT:-/home/ubuntu/mycode/robotwin-gr00t-lab/vendor/Isaac-GR00T-N1.7-src}"
groot_python="${GROOT_PYTHON:-/home/ubuntu/env_groot/bin/python}"
sim_python="${SIM_PYTHON:-/home/ubuntu/env_robotwin/bin/python}"
model_path="${MODEL_PATH:?Set MODEL_PATH to a trained GR00T N1.7 checkpoint directory}"
heldout_demo="${HELDOUT_DEMO:-${project_root}/data/replayed/blocks_ranking_rgb-vision-ep1/data/demo_0.hdf5}"
output="${OUTPUT_VIDEO:-${project_root}/results/n17_factory/aloha_n17_online.mp4}"
port="${POLICY_PORT:-5555}"

[[ -d "${model_path}" && -f "${heldout_demo}" ]] || {
  echo "The trained checkpoint and held-out initial-state demo must exist." >&2
  exit 1
}
[[ ! -e "${output}" ]] || {
  echo "Refusing to overwrite ${output}" >&2
  exit 1
}
mkdir -p "$(dirname "${output}")"
export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
# Isaac Sim's Python package prompts in noninteractive SSH sessions unless the
# already-required Omniverse EULA is acknowledged through its documented flag.
export OMNI_KIT_ACCEPT_EULA=YES

server_pid=""
cleanup() {
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill -TERM "${server_pid}"
    wait "${server_pid}" || true
  fi
}
trap cleanup EXIT

(
  cd "${groot_root}"
  exec "${groot_python}" "${project_root}/scripts/serve_aloha_n17.py" \
    --model-path "${model_path}" \
    --flow-steps "${FLOW_STEPS:-4}" \
    --sample-average "${SAMPLE_AVERAGE:-1}" \
    --host 127.0.0.1 \
    --port "${port}"
) > "${output%.mp4}.server.log" 2>&1 &
server_pid=$!

ready=0
for _ in $(seq 1 180); do
  kill -0 "${server_pid}" 2>/dev/null || {
    echo "N1.7 server exited during startup; inspect ${output%.mp4}.server.log" >&2
    exit 1
  }
  if ss -ltn "( sport = :${port} )" | grep -q LISTEN; then
    ready=1
    break
  fi
  sleep 2
done
[[ "${ready}" == 1 ]] || {
  echo "N1.7 server did not become ready within 6 minutes." >&2
  exit 1
}

cd "${project_root}"
prompt_args=()
if [[ -n "${POLICY_PROMPT:-}" ]]; then
  prompt_args=(--prompt "${POLICY_PROMPT}")
fi
camera_args=()
if [[ "${POLICY_CAMERA_BGR_SWAP:-0}" == 1 ]]; then
  camera_args=(--policy-camera-bgr-swap)
fi
"${sim_python}" scripts/run_aloha_n17_closedloop.py "${heldout_demo}" \
  --output "${output}" \
  --checkpoint "${model_path}" \
  --action-stats "${project_root}/data/lerobot/blocks_ranking_rgb-train/meta/stats.json" \
  --port "${port}" \
  --max-steps "${MAX_STEPS:-1500}" \
  --execute-steps "${EXECUTE_STEPS:-8}" \
  --record-every "${RECORD_EVERY:-2}" \
  --initial-state-split "${INITIAL_STATE_SPLIT:-heldout}" \
  --flow-steps "${FLOW_STEPS:-4}" \
  --sample-average "${SAMPLE_AVERAGE:-1}" \
  --action-alpha "${ACTION_ALPHA:-0.2}" \
  --max-joint-step "${MAX_JOINT_STEP:-0.06}" \
  --max-gripper-step "${MAX_GRIPPER_STEP:-0.008}" \
  "${prompt_args[@]}" \
  "${camera_args[@]}" \
  --headless \
  --environment \
    manip_eval_tasks.examples.manipulation.blocks_ranking_rgb_twingroot_environment:BlocksRankingRgbEnvironment \
    blocks_ranking_rgb \
  --embodiment aloha \
  --enable_cameras True
