#!/usr/bin/env bash
# Follow an already-running training process with bounded, sequential diagnostics.
# No publication or human-video-review status is changed by this script.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"
training_pid="${TRAINING_PID:?Set the PID of the existing training process}"
[[ "${training_pid}" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid training PID" >&2; exit 2; }
checkpoint="${MODEL_PATH:?Set the expected completed checkpoint directory}"
training_output="$(dirname "${checkpoint}")"
run_id="${RUN_ID:?Set a unique recording prefix}"
[[ "${run_id}" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "Invalid run prefix" >&2; exit 2; }
expected_step="${EXPECTED_TRAINING_STEP:-4000}"
[[ "${expected_step}" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid expected step" >&2; exit 2; }
groot_root="${GROOT_ROOT:-/home/ubuntu/mycode/robotwin-gr00t-lab/vendor/Isaac-GR00T-N1.7-src}"
convert_python="${CONVERT_PYTHON:-/home/ubuntu/env_convert/bin/python}"
groot_python="${GROOT_PYTHON:-/home/ubuntu/env_groot/bin/python}"
gpu_id="${DIAGNOSTIC_GPU_ID:-0}"
port="${POLICY_PORT:-5555}"
train_demo="${TRAIN_DEMO:-${project_root}/data/replayed/blocks_ranking_rgb-vision-batch/data/demo_0.hdf5}"
heldout_demo="${HELDOUT_DEMO:-${project_root}/data/replayed/blocks_ranking_rgb-vision-ep1/data/demo_0.hdf5}"
action_stats="${ACTION_STATS:-${project_root}/data/lerobot/blocks_ranking_rgb-aligned-v1/meta/stats.json}"
correct_legacy_jpeg_bgr="${CORRECT_LEGACY_JPEG_BGR-1}"
# Online frames come directly from Isaac in RGB, independently of the legacy
# offline JPEG encoding. Refuse contradictory inherited diagnostic settings.
if [[ "${POLICY_CAMERA_BGR_SWAP-0}" != 0 ]]; then
  echo "Online Isaac cameras are RGB: POLICY_CAMERA_BGR_SWAP must be unset or 0; offline JPEG correction is independent." >&2
  exit 2
fi
case "${correct_legacy_jpeg_bgr}" in
  0|1) ;;
  *) echo "CORRECT_LEGACY_JPEG_BGR must be exactly 0 or 1" >&2; exit 2 ;;
esac
legacy_color_args=()
if [[ "${correct_legacy_jpeg_bgr}" == 1 ]]; then legacy_color_args+=(--correct-legacy-jpeg-bgr); fi
[[ "${gpu_id}" =~ ^[0-9]+$ && "${port}" =~ ^[1-9][0-9]*$ && "${port}" -le 65535 ]] || {
  echo "Invalid GPU index or policy port" >&2; exit 2;
}
[[ -x "${convert_python}" && -x "${groot_python}" ]] || { echo "Python environment missing" >&2; exit 2; }
for tool in flock nvidia-smi ss; do command -v "${tool}" >/dev/null; done
export PYTHONPATH="${project_root}/src:${groot_root}${PYTHONPATH:+:${PYTHONPATH}}"
export NO_ALBUMENTATIONS_UPDATE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${gpu_id}"

# A disappeared process alone is not evidence of successful training. Keep this
# CPU guard self-contained so it can be tested without importing Torch or Isaac.
"${convert_python}" - "${checkpoint}" "${training_output}" "${expected_step}" "${training_pid}" <<'PY'
import json
import os
import struct
import sys
import time
from pathlib import Path


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_process(pid, proc_root=Path("/proc")):
    directory = proc_root / str(pid)
    try:
        # comm can contain spaces/parentheses; fields after its final ')' start at state (3).
        fields = (directory / "stat").read_text().rsplit(") ", 1)[1].split()
        return {
            "state": fields[0], "start_ticks": int(fields[19]),
            "argv": (directory / "cmdline").read_bytes().decode().rstrip("\0").split("\0"),
            "cwd": (directory / "cwd").resolve(),
        }
    except FileNotFoundError:
        return None


def verify_training_identity(process, output):
    require(process is not None and process["state"] not in ("Z", "X"), "Training PID is not running at watcher startup")
    argv = process["argv"]
    require(any(Path(arg).name == "launch_aloha_n17_lowmem.py" for arg in argv), "PID is not the N1.7 training launcher")
    requested = None
    for index, arg in enumerate(argv):
        if arg == "--output-dir" and index + 1 < len(argv):
            requested = argv[index + 1]
        elif arg.startswith("--output-dir="):
            requested = arg.split("=", 1)[1]
    require(requested is not None, "Training PID has no explicit output directory")
    require((process["cwd"] / requested).resolve() == output.resolve(), "Training PID output directory does not match checkpoint")


def wait_for_training(pid, output):
    initial = read_process(pid)
    verify_training_identity(initial, output)
    boot = next(int(line.split()[1]) for line in Path("/proc/stat").read_text().splitlines() if line.startswith("btime "))
    started = boot + initial["start_ticks"] / os.sysconf("SC_CLK_TCK")
    print(f"Waiting for training PID {pid}, start_ticks={initial['start_ticks']}, output={output}", flush=True)
    while True:
        current = read_process(pid)
        if current is None:
            break
        require(current["start_ticks"] == initial["start_ticks"], "Training PID was reused; refusing ambiguous completion")
        if current["state"] in ("Z", "X"):
            break
        verify_training_identity(current, output)
        time.sleep(30)
    return started


def read_json(path):
    return json.loads(path.read_text())


def processor_config(directory):
    path = directory / "processor_config.json"
    return path if path.is_file() else directory / "processor" / "processor_config.json"


def verify_checkpoint(checkpoint, output, expected_step, training_started):
    checkpoint, output = checkpoint.resolve(), output.resolve()
    require(checkpoint.parent == output and checkpoint.name == f"checkpoint-{expected_step}", "Checkpoint path/step mismatch")
    state = read_json(checkpoint / "trainer_state.json")
    require(type(state.get("global_step")) is int and state["global_step"] == expected_step, "Incomplete training checkpoint")
    audit_path = output / "visual_augmentation_audit.json"
    require(audit_path.stat().st_mtime >= training_started - 1, "Visual audit predates this training process")
    audit = read_json(audit_path)
    require(audit.get("status") == "verified_color_off" and audit.get("mode") == "color-off", "Effective augmentation not verified")
    effective_path = processor_config(output)
    require(Path(audit["effective_processor_source"]).resolve() == effective_path.resolve(), "Visual audit refers to another processor")
    effective = read_json(effective_path)["processor_kwargs"]
    saved = read_json(processor_config(checkpoint))["processor_kwargs"]
    visual_keys = ("color_jitter_params", "random_rotation_angle", "crop_fraction", "image_crop_size", "image_target_size", "use_albumentations")
    visual = {key: effective.get(key) for key in visual_keys}
    require(audit.get("effective_visual_config") == visual, "Visual audit does not match actual processor")
    require(all(saved.get(key) == visual[key] for key in visual_keys), "Checkpoint and output visual processors differ")
    jitter = visual["color_jitter_params"]
    require(isinstance(jitter, dict) and all(jitter.get(key) == 0 for key in ("brightness", "contrast", "saturation", "hue")), "Color jitter remains enabled")
    for name in ("statistics.json", "embodiment_id.json"):
        require(read_json(processor_config(checkpoint).parent / name) == read_json(effective_path.parent / name), f"Checkpoint {name} differs from effective training processor")
    read_json(checkpoint / "config.json")
    weight_map = read_json(checkpoint / "model.safetensors.index.json")["weight_map"]
    require(isinstance(weight_map, dict) and bool(weight_map), "Empty checkpoint weight index")
    for shard_name in set(weight_map.values()):
        shard = checkpoint / shard_name
        require(shard.resolve().parent == checkpoint and shard.suffix == ".safetensors", "Invalid checkpoint shard path")
        size = shard.stat().st_size
        with shard.open("rb") as stream:
            header_prefix = stream.read(8)
            require(len(header_prefix) == 8, "Truncated safetensors header")
            header_size = struct.unpack("<Q", header_prefix)[0]
            require(2 <= header_size <= min(size - 8, 100_000_000), "Incomplete safetensors header")
            header = json.loads(stream.read(header_size))
        data_size = size - 8 - header_size
        for tensor, item in header.items():
            if tensor == "__metadata__":
                continue
            begin, end = item["data_offsets"]
            require(type(begin) is int and type(end) is int and 0 <= begin <= end <= data_size, "Truncated safetensors payload")
        require(all(tensor in header for tensor, mapped in weight_map.items() if mapped == shard_name), "Indexed tensor missing from safetensors shard")
    print("Completed checkpoint, tensor payload bounds and effective augmentation verified", flush=True)


if __name__ == "__main__":
    checkpoint, output = map(Path, sys.argv[1:3])
    started = wait_for_training(int(sys.argv[4]), output)
    verify_checkpoint(checkpoint, output, int(sys.argv[3]), started)
PY

# Cooperative lock plus a live GPU process check: never stop another job. Jobs
# outside this launcher must use the same lock to eliminate launch-time races.
gpu_lock="${GPU_LOCK_PATH:-/tmp/twingroot-n17-gpu-${gpu_id}.lock}"
exec 9>"${gpu_lock}"
flock -n 9 || { echo "Another GPU diagnostic holds ${gpu_lock}" >&2; exit 75; }
gpu_guard() {
  local free_mib active
  free_mib="$(nvidia-smi -i "${gpu_id}" --query-gpu=memory.free --format=csv,noheader,nounits)"
  free_mib="${free_mib//[[:space:]]/}"
  [[ "${free_mib}" =~ ^[0-9]+$ && "${free_mib}" -ge 16000 ]] || {
    echo "Not enough free GPU memory; existing jobs will not be stopped." >&2
    exit 75
  }
  active="$(nvidia-smi -i "${gpu_id}" --query-compute-apps=pid --format=csv,noheader,nounits)"
  [[ -z "${active//[[:space:]]/}" ]] || {
    echo "GPU compute processes still active (${active}); existing jobs will not be stopped." >&2; exit 75;
  }
}
gpu_guard
mkdir -p results/logs results/n17_factory
offline="results/logs/${run_id}_offline.json"
[[ -f "${train_demo}" && -f "${heldout_demo}" && -f "${action_stats}" ]] || {
  echo "Diagnostic demo or action statistics files missing" >&2; exit 2;
}
# A run prefix is immutable, including stderr, partial videos and prior audits.
if compgen -G "results/logs/${run_id}_*" >/dev/null || compgen -G "results/n17_factory/${run_id}_*" >/dev/null; then
  echo "Refusing to overwrite existing run prefix ${run_id}" >&2; exit 2
fi
set -o noclobber
prompt="$("${convert_python}" - "${train_demo}" "${heldout_demo}" "${action_stats}" \
  "${correct_legacy_jpeg_bgr}" "${POLICY_PROMPT:-}" "results/logs/${run_id}_inputs.json" <<'PY'
import json
import sys
from pathlib import Path


def resolve_diagnostic_config(train_demo, heldout_demo, action_stats, correct_legacy_jpeg_bgr, online_override=""):
    if correct_legacy_jpeg_bgr not in ("0", "1"):
        raise ValueError("CORRECT_LEGACY_JPEG_BGR must be exactly 0 or 1")
    paths = {"train_demo": Path(train_demo).resolve(), "heldout_demo": Path(heldout_demo).resolve(),
             "action_stats": Path(action_stats).resolve()}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing diagnostic {name}: {path}")
    prompts = {}
    for split in ("train", "heldout"):
        demo = paths[f"{split}_demo"]
        instruction_path = demo.parent.parent / "instructions" / f"{demo.stem}.json"
        seen = json.loads(instruction_path.read_text(encoding="utf-8")).get("seen")
        if not isinstance(seen, list) or not seen or not isinstance(seen[0], str) or not seen[0].strip():
            raise ValueError(f"Missing usable seen[0] instruction: {instruction_path}")
        prompts[split] = {"text": seen[0].strip(), "source": "source_seen_0", "instruction_path": str(instruction_path)}
    if online_override and not online_override.strip():
        raise ValueError("POLICY_PROMPT override must not be blank")
    online = {"text": online_override.strip(), "source": "POLICY_PROMPT_override"} if online_override else prompts["train"].copy()
    return {
        **{name: str(path) for name, path in paths.items()},
        "correct_legacy_jpeg_bgr": correct_legacy_jpeg_bgr == "1",
        "policy_camera_bgr_swap": False, "online_camera_encoding": "isaac_rgb",
        "observation_action_offset": 1, "offline_prompts": prompts, "online_prompt": online,
        "initial_state_split": "train", "status": "diagnostic_configuration_not_inference_success",
    }


if __name__ == "__main__":
    config = resolve_diagnostic_config(*sys.argv[1:6])
    with Path(sys.argv[6]).open("x", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2, ensure_ascii=False)
    print(config["online_prompt"]["text"])
PY
)"
"${groot_python}" scripts/diagnose_n17_offline.py \
  --checkpoint "${checkpoint}" --train-demo "${train_demo}" --heldout-demo "${heldout_demo}" \
  "${legacy_color_args[@]}" --observation-action-offset 1 \
  --flow-steps 8 --repeats 3 --seed 42 > "${offline}" 2> "${offline%.json}.stderr"

failed=0
for style in factory no-factory-dressing; do
  gpu_guard
  if ss -H -ltn "( sport = :${port} )" | grep -q .; then
    echo "Policy port ${port} is already occupied; will not connect to an unrelated service." >&2; exit 75
  fi
  output="${project_root}/results/n17_factory/${run_id}_${style}_train0.mp4"
  record_rc=0
  MODEL_PATH="${checkpoint}" GROOT_PYTHON="${groot_python}" POLICY_PORT="${port}" \
  ACTION_STATS="${action_stats}" \
  HELDOUT_DEMO="${train_demo}" INITIAL_STATE_SPLIT=train POLICY_PROMPT="${prompt}" \
  POLICY_CAMERA_BGR_SWAP=0 \
  ACTION_ALPHA=1 FLOW_STEPS=8 POLICY_SEED=42 SIM_SEED=42 SCENE_STYLE="${style}" \
  OUTPUT_VIDEO="${output}" bash scripts/record_aloha_n17_factory.sh \
    > "results/logs/${run_id}_${style}.log" 2>&1 || record_rc=$?
  verify_rc=0
  if [[ -f "${output%.mp4}.json" ]]; then
    "${convert_python}" scripts/verify_n17_recording.py \
      "${output%.mp4}.json" --allow-failed > "${output%.mp4}.verify.json" \
      2> "${output%.mp4}.verify.stderr" || verify_rc=$?
  else
    echo "Recording ${style} produced no manifest; not accepted." >&2
    verify_rc=1
  fi
  if (( record_rc != 0 || verify_rc != 0 )); then failed=1; fi
  printf '%s recorder_exit=%s verifier_exit=%s\n' "${style}" "${record_rc}" "${verify_rc}"
done
echo "Diagnostics finished. Inspect acceptance and complete video review; this is not a formal success declaration."
exit "${failed}"
