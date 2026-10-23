#!/usr/bin/env bash
# Stage 4: validated factory teacher recollection -> fresh statistics -> N1.7.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
groot_root="${GROOT_ROOT:-/home/ubuntu/mycode/robotwin-gr00t-lab/vendor/Isaac-GR00T-N1.7-src}"
convert_python="${CONVERT_PYTHON:-/home/ubuntu/env_convert/bin/python}"
groot_python="${GROOT_PYTHON:-/home/ubuntu/env_groot/bin/python}"
input="${INPUT_DIR:-${project_root}/data/replayed/blocks_ranking_rgb-factory-v1}"
source="${SOURCE_DIR:-${project_root}/data/replayed/blocks_ranking_rgb-vision-batch}"
dataset="${DATASET_PATH:-${project_root}/data/lerobot/blocks_ranking_rgb-factory-aligned-v1}"
output="${OUTPUT_DIR:-${project_root}/artifacts/checkpoints/n17-factory-v4}"
base="${BASE_MODEL_PATH:-${project_root}/artifacts/checkpoints/n17-aligned-v3-coloroff/checkpoint-4000}"
prepare_only="${PREPARE_ONLY:-0}"
case "${prepare_only}" in 0|1) ;; *) echo "PREPARE_ONLY must be 0 or 1" >&2; exit 2 ;; esac
if [[ "${FPS:-62.5}" != 62.5 || "${OBSERVATION_ACTION_OFFSET:-1}" != 1 ||
      "${CORRECT_LEGACY_JPEG_BGR:-0}" != 0 || "${POLICY_CAMERA_BGR_SWAP:-0}" != 0 ]]; then
  echo "Factory recollection requires 62.5 Hz, offset 1, and RGB JPEG without legacy BGR swap." >&2
  exit 2
fi
for executable in "${convert_python}" "${groot_python}"; do
  [[ -x "${executable}" ]] || { echo "Missing Python: ${executable}" >&2; exit 2; }
done
cd "${project_root}"
# Stats runs from the upstream checkout, so all filesystem inputs must be absolute.
input="$(realpath -m -- "${input}")"
source="$(realpath -m -- "${source}")"
dataset="$(realpath -m -- "${dataset}")"
output="$(realpath -m -- "${output}")"
base="$(realpath -m -- "${base}")"
groot_root="$(realpath -m -- "${groot_root}")"
[[ -f "${groot_root}/gr00t/data/stats.py" ]] || { echo "Pinned upstream stats.py is missing" >&2; exit 2; }
export PYTHONPATH="${project_root}/src:${groot_root}${PYTHONPATH:+:${PYTHONPATH}}"
export NO_ALBUMENTATIONS_UPDATE=1 TOKENIZERS_PARALLELISM=false
cd "${project_root}"

# Keep this CPU guard self-contained so it can be unit-tested without Isaac/Torch.
"${convert_python}" - "${input}" "${source}" "${dataset}" "${output}" "${base}" <<'PY'
from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import asdict
from io import BytesIO
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from twingr00t_lab.dataset import CAMERAS, canonical_proprio, collapse_isaac_action
from twingr00t_lab.grasp_acceptance import GraspAcceptanceConfig, evaluate_grasp_sorting


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def validate_episode(input_dir, source_dir, episode_id):
    name = f"demo_{episode_id}"
    report_path = input_dir / "reports" / f"{name}.json"
    report = read_json(report_path)
    fixed = {
        "schema_version": 1,
        "controller_source": "teacher_replay_for_training",
        "replay_used": True,
        "training_eligible": True,
        "status": "accepted_for_training",
        "scene_style": "factory",
        "model_inference_result": False,
        "source_episode_id": episode_id,
        "hdf5": f"data/{name}.hdf5",
        "action_dim": 16,
        "state_dim": 14,
        "observation_action_alignment": "post_step_obs_with_executed_action",
        "recommended_observation_action_offset": 1,
        "camera_encoding": "jpeg_rgb_pillow",
        "correct_legacy_jpeg_bgr": False,
    }
    for key, expected in fixed.items():
        require(type(report.get(key)) is type(expected) and report[key] == expected,
                f"{name}: invalid {key}; expected {expected!r}")
    cameras = report.get("cameras", [])
    require(isinstance(cameras, list) and len(set(cameras)) == len(cameras) and set(CAMERAS).issubset(cameras),
            f"{name}: three unique policy cameras required")
    for key, expected in (("control_fps", 62.5), ("step_dt", 0.016)):
        value = report.get(key)
        require(type(value) in (int, float) and math.isfinite(value) and abs(value - expected) < 1e-9,
                f"{name}: invalid actual {key}")
    source_demo = source_dir / "data" / f"{name}.hdf5"
    source_instruction = source_dir / "instructions" / f"{name}.json"
    demo = input_dir / "data" / f"{name}.hdf5"
    instruction = input_dir / "instructions" / f"{name}.json"
    for path, field in ((source_demo, "source_demo_sha256"), (source_instruction, "source_instruction_sha256"),
                        (demo, "hdf5_sha256"), (instruction, "instruction_sha256")):
        require(path.is_file() and sha256(path) == report.get(field), f"{name}: {field} content mismatch")
    require(report["source_instruction_sha256"] == report["instruction_sha256"],
            f"{name}: source instruction must be copied unchanged")
    seen = read_json(instruction).get("seen")
    require(isinstance(seen, list) and seen and isinstance(seen[0], str) and seen[0].strip(),
            f"{name}: no usable seen[0] instruction")
    acceptance = report.get("acceptance", {})
    require(acceptance.get("config") == asdict(GraspAcceptanceConfig()), f"{name}: weakened/unknown acceptance config")
    for key in ("success", "all_objects_grasped", "both_arms_grasp_evidence", "stable_placement"):
        require(acceptance.get(key) is True, f"{name}: physical acceptance {key} did not pass")

    with h5py.File(source_demo, "r") as original, h5py.File(demo, "r") as handle:
        source_actions = np.asarray(original["observations/policy/actions"][:], dtype=np.float32)
        collapse_isaac_action(source_actions)
        source_steps = len(source_actions)
        policy = handle["observations/policy"]
        actions = np.asarray(policy["actions"][:], dtype=np.float32)
        logical_actions = collapse_isaac_action(actions)
        state = canonical_proprio(policy)
        steps = len(actions)
        tail_steps = steps - source_steps
        require(source_steps >= 16 and tail_steps >= math.ceil(1.0 / 0.016), f"{name}: incomplete source/tail")
        require(state.shape == logical_actions.shape == (steps, 14), f"{name}: state/action frame mismatch")
        for key, expected in (("source_action_steps", source_steps), ("tail_steps", tail_steps), ("total_steps", steps)):
            require(type(report.get(key)) is int and report[key] == expected, f"{name}: {key} mismatch")
        require(abs(report.get("tail_duration_s", -1) - tail_steps * 0.016) < 1e-9,
                f"{name}: tail duration mismatch")
        require(np.array_equal(actions[:source_steps], source_actions), f"{name}: source targets changed/truncated")
        require(np.array_equal(actions[source_steps:], np.repeat(source_actions[-1:], tail_steps, axis=0)),
                f"{name}: tail must hold the final source action")
        require(np.array_equal(handle["evidence/source_action_index"][:],
                               np.r_[np.arange(source_steps), np.full(tail_steps, source_steps - 1)]),
                f"{name}: source action chronology mismatch")
        require(np.array_equal(handle["evidence/is_stable_tail"][:],
                               np.r_[np.zeros(source_steps, dtype=bool), np.ones(tail_steps, dtype=bool)]),
                f"{name}: tail chronology mismatch")
        require(np.array_equal(handle["evidence/control_step"][:], np.arange(1, steps + 1)),
                f"{name}: missing/duplicated control steps")
        timestamps = handle["evidence/timestamp_s"][:]
        require(timestamps.shape == (steps,) and np.allclose(timestamps, np.arange(1, steps + 1) * 0.016,
                                                           rtol=0, atol=1e-9),
                f"{name}: actual frame timestamps are not 62.5 Hz")
        for key in ("controller_source", "replay_used", "model_inference_result", "source_episode_id",
                    "source_demo_sha256", "scene_style", "control_fps", "step_dt", "camera_encoding",
                    "correct_legacy_jpeg_bgr", "observation_action_alignment",
                    "recommended_observation_action_offset"):
            require(key in handle.attrs and handle.attrs[key] == report[key], f"{name}: HDF5 {key} mismatch")
        require(isinstance(handle.attrs.get("success"), (bool, np.bool_)) and bool(handle.attrs["success"]),
                f"{name}: HDF5 success flag missing")
        require(handle.attrs.get("num_samples") == steps, f"{name}: HDF5 num_samples mismatch")
        for camera in cameras:
            frames = handle[f"observations/camera_obs/{camera}"]
            require(len(frames) == steps, f"{name}: camera frame mismatch for {camera}")
            require(frames.attrs.get("encoding") == "jpeg_rgb_pillow" and frames.attrs.get("color_space") == "RGB"
                    and frames.attrs.get("correct_legacy_jpeg_bgr") == False,
                    f"{name}: camera encoding metadata mismatch for {camera}")
            for index in (0, steps - 1):
                with Image.open(BytesIO(bytes(frames[index]))) as image:
                    image.load()
                    require(image.format == "JPEG" and image.mode == "RGB", f"{name}: invalid RGB JPEG {camera}")
        block_names = ["block1", "block2", "block3"]
        initial = np.stack([handle[f"initial_state/rigid_object/{key}/root_pose"][0, :3] for key in block_names])
        positions = np.stack([handle[f"states/rigid_object/{key}/root_pose"][:, :3] for key in block_names], axis=1)
        velocities = np.stack([handle[f"states/rigid_object/{key}/root_velocity"][:] for key in block_names], axis=1)
        eef = np.stack([policy[f"{arm}_eef_pos"][:] for arm in ("left", "right")], axis=1)
        require(not np.array_equal(eef[:, 0], eef[:, 1]), f"{name}: duplicated left/right EEF evidence")
        actual_acceptance = evaluate_grasp_sorting(
            initial_block_positions=initial, block_positions=positions,
            block_linear_velocities=velocities[:, :, :3], block_angular_velocities=velocities[:, :, 3:],
            eef_positions=eef, measured_states=state,
            sorting_predicate=handle["evidence/sorting_predicate"][:], step_dt=0.016,
        )
        require(actual_acceptance["success"], f"{name}: independently recomputed physical acceptance failed")
    return {
        "source_episode_id": episode_id, "source_demo_sha256": report["source_demo_sha256"],
        "hdf5_sha256": report["hdf5_sha256"], "instruction_sha256": report["instruction_sha256"],
        "report_sha256": sha256(report_path), "frames": steps, "aligned_frames": steps - 1,
        "recomputed_acceptance": actual_acceptance,
    }


def validate_collection(input_dir, source_dir):
    expected = {f"demo_{index}.hdf5" for index in range(9)}
    actual = {path.name for path in (input_dir / "data").glob("demo_*.hdf5")}
    require(actual == expected, "Need exactly the same nine source episodes 0-8; no failed/missing/extra episodes")
    episodes = [validate_episode(input_dir, source_dir, index) for index in range(9)]
    return {
        "status": "validated_factory_training_data", "controller_source": "teacher_replay_for_training",
        "model_inference_result": False, "source_episode_ids": list(range(9)), "episodes": episodes,
        "fps": 62.5, "observation_action_offset": 1, "correct_legacy_jpeg_bgr": False,
        "scope": "complete three-block bimanual episodes; same training split, not generalization evidence",
    }


def main():
    input_dir, source_dir, dataset, output, base = [Path(value).resolve() for value in sys.argv[1:]]
    for path in (dataset, output, Path(str(dataset) + ".source_audit.json"), Path(str(dataset) + ".conversion.json")):
        require(not path.exists(), f"Refusing to overwrite/reuse existing output: {path}")
    require(base.is_dir() and (base / "config.json").is_file(), "Starting checkpoint is missing")
    require(read_json(base / "config.json").get("model_type") == "Gr00tN1d7", "Starting checkpoint is not N1.7")
    audit = validate_collection(input_dir, source_dir)
    dataset.parent.mkdir(parents=True, exist_ok=True)
    audit_path = Path(str(dataset) + ".source_audit.json")
    with audit_path.open("x", encoding="utf-8") as stream:
        json.dump(audit, stream, indent=2)
    print(json.dumps({"source_audit": str(audit_path), "episodes": 9, "status": audit["status"]}), flush=True)


if __name__ == "__main__":
    main()
PY

# New Pillow JPEGs already contain RGB; never pass --correct-legacy-jpeg-bgr.
"${convert_python}" -m twingr00t_lab.conversion \
  --input-dir "${input}" --output-dir "${dataset}" --start-episode 0 --limit 9 \
  --fps 62.5 --observation-action-offset 1 --manifest "${dataset}.conversion.json"
(
  cd "${groot_root}"
  "${groot_python}" gr00t/data/stats.py --dataset-path "${dataset}" \
    --embodiment-tag NEW_EMBODIMENT --modality-config-path "${project_root}/configs/groot_robotwin_aloha.py"
)
"${convert_python}" - "${dataset}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

def validate_prepared_dataset(root):
    info = json.loads((root / "meta/info.json").read_text())
    conversion = json.loads(Path(str(root) + ".conversion.json").read_text())
    audit = json.loads(Path(str(root) + ".source_audit.json").read_text())
    if info["fps"] != 62.5 or info["total_episodes"] != 9:
        raise ValueError("Factory dataset must retain nine episodes at 62.5 Hz")
    if conversion["source_episode_ids"] != list(range(9)):
        raise ValueError("Factory conversion changed the training split")
    if conversion["observation_action_offset"] != 1 or conversion["correct_legacy_jpeg_bgr"] is not False:
        raise ValueError("Factory conversion has wrong observation/action or RGB contract")
    if info["total_frames"] != sum(episode["aligned_frames"] for episode in audit["episodes"]):
        raise ValueError("Factory conversion dropped unexpected frames")
    absolute = json.loads((root / "meta/stats.json").read_text())
    relative = json.loads((root / "meta/relative_stats.json").read_text())
    for source, keys, shape in ((absolute, ("action", "observation.state"), (14,)),
                               (relative, ("left_arm", "right_arm"), (16, 6))):
        for key in keys:
            for statistic in ("mean", "std", "min", "max", "q01", "q99"):
                value = np.asarray(source[key][statistic])
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError(f"Invalid statistics {key}/{statistic}: expected {shape}")
    print("Factory RGB next-action dataset and newly generated absolute/relative statistics verified.")


if __name__ == "__main__":
    validate_prepared_dataset(Path(sys.argv[1]))
PY
if [[ "${prepare_only}" == 1 ]]; then
  echo "PREPARE_ONLY: validated/converted data and statistics; no training or inference success claimed."
  exit 0
fi
# Do not interrupt another project or race its cooperating GPU launcher.
gpu_id="${GPU_ID:-0}"
[[ "${gpu_id}" =~ ^[0-9]+$ ]] || { echo "GPU_ID must be a numeric device index" >&2; exit 2; }
command -v flock >/dev/null || { echo "flock is required for the cooperative GPU guard" >&2; exit 2; }
exec 9>"/tmp/twingroot-n17-gpu-${gpu_id}.lock"
flock -n 9 || { echo "Another N1.7 operation holds the GPU lock; no jobs were stopped." >&2; exit 75; }
"${convert_python}" - "${output}" <<'PY'
import shutil
import sys
from pathlib import Path

path = Path(sys.argv[1]).resolve().parent
while not path.exists():
    path = path.parent
free = shutil.disk_usage(path).free
if free < 40 * 1024**3:
    raise SystemExit(f"Stage 4 needs at least 40 GiB free before training; currently {free / 1024**3:.2f} GiB")
PY
active="$(nvidia-smi -i "${gpu_id}" --query-compute-apps=pid --format=csv,noheader,nounits)"
[[ -z "${active//[[:space:]]/}" ]] || {
  echo "GPU ${gpu_id} has active compute processes (${active}); no jobs were stopped." >&2; exit 75;
}
export CUDA_VISIBLE_DEVICES="${gpu_id}"
"${groot_python}" scripts/launch_aloha_n17_lowmem.py \
  --dataset-path "${dataset}" --output-dir "${output}" --base-model-path "${base}" \
  --modality-config-path configs/groot_robotwin_aloha.py \
  --max-steps "${MAX_STEPS:-4000}" --save-steps "${SAVE_STEPS:-1000}" --save-total-limit 2 \
  --batch-size 1 --gradient-accumulation-steps 8 --optimizer adafactor --min-free-mib 22000 \
  --state-dropout-prob 0 --learning-rate "${LEARNING_RATE:-0.00005}" --visual-augmentation color-off
