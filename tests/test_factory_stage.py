"""Stage-4 fail-closed data gates, without Torch, GPU jobs, or remote access."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from twingr00t_lab.grasp_acceptance import evaluate_grasp_sorting
from twingr00t_lab.teacher_collection import (
    POLICY_CAMERAS,
    TeacherHdf5Writer,
    collection_metadata,
    sha256_file,
)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/train_n17_factory_stage.sh"


@pytest.fixture
def guards():
    namespace = {"__name__": "factory_stage_guard_test"}
    parts = SCRIPT.read_text(encoding="utf-8").split("<<'PY'\n")
    for part in parts[1:3]:
        exec(compile(part.split("\nPY\n", 1)[0], str(SCRIPT), "exec"), namespace)
    return namespace


@pytest.fixture(scope="module")
def accepted_template(tmp_path_factory):
    """Use the actual collector writer; synthetic states only test the contract."""
    root = tmp_path_factory.mktemp("factory_teacher_contract")
    source = root / "source"
    collected = root / "collected"
    for parent, directories in ((source, ("data", "instructions")), (collected, ("data", "instructions", "reports"))):
        for directory in directories:
            (parent / directory).mkdir(parents=True)
    initial = np.asarray([[0.2, 0.1, 0.765], [-0.2, 0.1, 0.765], [0, 0.1, 0.765]])
    steps, source_steps = 163, 100
    positions = np.tile(initial, (steps, 1, 1))
    eef = np.full((steps, 2, 3), 10.0)
    state = np.zeros((steps, 14), dtype=np.float32)
    state[:source_steps, 0] = state[:source_steps, 7] = np.linspace(0, 0.2, source_steps)
    state[source_steps:, (0, 7)] = 0.2
    state[:, (6, 13)] = 0.048
    for block, (start, arm) in enumerate(((10, 0), (30, 1), (50, 0))):
        positions[start:, block] = [-0.08 + block * 0.08, 0, 0.765]
        positions[start : start + 10, block, 2] += 0.05
        eef[start : start + 10, arm] = positions[start : start + 10, block] + [0, 0, 0.04]
        state[start : start + 10, (6, 13)[arm]] = 0.02
    actions = np.concatenate(
        (state[:, :6], state[:, 6:7], state[:, 6:7], state[:, 7:13], state[:, 13:], state[:, 13:]), axis=1
    )
    velocities = np.diff(positions, axis=0, prepend=initial[None]) / 0.016
    delta = positions[:, 1:, :2] - positions[:, :-1, :2]
    sorting = np.all((delta[:, :, 0] > 0) & (delta[:, :, 0] < 0.13) & (abs(delta[:, :, 1]) < 0.03), axis=1)
    original = source / "data/demo_0.hdf5"
    with h5py.File(original, "w") as handle:
        handle.create_dataset("observations/policy/actions", data=actions[:source_steps])
        for block in range(3):
            handle.create_dataset(
                f"initial_state/rigid_object/block{block + 1}/root_pose", data=np.r_[initial[block], 1, 0, 0, 0][None]
            )
    instruction = source / "instructions/demo_0.json"
    instruction.write_text(json.dumps({"seen": ["Sort all three blocks using both arms"]}), encoding="utf-8")
    shutil.copyfile(instruction, collected / "instructions/demo_0.json")
    report = collection_metadata(
        source_demo=original,
        source_instruction=instruction,
        episode_id=0,
        source_steps=source_steps,
        total_steps=steps,
        step_dt=0.016,
        cameras=POLICY_CAMERAS,
    )
    demo = collected / "data/demo_0.hdf5"
    writer = TeacherHdf5Writer(demo, original, report, {})
    rgb = np.full((8, 8, 3), [230, 12, 25], dtype=np.uint8)
    for index in range(steps):
        values = {
            "observations/policy/actions": actions[index],
            "observations/policy/left_arm_pos": state[index, :6],
            "observations/policy/left_gripper_pos": state[index, 6:7],
            "observations/policy/right_arm_pos": state[index, 7:13],
            "observations/policy/right_gripper_pos": state[index, 13:14],
            "observations/policy/left_eef_pos": eef[index, 0],
            "observations/policy/right_eef_pos": eef[index, 1],
            "evidence/source_action_index": np.int64(min(index, source_steps - 1)),
            "evidence/is_stable_tail": np.bool_(index >= source_steps),
            "evidence/sorting_predicate": sorting[index],
            "evidence/control_step": np.int64(index + 1),
            "evidence/timestamp_s": (index + 1) * 0.016,
        }
        for block in range(3):
            values[f"states/rigid_object/block{block + 1}/root_pose"] = np.r_[positions[index, block], 1, 0, 0, 0]
            values[f"states/rigid_object/block{block + 1}/root_velocity"] = np.r_[velocities[index, block], 0, 0, 0]
        writer.append(values, dict.fromkeys(POLICY_CAMERAS, rgb))
    writer.close(accepted=True)
    report.update(
        training_eligible=True,
        status="accepted_for_training",
        hdf5="data/demo_0.hdf5",
        total_steps=steps,
        hdf5_sha256=sha256_file(demo),
        acceptance=evaluate_grasp_sorting(
            initial_block_positions=initial,
            block_positions=positions,
            block_linear_velocities=velocities,
            block_angular_velocities=np.zeros_like(velocities),
            eef_positions=eef,
            measured_states=state,
            sorting_predicate=sorting,
            step_dt=0.016,
        ),
    )
    assert report["acceptance"]["success"]
    (collected / "reports/demo_0.json").write_text(json.dumps(report), encoding="utf-8")
    return root


@pytest.fixture
def accepted(accepted_template, tmp_path):
    shutil.copytree(accepted_template, tmp_path, dirs_exist_ok=True)
    return tmp_path / "collected", tmp_path / "source"


def test_actual_collector_output_is_validated_without_training(guards, accepted):
    result = guards["validate_episode"](*accepted, 0)
    assert result["frames"] == 163 and result["aligned_frames"] == 162
    assert result["recomputed_acceptance"]["object_grasped"] == [True, True, True]
    assert result["recomputed_acceptance"]["both_arms_grasp_evidence"] is True


@pytest.mark.parametrize(
    "field,value",
    [
        ("training_eligible", False),
        ("status", "rejected_physics"),
        ("scene_style", "no-factory-dressing"),
        ("control_fps", 50),
        ("step_dt", 0.02),
        ("correct_legacy_jpeg_bgr", True),
        ("source_episode_id", 1),
        ("action_dim", 14),
        ("state_dim", 16),
        ("observation_action_alignment", "pre_step"),
        ("source_demo_sha256", "bad"),
        ("source_instruction_sha256", "bad"),
        ("hdf5_sha256", "bad"),
        ("model_inference_result", True),
    ],
)
def test_report_contract_failures_are_rejected(guards, accepted, field, value):
    collected, source = accepted
    path = collected / "reports/demo_0.json"
    report = json.loads(path.read_text())
    report[field] = value
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        guards["validate_episode"](collected, source, 0)


@pytest.mark.parametrize("failure", ["actions", "grippers", "timeline", "tail", "eef", "physics", "camera", "weakened"])
def test_success_flags_and_matching_hash_cannot_hide_bad_physical_evidence(guards, accepted, failure):
    collected, source = accepted
    path = collected / "data/demo_0.hdf5"
    report_path = collected / "reports/demo_0.json"
    report = json.loads(report_path.read_text())
    with h5py.File(path, "r+") as handle:
        if failure == "actions":
            handle["observations/policy/actions"][3, 0] += 0.1
        elif failure == "grippers":
            handle["observations/policy/actions"][3, 7] += 0.1
        elif failure == "timeline":
            handle["evidence/timestamp_s"][4] = 0.02
        elif failure == "tail":
            handle["evidence/is_stable_tail"][100] = False
        elif failure == "eef":
            handle["observations/policy/right_eef_pos"][:] = handle["observations/policy/left_eef_pos"][:]
        elif failure == "physics":
            handle["states/rigid_object/block1/root_pose"][:, 2] = 0.765
        elif failure == "camera":
            handle["observations/camera_obs/head_camera_rgb"].attrs["color_space"] = "BGR"
        else:
            report["acceptance"]["config"]["min_lift_m"] = 0.001
    report["hdf5_sha256"] = sha256_file(path)
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        guards["validate_episode"](collected, source, 0)


def test_all_nine_training_episodes_required_and_no_extra_accepted(guards, accepted):
    collected, source = accepted
    with pytest.raises(ValueError, match="exactly the same nine"):
        guards["validate_collection"](collected, source)
    for index in range(1, 10):
        shutil.copyfile(collected / "data/demo_0.hdf5", collected / f"data/demo_{index}.hdf5")
    with pytest.raises(ValueError, match="exactly the same nine"):
        guards["validate_collection"](collected, source)


@pytest.fixture
def prepared(tmp_path):
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text(json.dumps({"fps": 62.5, "total_episodes": 9, "total_frames": 1458}))
    Path(str(root) + ".conversion.json").write_text(
        json.dumps(
            {"source_episode_ids": list(range(9)), "observation_action_offset": 1, "correct_legacy_jpeg_bgr": False}
        )
    )
    Path(str(root) + ".source_audit.json").write_text(json.dumps({"episodes": [{"aligned_frames": 162}] * 9}))
    for filename, keys, shape in (
        ("stats.json", ("action", "observation.state"), (14,)),
        ("relative_stats.json", ("left_arm", "right_arm"), (16, 6)),
    ):
        (root / "meta" / filename).write_text(
            json.dumps(
                {
                    key: {stat: np.zeros(shape).tolist() for stat in ("mean", "std", "min", "max", "q01", "q99")}
                    for key in keys
                }
            )
        )
    return root


def test_fresh_statistics_match_relative_horizon_and_logical_dimensions(guards, prepared):
    guards["validate_prepared_dataset"](prepared)


@pytest.mark.parametrize("failure", ["fps", "frames", "split", "offset", "bgr", "absolute", "relative", "nan"])
def test_bad_prepared_contract_never_reaches_training(guards, prepared, failure):
    if failure in ("fps", "frames"):
        path = prepared / "meta/info.json"
        value = json.loads(path.read_text())
        value["fps" if failure == "fps" else "total_frames"] = 50
    elif failure in ("split", "offset", "bgr"):
        path = Path(str(prepared) + ".conversion.json")
        value = json.loads(path.read_text())
        value.update(
            {
                "split": {"source_episode_ids": list(range(1, 10))},
                "offset": {"observation_action_offset": 0},
                "bgr": {"correct_legacy_jpeg_bgr": True},
            }[failure]
        )
    else:
        path = prepared / "meta" / ("stats.json" if failure == "absolute" else "relative_stats.json")
        value = json.loads(path.read_text())
        key = "action" if failure == "absolute" else "left_arm"
        value[key]["mean"] = [float("nan")] * 6 if failure == "nan" else [0] * 16
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        guards["validate_prepared_dataset"](prepared)


def test_wrapper_uses_new_rgb_actual_stats_and_noninterrupting_resource_guards():
    text = SCRIPT.read_text(encoding="utf-8")
    conversion = text.split('"${convert_python}" -m twingr00t_lab.conversion', 1)[1].split("\n(", 1)[0]
    assert "--correct-legacy-jpeg-bgr" not in conversion
    assert "--fps 62.5 --observation-action-offset 1" in conversion
    assert "--start-episode 0 --limit 9" in conversion
    assert 'gr00t/data/stats.py --dataset-path "${dataset}"' in text
    assert "--save-total-limit 2" in text
    assert "free < 40 * 1024**3" in text
    assert "--query-compute-apps=pid" in text and "flock -n 9" in text
    assert "kill " not in text
    assert "run_n17_posttrain_checks.sh" not in text
