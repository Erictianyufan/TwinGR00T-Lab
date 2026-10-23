from __future__ import annotations

import importlib.util
import json
import sys
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from PIL import Image

from twingr00t_lab.conversion import _encoded_frame_bytes
from twingr00t_lab.dataset import audit_demo
from twingr00t_lab.teacher_collection import (
    ALIGNMENT,
    POLICY_CAMERAS,
    TeacherHdf5Writer,
    collection_metadata,
    encode_rgb_jpeg,
    named_eef_indices,
    replay_plan,
    rgb_uint8,
    sha256_file,
)


def source_fixture(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source" / "data" / "demo_0.hdf5"
    source.parent.mkdir(parents=True)
    with h5py.File(source, "w") as handle:
        handle.create_dataset("observations/policy/actions", data=np.zeros((3, 16), dtype=np.float32))
        handle.create_dataset("initial_state/articulation/robot/joint_position", data=np.zeros((1, 38)))
        for number in (1, 2, 3):
            handle.create_dataset(
                f"initial_state/rigid_object/block{number}/root_pose", data=np.zeros((1, 7), dtype=np.float32)
            )
    instruction = source.parent.parent / "instructions" / "demo_0.json"
    instruction.parent.mkdir()
    instruction.write_text(json.dumps({"seen": ["Sort red, green and blue"], "unseen": []}), encoding="utf-8")
    return source, instruction


def test_tail_is_real_command_plan_with_untouched_source_and_explicit_indices():
    source = np.zeros((3, 16), dtype=np.float32)
    source[:, :6] = np.arange(3)[:, None]
    source[:, 6:8] = np.asarray([0.048, -0.01, 0.048])[:, None]
    plan, indices = replay_plan(source, step_dt=0.016)
    assert plan.shape == (66, 16)
    np.testing.assert_array_equal(plan[:3], source)
    np.testing.assert_array_equal(plan[3:], np.repeat(source[-1:], 63, axis=0))
    np.testing.assert_array_equal(indices, [0, 1, 2] + [2] * 63)
    source[-1] = 999
    assert plan[-1, 0] == 2  # No reference to a mutable simulator/source buffer.


@pytest.mark.parametrize("dt,seconds", [(0, 1), (-1, 1), (np.nan, 1), (np.inf, 1), (0.016, 0.99), (0.016, np.nan)])
def test_replay_plan_rejects_invalid_duration(dt, seconds):
    with pytest.raises(ValueError):
        replay_plan(np.zeros((1, 16)), step_dt=dt, stable_tail_s=seconds)


def test_replay_plan_rejects_bad_actions():
    for source in (np.zeros((0, 16)), np.zeros((2, 14)), np.full((2, 16), np.nan)):
        with pytest.raises(ValueError):
            replay_plan(source, step_dt=0.016)
    source = np.zeros((2, 16))
    source[0, 7] = 1
    with pytest.raises(ValueError, match="gripper"):
        replay_plan(source, step_dt=0.016)


def test_rgb_jpeg_never_reverses_red_and_blue():
    rgb = np.zeros((24, 32, 3), dtype=np.uint8)
    rgb[..., 0] = 230
    rgb[..., 2] = 12
    encoded = encode_rgb_jpeg(rgb[None])
    image = np.asarray(Image.open(BytesIO(encoded.tobytes())).convert("RGB"))
    np.testing.assert_allclose(image.mean(axis=(0, 1)), [230, 0, 12], atol=3)
    np.testing.assert_array_equal(rgb_uint8(rgb.astype(np.float32) / 255), rgb)
    np.testing.assert_array_equal(rgb_uint8(np.concatenate((rgb, rgb[..., :1]), axis=-1)), rgb)


@pytest.mark.parametrize("frame", [np.full((4, 4, 3), 255.0), np.zeros((2, 4, 4, 3)), np.full((4, 4, 3), np.nan)])
def test_ambiguous_rgb_is_rejected(frame):
    with pytest.raises(ValueError):
        rgb_uint8(frame)


def test_named_eef_indices_reorders_and_refuses_missing_or_duplicate_names():
    assert named_eef_indices(["right_end_effector", "unrelated", "left_end_effector"]) == (2, 0)
    for names in (["left_end_effector"], ["left_end_effector", "left_end_effector", "right_end_effector"]):
        with pytest.raises(ValueError):
            named_eef_indices(names)


def test_collection_metadata_is_auditable_and_never_model_inference(tmp_path):
    source, instruction = source_fixture(tmp_path)
    report = collection_metadata(
        source_demo=source,
        source_instruction=instruction,
        episode_id=0,
        source_steps=3,
        total_steps=66,
        step_dt=0.016,
        cameras=POLICY_CAMERAS,
    )
    assert report["source_demo_sha256"] == sha256_file(source)
    assert report["source_instruction_sha256"] == report["instruction_sha256"] == sha256_file(instruction)
    assert report["controller_source"] == "teacher_replay_for_training"
    assert report["replay_used"] is True and report["model_inference_result"] is False
    assert report["training_eligible"] is False
    assert report["tail_steps"] == 63 and report["tail_duration_s"] >= 1
    assert report["control_fps"] == 62.5
    assert report["observation_action_alignment"] == ALIGNMENT
    assert report["recommended_observation_action_offset"] == 1
    assert report["camera_encoding"] == "jpeg_rgb_pillow" and report["correct_legacy_jpeg_bgr"] is False
    assert report["total_steps"] == 0  # Planned steps are not executed steps.


def test_writer_streams_fresh_post_step_rgb_and_actual_initial_state(tmp_path):
    source, instruction = source_fixture(tmp_path)
    report = collection_metadata(
        source_demo=source,
        source_instruction=instruction,
        episode_id=0,
        source_steps=3,
        total_steps=66,
        step_dt=0.016,
        cameras=POLICY_CAMERAS,
    )
    destination = tmp_path / "diagnostic.hdf5"
    actual_joint = np.arange(38, dtype=np.float32)
    writer = TeacherHdf5Writer(
        destination,
        source,
        report,
        {
            "articulation/robot/joint_position": actual_joint,
            "rigid_object/block1/root_pose": np.arange(7, dtype=np.float32),
        },
    )
    for index in range(2):
        state = np.arange(14, dtype=np.float32) + index
        values = {
            "observations/policy/actions": np.full(16, index, dtype=np.float32),
            "observations/policy/left_arm_pos": state[:6],
            "observations/policy/left_gripper_pos": state[6:7],
            "observations/policy/right_arm_pos": state[7:13],
            "observations/policy/right_gripper_pos": state[13:14],
            "observations/policy/left_eef_pos": np.asarray([1, 2, 3], dtype=np.float32),
            "observations/policy/right_eef_pos": np.asarray([4, 5, 6], dtype=np.float32),
            "evidence/source_action_index": np.int64(index),
            "evidence/is_stable_tail": np.bool_(False),
        }
        rgb = np.zeros((16, 16, 3), dtype=np.uint8)
        rgb[..., index * 2] = 200
        writer.append(values, dict.fromkeys(POLICY_CAMERAS, rgb))
        rgb[:] = 0
    writer.close()
    assert audit_demo(destination).frames == 2
    with h5py.File(destination, "r") as handle:
        np.testing.assert_array_equal(handle["initial_state/articulation/robot/joint_position"][0], actual_joint)
        assert handle["initial_state/rigid_object/block2/root_pose"].shape == (1, 7)
        assert not handle.attrs["success"] and handle.attrs["num_samples"] == 2
        assert not handle.attrs["correct_legacy_jpeg_bgr"]
        assert handle.attrs["recommended_observation_action_offset"] == 1
        np.testing.assert_array_equal(handle["observations/policy/right_eef_pos"][0], [4, 5, 6])
        dataset = handle["observations/camera_obs/head_camera_rgb"]
        assert dataset.dtype.kind == "O" and dataset.attrs["color_space"] == "RGB"
        first = np.asarray(Image.open(BytesIO(_encoded_frame_bytes(dataset[0]))))
        second = np.asarray(Image.open(BytesIO(_encoded_frame_bytes(dataset[1]))))
        assert first[..., 0].mean() > 190 and first[..., 2].mean() < 5
        assert second[..., 2].mean() > 190 and second[..., 0].mean() < 5
    with pytest.raises(FileExistsError):
        TeacherHdf5Writer(destination, source, report, {})


def test_writer_rejects_incomplete_or_nonfinite_measurements_before_appending(tmp_path):
    source, instruction = source_fixture(tmp_path)
    report = collection_metadata(
        source_demo=source,
        source_instruction=instruction,
        episode_id=0,
        source_steps=3,
        total_steps=66,
        step_dt=0.016,
        cameras=POLICY_CAMERAS,
    )
    writer = TeacherHdf5Writer(tmp_path / "bad.hdf5", source, report, {})
    cameras = dict.fromkeys(POLICY_CAMERAS, np.zeros((8, 8, 3), dtype=np.uint8))
    writer.append({"state": np.ones(14)}, cameras)
    for invalid in ({"state": np.full(14, np.nan)}, {"different_key": np.ones(14)}, {"state": np.ones(13)}):
        with pytest.raises(ValueError):
            writer.append(invalid, cameras)
    assert writer.rows == 1
    writer.close()


def test_scene_snapshot_uses_world_state_after_settle_and_copies_buffers():
    script = Path(__file__).resolve().parents[1] / "scripts/collect_aloha_factory_teacher.py"
    spec = importlib.util.spec_from_file_location("factory_collector_cpu_test", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    joint = np.arange(38, dtype=np.float32)[None]
    calls = []

    def get_state(*, is_relative):
        calls.append(is_relative)
        return {"articulation": {"robot": {"joint_position": joint, "root_pose": np.zeros((1, 7))}}}

    result = module.scene_snapshot(SimpleNamespace(scene=SimpleNamespace(get_state=get_state)))
    assert calls == [False]
    joint[:] = 999
    np.testing.assert_array_equal(result["articulation/robot/joint_position"], np.arange(38))


@pytest.mark.parametrize("physics_pass,termination_step", [(True, None), (False, None), (True, 2)])
def test_collection_promotion_requires_completed_real_steps_and_physics_gate(
    tmp_path, monkeypatch, physics_pass, termination_step
):
    """Test the CPU orchestration with a fake simulator, not robot physics."""
    source, _ = source_fixture(tmp_path)
    output = tmp_path / "collection"
    for name in ("data", "diagnostics", "reports", "instructions"):
        (output / name).mkdir(parents=True)
    spec = importlib.util.spec_from_file_location(
        "factory_collector_orchestration_test",
        Path(__file__).resolve().parents[1] / "scripts/collect_aloha_factory_teacher.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frame = SimpleNamespace(
        target_frame_names=["right_end_effector", "left_end_effector"],
        target_pos_w=np.zeros((1, 2, 3), dtype=np.float32),
        target_quat_w=np.tile([1, 0, 0, 0], (1, 2, 1)).astype(np.float32),
    )

    class FakeScene(dict):
        def __init__(self, env):
            super().__init__(ee_frame=SimpleNamespace(data=frame))
            self.env = env
            self.rigid_objects = dict.fromkeys(["block1", "block2", "block3"])

        def get_state(self, *, is_relative):
            assert is_relative is False
            joint = np.full((1, 38), self.env.tick, dtype=np.float32)
            return {
                "articulation": {
                    "robot": {
                        "joint_position": joint,
                        "joint_velocity": np.zeros_like(joint),
                        "root_pose": np.zeros((1, 7)),
                        "root_velocity": np.zeros((1, 6)),
                    }
                },
                "rigid_object": {
                    name: {
                        "root_pose": np.asarray([[index, self.env.tick / 1000, 0.765, 1, 0, 0, 0]]),
                        "root_velocity": np.zeros((1, 6)),
                    }
                    for index, name in enumerate(self.rigid_objects)
                },
            }

    class FakeEnv:
        device = "cpu"
        num_envs = 1
        step_dt = 0.016

        def __init__(self):
            self.tick = 0
            self.scene = FakeScene(self)
            self.commands = []

        def observation(self):
            image = np.zeros((1, 8, 8, 3), dtype=np.uint8)
            image[..., 0] = self.tick
            return {
                "policy": np.full(14, self.tick, dtype=np.float32),
                "camera_obs": dict.fromkeys(POLICY_CAMERAS, image),
            }

        def reset(self):
            self.tick = 0

        def step(self, action):
            self.commands.append(np.asarray(action).copy())
            self.tick += 1
            frame.target_pos_w[0, 0] = [20 + self.tick, 0, 0]
            frame.target_pos_w[0, 1] = [10 + self.tick, 0, 0]
            return self.observation(), 0, self.tick == termination_step, False, {}

    def read_demo(path, device):
        with h5py.File(path, "r") as handle:
            return (
                handle["observations/policy/actions"][:],
                np.zeros((1, 38)),
                {name: np.zeros((1, 7)) for name in ["block1", "block2", "block3"]},
            )

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            as_tensor=lambda value, device: SimpleNamespace(unsqueeze=lambda axis: np.asarray(value)[None])
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "replay_live_multiview",
        SimpleNamespace(read_demo=read_demo, teleport=lambda env, joint, objects: env.observation()),
    )
    monkeypatch.setitem(
        sys.modules,
        "run_aloha_n17_closedloop",
        SimpleNamespace(measured_state=lambda value: value.copy(), success_from_term=lambda env, term: True),
    )
    gate_calls = []

    def gate(**kwargs):
        assert "config" not in kwargs  # Must always use official defaults.
        gate_calls.append(kwargs)
        return {"success": physics_pass}

    monkeypatch.setattr(module, "evaluate_grasp_sorting", gate)
    env = FakeEnv()
    args = SimpleNamespace(
        output_dir=output, stable_tail_seconds=1.0, jpeg_quality=95, seed=42, environment="factory-test"
    )
    report = module.collect_episode(env, source, args, object(), POLICY_CAMERAS)
    accepted = physics_pass and termination_step is None
    assert report["training_eligible"] is accepted
    assert report["model_inference_result"] is False and report["replay_used"] is True
    assert bool(list((output / "data").glob("*.hdf5"))) is accepted
    assert bool(list((output / "instructions").glob("*.json"))) is accepted
    destination = output / report["hdf5"]
    assert report["hdf5_sha256"] == sha256_file(destination)
    with h5py.File(destination, "r") as handle:
        assert bool(handle.attrs["success"]) is accepted
        np.testing.assert_array_equal(handle["observations/policy/left_eef_pos"][0], [11, 0, 0])
        np.testing.assert_array_equal(handle["observations/policy/right_eef_pos"][0], [21, 0, 0])
        if termination_step is not None:
            assert report["status"] == "error" and report["total_steps"] == 1
            assert not gate_calls
        else:
            assert len(env.commands) == report["total_steps"] == 66
            np.testing.assert_array_equal(handle["evidence/source_action_index"][:], [0, 1, 2] + [2] * 63)
            np.testing.assert_array_equal(handle["evidence/is_stable_tail"][:], [False] * 3 + [True] * 63)
            np.testing.assert_array_equal(handle["states/articulation/robot/joint_position"][:, 0], np.arange(1, 67))
            assert gate_calls[0]["measured_states"][-1, 0] == 66
            assert handle["evidence/timestamp_s"][-1] == pytest.approx(66 * 0.016)
