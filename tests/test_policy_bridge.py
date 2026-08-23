import numpy as np
import pytest

from twingr00t_lab.policy_bridge import build_groot_observation, groot_action_to_isaac


def test_build_groot_observation_shapes() -> None:
    policy = {
        "left_arm_pos": np.zeros((2, 6), dtype=np.float32),
        "left_gripper_pos": np.zeros((2, 1), dtype=np.float32),
        "right_arm_pos": np.zeros((2, 6), dtype=np.float32),
        "right_gripper_pos": np.zeros((2, 1), dtype=np.float32),
    }
    cameras = {
        "head_camera_rgb": np.zeros((2, 8, 10, 3), dtype=np.uint8),
        "left_camera_rgb": np.zeros((2, 8, 10, 3), dtype=np.uint8),
        "right_camera_rgb": np.zeros((2, 8, 10, 3), dtype=np.uint8),
    }
    observation = build_groot_observation(policy, cameras, "stack all three bowls")
    assert observation["video"]["head"].shape == (2, 1, 8, 10, 3)
    assert observation["state"]["left_arm"].shape == (2, 1, 6)
    assert observation["language"]["annotation.human.task_description"] == [
        ["stack all three bowls"],
        ["stack all three bowls"],
    ]


def test_groot_action_expands_grippers() -> None:
    action = {
        "left_arm": np.arange(12, dtype=np.float32).reshape(1, 2, 6),
        "left_gripper": np.asarray([[[0.1], [0.2]]], dtype=np.float32),
        "right_arm": np.arange(12, 24, dtype=np.float32).reshape(1, 2, 6),
        "right_gripper": np.asarray([[[0.3], [0.4]]], dtype=np.float32),
    }
    isaac = groot_action_to_isaac(action)
    assert isaac.shape == (1, 2, 16)
    np.testing.assert_allclose(isaac[0, :, 6], isaac[0, :, 7])
    np.testing.assert_allclose(isaac[0, :, 14], isaac[0, :, 15])


def test_groot_action_rejects_wrong_group_dimension() -> None:
    action = {
        "left_arm": np.zeros((1, 2, 5), dtype=np.float32),
        "left_gripper": np.zeros((1, 2, 1), dtype=np.float32),
        "right_arm": np.zeros((1, 2, 6), dtype=np.float32),
        "right_gripper": np.zeros((1, 2, 1), dtype=np.float32),
    }
    with pytest.raises(ValueError, match="left_arm"):
        groot_action_to_isaac(action)
