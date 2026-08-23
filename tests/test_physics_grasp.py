import numpy as np
import pytest

from twingr00t_lab.physics_grasp import evaluate_physics_grasps


def physical_grasp_trace() -> tuple[np.ndarray, ...]:
    frames = 40
    objects = np.zeros((frames, 1, 3), dtype=float)
    objects[:, 0, 2] = 0.765
    eefs = np.zeros((frames, 2, 3), dtype=float)
    eefs[:, 0, 2] = 0.90
    forces = np.zeros((frames, 1, 2), dtype=float)
    for frame in range(8, 25):
        height = min((frame - 8) * 0.004, 0.05)
        objects[frame, 0] = (0.002 * (frame - 8), 0.0, 0.765 + height)
        eefs[frame, 0] = objects[frame, 0] + (0.0, 0.0, 0.135)
        forces[frame, 0, 0] = 2.0
    objects[25:, 0] = objects[24, 0]
    linear = np.zeros_like(objects)
    angular = np.zeros_like(objects)
    return objects, linear, angular, eefs, forces


def test_contact_grasp_trace_passes() -> None:
    result = evaluate_physics_grasps(*physical_grasp_trace(), settle_frames=10)
    assert result["pass"]
    assert result["objects"][0]["dominant_hand"] == 0
    assert result["objects"][0]["max_relative_translation_drift_m"] < 1e-9


def test_pose_animation_without_contact_fails() -> None:
    values = list(physical_grasp_trace())
    values[-1][:] = 0.0
    result = evaluate_physics_grasps(*values, settle_frames=10)
    assert not result["pass"]
    assert result["objects"][0]["held_and_lifted_frames"] == 0


def test_incidental_later_contact_does_not_pollute_grasp_episode() -> None:
    values = list(physical_grasp_trace())
    objects, _, _, eefs, forces = values
    objects[30, 0, 2] += 0.04
    eefs[30, 0] = objects[30, 0] + (0.4, 0.0, 0.0)
    forces[30, 0, 0] = 1.0
    result = evaluate_physics_grasps(*values, settle_frames=5)
    assert result["pass"]
    assert result["objects"][0]["continuous_grasp_episode_frames"] >= 10


def test_opposite_hand_brush_does_not_change_grasp_reference() -> None:
    values = list(physical_grasp_trace())
    objects, _, _, eefs, forces = values
    # The opposite hand briefly brushes the parcel during a valid left-hand
    # transport.  Its far-away pose must not be mixed into the grasp frame.
    eefs[14:21, 1] = objects[14:21, 0] + (0.4, 0.0, 0.0)
    forces[14:21, 0, 1] = 4.0
    result = evaluate_physics_grasps(*values, settle_frames=10)
    assert result["pass"]
    assert result["objects"][0]["dominant_hand"] == 0
    assert result["objects"][0]["max_relative_translation_drift_m"] < 1e-9


def test_wrist_rotation_is_evaluated_in_the_local_grasp_frame() -> None:
    values = list(physical_grasp_trace())
    objects, _, _, eefs, _ = values
    quaternions = np.zeros((40, 2, 4), dtype=float)
    quaternions[..., 0] = 1.0
    offset_local = np.array([0.10, 0.0, -0.05])
    for frame, angle in zip(range(8, 25), np.linspace(0.0, np.pi, 17), strict=True):
        quaternions[frame, 0] = (
            np.cos(angle / 2.0),
            0.0,
            0.0,
            np.sin(angle / 2.0),
        )
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        eefs[frame, 0] = objects[frame, 0] - rotation @ offset_local
    result = evaluate_physics_grasps(
        *values,
        settle_frames=10,
        eef_orientations=quaternions,
    )
    assert result["pass"]
    assert result["objects"][0]["max_relative_translation_drift_m"] < 1e-9


def test_shape_validation_rejects_invalid_input() -> None:
    values = list(physical_grasp_trace())
    values[0] = np.zeros((40, 3))
    with pytest.raises(ValueError, match="object_positions"):
        evaluate_physics_grasps(*values)
