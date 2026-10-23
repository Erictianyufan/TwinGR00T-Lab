"""Kinematic evidence gate for a bimanual grasp-and-sort recording.

This does not measure finger contact forces. Sustained object lift near a closed
gripper is a proxy that must be checked against the corresponding camera video.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class GraspAcceptanceConfig:
    min_lift_m: float = 0.025
    min_lift_duration_s: float = 0.08
    max_eef_distance_m: float = 0.15
    max_gripper_position_m: float = 0.04
    min_object_displacement_m: float = 0.03
    min_arm_motion_rad: float = 0.10
    stable_duration_s: float = 0.5
    max_final_height_error_m: float = 0.02
    max_linear_speed_m_s: float = 0.03
    max_angular_speed_rad_s: float = 0.5
    max_stable_position_span_m: float = 0.01


def longest_true_run(values: np.ndarray) -> int:
    best = run = 0
    for value in values:
        run = run + 1 if value else 0
        best = max(best, run)
    return best


def evaluate_grasp_sorting(
    *,
    initial_block_positions: np.ndarray,
    block_positions: np.ndarray,
    block_linear_velocities: np.ndarray,
    block_angular_velocities: np.ndarray,
    eef_positions: np.ndarray,
    measured_states: np.ndarray,
    sorting_predicate: np.ndarray,
    step_dt: float,
    config: GraspAcceptanceConfig | None = None,
) -> dict:
    """Recompute acceptance from measured arrays, never from commanded actions."""
    config = config or GraspAcceptanceConfig()
    if not math.isfinite(step_dt) or step_dt <= 0:
        raise ValueError("step_dt must be finite and positive")
    if any(not math.isfinite(value) or value <= 0 for value in asdict(config).values()):
        raise ValueError("Acceptance thresholds must be finite and positive")
    initial = np.asarray(initial_block_positions, dtype=np.float64)
    positions = np.asarray(block_positions, dtype=np.float64)
    linear = np.asarray(block_linear_velocities, dtype=np.float64)
    angular = np.asarray(block_angular_velocities, dtype=np.float64)
    eef = np.asarray(eef_positions, dtype=np.float64)
    states = np.asarray(measured_states, dtype=np.float64)
    sorting = np.asarray(sorting_predicate)
    steps = len(positions)
    for name, value, shape in (
        ("initial positions", initial, (3, 3)),
        ("positions", positions, (steps, 3, 3)),
        ("linear velocity", linear, (steps, 3, 3)),
        ("angular velocity", angular, (steps, 3, 3)),
        ("end effectors", eef, (steps, 2, 3)),
        ("measured states", states, (steps, 14)),
        ("sorting predicate", sorting, (steps,)),
    ):
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid {name} trace: expected finite {shape}, got {value.shape}")
    if steps < 1 or not np.isin(sorting, (False, True)).all():
        raise ValueError("Need nonempty measured traces and boolean sorting flags")

    lift = positions[:, :, 2] - initial[None, :, 2]
    distances = np.linalg.norm(positions[:, :, None, :] - eef[:, None, :, :], axis=-1)
    nearest_arm = np.argmin(distances, axis=2)
    closed = states[:, (6, 13)] < config.max_gripper_position_m
    # N sampled states span N-1 intervals; include both interval endpoints.
    required_lift_steps = max(2, math.ceil(config.min_lift_duration_s / step_dt) + 1)
    # Assign each lifted object to its nearest arm, avoiding one lift being
    # credited to both arms just because the workcell is compact.
    runs = np.zeros((3, 2), dtype=np.int64)
    for block in range(3):
        for arm in range(2):
            candidate = (
                (lift[:, block] >= config.min_lift_m)
                & (distances[:, block, arm] <= config.max_eef_distance_m)
                & (nearest_arm[:, block] == arm)
                & closed[:, arm]
            )
            runs[block, arm] = longest_true_run(candidate)
    qualified = runs >= required_lift_steps
    displacement = np.max(np.linalg.norm(positions - initial[None], axis=-1), axis=0)
    object_grasps = qualified.any(axis=1) & (displacement >= config.min_object_displacement_m)
    arm_grasps = qualified.any(axis=0)
    arm_motion = np.asarray([np.max(np.ptp(states[:, :6], axis=0)), np.max(np.ptp(states[:, 7:13], axis=0))])

    window_steps = max(2, math.ceil(config.stable_duration_s / step_dt) + 1)
    final_window = positions[-window_steps:]
    enough_window = steps >= window_steps
    # Independently recheck this task's unchanged upstream xy ordering rule.
    # A stored "success" flag alone is not physical evidence.
    delta = positions[:, 1:, :2] - positions[:, :-1, :2]
    geometric_sorting = np.all((delta[:, :, 0] > 0) & (delta[:, :, 0] < 0.13) & (np.abs(delta[:, :, 1]) < 0.03), axis=1)
    sorted_window = bool(enough_window and np.all(sorting[-window_steps:] & geometric_sorting[-window_steps:]))
    height_ok = bool(np.all(np.abs(final_window[:, :, 2] - initial[None, :, 2]) <= config.max_final_height_error_m))
    linear_max = float(np.max(np.linalg.norm(linear[-window_steps:], axis=2)))
    angular_max = float(np.max(np.linalg.norm(angular[-window_steps:], axis=2)))
    position_span = float(np.max(np.linalg.norm(np.ptp(final_window, axis=0), axis=1)))
    grippers_released = bool(np.all(states[-window_steps:, (6, 13)] >= config.max_gripper_position_m))
    stable = bool(
        enough_window
        and height_ok
        and linear_max <= config.max_linear_speed_m_s
        and angular_max <= config.max_angular_speed_rad_s
        and position_span <= config.max_stable_position_span_m
        and grippers_released
    )
    both_arms_motion = bool(np.all(arm_motion >= config.min_arm_motion_rad))
    success = bool(object_grasps.all() and arm_grasps.all() and both_arms_motion and sorted_window and stable)
    return {
        "schema_version": 1,
        "success": success,
        "evidence_type": "kinematic_lift_near_closed_gripper",
        "direct_contact_sensor_verified": False,
        "video_review_required": True,
        "config": asdict(config),
        "step_dt": step_dt,
        "grasp_milestone": bool(object_grasps.any()),
        "all_objects_grasped": bool(object_grasps.all()),
        "object_grasped": object_grasps.tolist(),
        "both_arms_grasp_evidence": bool(arm_grasps.all()),
        "arm_grasp_evidence": arm_grasps.tolist(),
        "both_arms_measured_motion": both_arms_motion,
        "arm_motion_rad": arm_motion.tolist(),
        "max_lift_m": np.max(lift, axis=0).tolist(),
        "max_displacement_m": displacement.tolist(),
        "longest_lift_runs_steps_by_object_arm": runs.tolist(),
        "required_lift_steps": required_lift_steps,
        "sorting_stable_window": sorted_window,
        "stable_placement": stable,
        "stable_window_steps": window_steps,
        "final_max_linear_speed_m_s": linear_max,
        "final_max_angular_speed_rad_s": angular_max,
        "final_position_span_m": position_span,
        "final_height_ok": height_ok,
        "final_grippers_released": grippers_released,
    }
