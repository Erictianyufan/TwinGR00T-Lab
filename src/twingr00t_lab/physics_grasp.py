"""Dependency-light acceptance metrics for contact-physics grasp replays."""

from __future__ import annotations

from typing import Any

import numpy as np


def _longest_true_run(mask: np.ndarray) -> np.ndarray:
    indices = np.flatnonzero(mask)
    if not indices.size:
        return indices
    boundaries = np.flatnonzero(np.diff(indices) > 1) + 1
    runs = np.split(indices, boundaries)
    return max(runs, key=len)


def _quat_rotate_inverse_wxyz(quaternions: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Rotate world vectors into quaternion-local frames (scalar-first)."""
    norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("eef_orientations contains a zero-length quaternion")
    quaternions = quaternions / norms
    scalar = quaternions[..., :1]
    imaginary = quaternions[..., 1:]
    return (
        (2.0 * scalar**2 - 1.0) * vectors
        + 2.0 * imaginary * np.sum(imaginary * vectors, axis=-1, keepdims=True)
        - 2.0 * scalar * np.cross(imaginary, vectors)
    )


def evaluate_physics_grasps(
    object_positions: np.ndarray,
    object_linear_velocities: np.ndarray,
    object_angular_velocities: np.ndarray,
    eef_positions: np.ndarray,
    hand_contact_forces: np.ndarray,
    *,
    eef_orientations: np.ndarray | None = None,
    settle_frames: int = 25,
    minimum_lift_m: float = 0.025,
    minimum_contact_force_n: float = 0.05,
    maximum_relative_drift_m: float = 0.025,
    maximum_final_linear_speed_m_s: float = 0.05,
    maximum_final_angular_speed_rad_s: float = 0.50,
) -> dict[str, Any]:
    """Validate that every object was contact-grasped, lifted, and released stably.

    Array shapes are ``(frames, objects, 3)`` for object vectors,
    ``(frames, hands, 3)`` for end-effectors, and ``(frames, objects, hands)``
    for filtered normal-contact force magnitudes.
    """
    object_positions = np.asarray(object_positions, dtype=float)
    object_linear_velocities = np.asarray(object_linear_velocities, dtype=float)
    object_angular_velocities = np.asarray(object_angular_velocities, dtype=float)
    eef_positions = np.asarray(eef_positions, dtype=float)
    hand_contact_forces = np.asarray(hand_contact_forces, dtype=float)
    if eef_orientations is None:
        eef_orientations = np.zeros((*eef_positions.shape[:2], 4), dtype=float)
        eef_orientations[..., 0] = 1.0
    else:
        eef_orientations = np.asarray(eef_orientations, dtype=float)
    if object_positions.ndim != 3 or object_positions.shape[-1] != 3:
        raise ValueError("object_positions must have shape (frames, objects, 3)")
    if object_linear_velocities.shape != object_positions.shape:
        raise ValueError("object_linear_velocities must match object_positions")
    if object_angular_velocities.shape != object_positions.shape:
        raise ValueError("object_angular_velocities must match object_positions")
    if eef_positions.ndim != 3 or eef_positions.shape[0] != len(object_positions):
        raise ValueError("eef_positions must have shape (frames, hands, 3)")
    expected_orientations = (*eef_positions.shape[:2], 4)
    if eef_orientations.shape != expected_orientations:
        raise ValueError(f"eef_orientations must have shape {expected_orientations}")
    expected_forces = (len(object_positions), object_positions.shape[1], eef_positions.shape[1])
    if hand_contact_forces.shape != expected_forces:
        raise ValueError(f"hand_contact_forces must have shape {expected_forces}")
    if not 1 <= settle_frames <= len(object_positions):
        raise ValueError("settle_frames must be within the recorded frame count")

    results: list[dict[str, Any]] = []
    for object_index in range(object_positions.shape[1]):
        positions = object_positions[:, object_index]
        initial_z = float(positions[0, 2])
        lift_m = float(np.max(positions[:, 2]) - initial_z)
        forces = hand_contact_forces[:, object_index]
        contact_mask = np.max(forces, axis=1) >= minimum_contact_force_n
        lifted_mask = positions[:, 2] >= initial_z + minimum_lift_m
        held_mask = contact_mask & lifted_mask
        held_indices = np.flatnonzero(held_mask)
        # Evaluate a continuous episode for each hand independently.  A parcel
        # can brush the opposite gripper while being transported; collapsing
        # both force channels into one mask would then compare the parcel with
        # the wrong end effector and report a large, artificial pose drift.
        hand_episodes = [
            _longest_true_run(
                (forces[:, hand] >= minimum_contact_force_n) & lifted_mask
            )
            for hand in range(forces.shape[1])
        ]
        episode_scores = [
            (episode.size, float(np.sum(forces[episode, hand])))
            for hand, episode in enumerate(hand_episodes)
        ]
        dominant_hand = int(max(range(len(hand_episodes)), key=episode_scores.__getitem__))
        grasp_episode = hand_episodes[dominant_hand]
        relative_drift = float("inf")
        if grasp_episode.size:
            world_offset = (
                positions[grasp_episode] - eef_positions[grasp_episode, dominant_hand]
            )
            relative = _quat_rotate_inverse_wxyz(
                eef_orientations[grasp_episode, dominant_hand], world_offset
            )
            center = np.median(relative, axis=0)
            relative_drift = float(np.max(np.linalg.norm(relative - center, axis=1)))
        else:
            dominant_hand = None

        final_linear_speed = float(
            np.max(np.linalg.norm(object_linear_velocities[-settle_frames:, object_index], axis=1))
        )
        final_angular_speed = float(
            np.max(np.linalg.norm(object_angular_velocities[-settle_frames:, object_index], axis=1))
        )
        contact_force_max = float(np.max(forces))
        passed = (
            contact_force_max >= minimum_contact_force_n
            and lift_m >= minimum_lift_m
            and grasp_episode.size >= 3
            and relative_drift <= maximum_relative_drift_m
            and final_linear_speed <= maximum_final_linear_speed_m_s
            and final_angular_speed <= maximum_final_angular_speed_rad_s
        )
        results.append(
            {
                "object_index": object_index,
                "contact_force_max_n": contact_force_max,
                "contact_frames": int(np.count_nonzero(contact_mask)),
                "held_and_lifted_frames": int(held_indices.size),
                "continuous_grasp_episode_frames": int(grasp_episode.size),
                "dominant_hand": dominant_hand,
                "lift_height_m": lift_m,
                "max_relative_translation_drift_m": relative_drift,
                "final_max_linear_speed_m_s": final_linear_speed,
                "final_max_angular_speed_rad_s": final_angular_speed,
                "pass": passed,
            }
        )

    frame_steps = np.linalg.norm(np.diff(object_positions, axis=0), axis=-1)
    return {
        "pass": all(item["pass"] for item in results),
        "objects": results,
        "max_object_translation_per_frame_m": float(np.max(frame_steps)) if frame_steps.size else 0.0,
        "thresholds": {
            "minimum_lift_m": minimum_lift_m,
            "minimum_contact_force_n": minimum_contact_force_n,
            "maximum_relative_drift_m": maximum_relative_drift_m,
            "maximum_final_linear_speed_m_s": maximum_final_linear_speed_m_s,
            "maximum_final_angular_speed_rad_s": maximum_final_angular_speed_rad_s,
        },
    }
