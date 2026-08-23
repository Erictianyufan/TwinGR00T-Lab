"""Continuous metrics for failed Isaac Lab Mimic episodes."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _rounded(value: float) -> float:
    return round(float(value), 6)


def _relative_positions(object_pose: np.ndarray, eef_pose: np.ndarray) -> np.ndarray:
    if object_pose.shape != eef_pose.shape or object_pose.shape[1:] != (4, 4):
        raise ValueError(f"Expected matching (T, 4, 4) poses, got {object_pose.shape} and {eef_pose.shape}")
    return (np.linalg.inv(object_pose) @ eef_pose)[:, :3, 3]


def audit_failed_mimic_episode(path: str | Path) -> dict:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install the data extra: pip install -e '.[data]'") from exc

    dataset_path = Path(path)
    with h5py.File(dataset_path, "r") as handle:
        demos = sorted(handle["data"])
        if len(demos) != 1:
            raise ValueError(f"Expected exactly one failed episode, found {len(demos)}")
        demo = handle[f"data/{demos[0]}"]
        plate = np.asarray(demo["states/rigid_object/brown_box/root_pose"][:, :3], dtype=np.float64)
        destination = np.asarray(
            demo["states/rigid_object/blue_sorting_bin/root_pose"][:, :3], dtype=np.float64
        )
        robot = np.asarray(demo["obs/robot_pos"][:], dtype=np.float64)

        initial_z = plate[0, 2]
        lift = plate[:, 2] - initial_z
        destination_distance = np.linalg.norm(plate - destination, axis=1)
        xy_displacement = np.linalg.norm(plate[:, :2] - plate[0, :2], axis=1)
        robot_displacement = np.linalg.norm(robot - robot[0], axis=1)

        wrist_metrics = {}
        datagen_path = "obs/datagen_info"
        if datagen_path in demo:
            datagen = demo[datagen_path]
            if "object_pose/brown_box" in datagen:
                object_pose = np.asarray(datagen["object_pose/brown_box"][:], dtype=np.float64)
                for arm in ("left", "right"):
                    key = f"eef_pose/{arm}"
                    if key in datagen:
                        relative = _relative_positions(object_pose, np.asarray(datagen[key][:], dtype=np.float64))
                        norm = np.linalg.norm(relative, axis=1)
                        closest = int(np.argmin(norm))
                        wrist_metrics[arm] = {
                            "min_center_distance_m": _rounded(norm[closest]),
                            "closest_frame": closest,
                            "relative_xyz_at_closest_m": [_rounded(value) for value in relative[closest]],
                        }

        max_lift = float(np.max(lift))
        if max_lift < 0.02:
            failure_reason = "no_lift"
        elif float(np.min(destination_distance)) > 0.35:
            failure_reason = "lift_without_delivery"
        else:
            failure_reason = "near_destination_without_success"

        return {
            "schema_version": 1,
            "dataset": str(dataset_path),
            "frames": int(plate.shape[0]),
            "success": False,
            "failure_reason": failure_reason,
            "initial_plate_position_w": [_rounded(value) for value in plate[0]],
            "final_plate_position_w": [_rounded(value) for value in plate[-1]],
            "max_plate_lift_m": _rounded(max_lift),
            "max_lift_frame": int(np.argmax(lift)),
            "max_plate_xy_displacement_m": _rounded(np.max(xy_displacement)),
            "final_plate_xy_displacement_m": _rounded(xy_displacement[-1]),
            "min_destination_l2_m": _rounded(np.min(destination_distance)),
            "min_destination_frame": int(np.argmin(destination_distance)),
            "max_robot_displacement_m": _rounded(np.max(robot_displacement)),
            "final_robot_displacement_m": _rounded(robot_displacement[-1]),
            "wrist_object_metrics": wrist_metrics,
        }
