from __future__ import annotations

import h5py
import numpy as np

from twingr00t_lab.mimic_failure import audit_failed_mimic_episode


def test_audit_failed_mimic_episode_reports_lift_and_wrist_distance(tmp_path) -> None:
    path = tmp_path / "failed.hdf5"
    with h5py.File(path, "w") as handle:
        demo = handle.create_group("data/demo_0")
        states = demo.create_group("states/rigid_object")
        plate = np.zeros((3, 7), dtype=np.float32)
        plate[:, :3] = [[0, 0, 0.1], [0.1, 0, 0.14], [0.2, 0, 0.12]]
        destination = np.zeros((3, 7), dtype=np.float32)
        destination[:, :3] = [1, 0, 0.1]
        states.create_group("brown_box").create_dataset("root_pose", data=plate)
        states.create_group("blue_sorting_bin").create_dataset("root_pose", data=destination)
        obs = demo.create_group("obs")
        obs.create_dataset("robot_pos", data=np.array([[0, 0, 0], [0.2, 0, 0], [0.4, 0, 0]]))
        datagen = obs.create_group("datagen_info")
        object_pose = np.repeat(np.eye(4)[None], 3, axis=0)
        datagen.create_group("object_pose").create_dataset("brown_box", data=object_pose)
        eef = np.repeat(np.eye(4)[None], 3, axis=0)
        eef[:, 0, 3] = [0.4, 0.2, 0.3]
        poses = datagen.create_group("eef_pose")
        poses.create_dataset("left", data=eef)
        poses.create_dataset("right", data=eef)

    result = audit_failed_mimic_episode(path)
    assert result["max_plate_lift_m"] == 0.04
    assert result["failure_reason"] == "lift_without_delivery"
    assert result["wrist_object_metrics"]["left"]["min_center_distance_m"] == 0.2
