from __future__ import annotations

import h5py
import numpy as np

from twingr00t_lab.plate_seed import shift_targets_in_object_frame


def test_shift_targets_uses_object_rotation(tmp_path) -> None:
    source = tmp_path / "source.hdf5"
    output = tmp_path / "output.hdf5"
    with h5py.File(source, "w") as handle:
        data = handle.create_group("data")
        demo = data.create_group("demo_0/obs/datagen_info")
        object_pose = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
        object_pose[1, :3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
        demo.create_group("object_pose").create_dataset("brown_box", data=object_pose)
        targets = demo.create_group("target_eef_pose")
        for arm in ("left", "right"):
            targets.create_dataset(arm, data=np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0))

    shift_targets_in_object_frame(source, output, local_offset_xyz=(-0.1, 0.0, 0.0))
    with h5py.File(output, "r") as handle:
        left = handle["data/demo_0/obs/datagen_info/target_eef_pose/left"][:]
        assert np.allclose(left[0, :3, 3], [-0.1, 0.0, 0.0])
        assert np.allclose(left[1, :3, 3], [0.0, -0.1, 0.0])


def test_shift_targets_can_reframe_upside_down_source_for_upright_plate(tmp_path) -> None:
    source = tmp_path / "source.hdf5"
    output = tmp_path / "output.hdf5"
    with h5py.File(source, "w") as handle:
        data = handle.create_group("data")
        demo = data.create_group("demo_0/obs/datagen_info")
        object_pose = np.eye(4, dtype=np.float32)[None]
        demo.create_group("object_pose").create_dataset("brown_box", data=object_pose)
        targets = demo.create_group("target_eef_pose")
        target = np.eye(4, dtype=np.float32)[None]
        target[0, :3, 3] = [0.2, 0.1, -0.04]
        for arm in ("left", "right"):
            targets.create_dataset(arm, data=target)

    shift_targets_in_object_frame(
        source,
        output,
        local_offset_xyz=(0.08, 0.0, -0.02),
        target_frame_rotation_y_deg=180.0,
    )
    with h5py.File(output, "r") as handle:
        left = handle["data/demo_0/obs/datagen_info/target_eef_pose/left"][:]
        assert np.allclose(left[0, :3, 3], [-0.12, 0.1, 0.02], atol=1e-6)
        assert np.allclose(left[0, :3, :3], np.diag([-1.0, 1.0, -1.0]), atol=1e-6)
