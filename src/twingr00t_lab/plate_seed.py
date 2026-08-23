"""Create geometry-adapted G1 Mimic seed data without mutating the source."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np


def shift_targets_in_object_frame(
    source: str | Path,
    output: str | Path,
    *,
    local_offset_xyz: tuple[float, float, float],
    target_frame_rotation_y_deg: float = 0.0,
) -> dict:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install the data extra: pip install -e '.[data]'") from exc

    source_path = Path(source)
    output_path = Path(output)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing seed dataset: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, output_path)

    local_offset = np.asarray(local_offset_xyz, dtype=np.float32)
    angle = np.deg2rad(target_frame_rotation_y_deg)
    frame_rotation = np.eye(4, dtype=np.float32)
    frame_rotation[:3, :3] = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ],
        dtype=np.float32,
    )
    modified_frames = 0
    with h5py.File(output_path, "r+") as handle:
        for demo_name in handle["data"]:
            datagen = handle[f"data/{demo_name}/obs/datagen_info"]
            object_pose = np.asarray(datagen["object_pose/brown_box"][:], dtype=np.float32)
            for arm in ("left", "right"):
                target = datagen[f"target_eef_pose/{arm}"]
                values = np.asarray(target[:], dtype=np.float32)
                relative = np.linalg.inv(object_pose) @ values
                relative = frame_rotation @ relative
                relative[:, :3, 3] += local_offset
                target[...] = object_pose @ relative
            modified_frames += object_pose.shape[0]
        handle["data"].attrs["twingroot_plate_target_offset_xyz"] = local_offset
        handle["data"].attrs["twingroot_plate_target_frame_rotation_y_deg"] = target_frame_rotation_y_deg

    return {
        "source": str(source_path),
        "output": str(output_path),
        "local_offset_xyz": local_offset.tolist(),
        "target_frame_rotation_y_deg": float(target_frame_rotation_y_deg),
        "modified_frames_per_arm": modified_frames,
    }
