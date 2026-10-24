"""Prepare a *candidate* Isaac factory replay from public RoboTwin ALOHA data.

Only HDF5, JSON and the first head-camera JPEG are read; the archive's pickle
members are never loaded. Camera geometry estimates initial block centers.
This output is NOT training-eligible until collect_aloha_factory_teacher.py
replays it in Isaac and independently passes the physical acceptance gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np

from twingr00t_lab.public_robotwin import archive_member


def block_centers_from_head(
    rgb: np.ndarray, intrinsic: np.ndarray, extrinsic: np.ndarray, *, center_z: float
) -> tuple[np.ndarray, dict]:
    """Back-project color-mask medians through the calibrated camera to a z plane."""
    image = np.asarray(rgb, dtype=np.uint8)
    k = np.asarray(intrinsic, dtype=np.float64)
    e = np.asarray(extrinsic, dtype=np.float64)
    if image.ndim != 3 or image.shape[2] != 3 or k.shape != (3, 3) or e.shape != (3, 4):
        raise ValueError("Expected RGB image, 3x3 intrinsic and 3x4 extrinsic")
    if not np.isfinite(k).all() or not np.isfinite(e).all() or not math.isfinite(center_z):
        raise ValueError("Non-finite camera calibration or table plane")
    rotation, translation = e[:, :3], e[:, 3]
    camera_origin = np.linalg.solve(rotation, -translation)
    channels = image.astype(np.float32)
    centers = []
    evidence = {}
    for index, color in enumerate(("red", "green", "blue")):
        other = [column for column in range(3) if column != index]
        mask = (
            (channels[:, :, index] > 100)
            & (channels[:, :, index] > 1.4 * channels[:, :, other[0]])
            & (channels[:, :, index] > 1.4 * channels[:, :, other[1]])
        )
        vertical, horizontal = np.nonzero(mask)
        if not 50 <= len(horizontal) <= image.shape[0] * image.shape[1] // 8:
            raise ValueError(f"Unreliable {color} mask area: {len(horizontal)}")
        pixel = np.asarray([np.median(horizontal), np.median(vertical), 1.0], dtype=np.float64)
        world_ray = np.linalg.solve(rotation, np.linalg.solve(k, pixel))
        if abs(world_ray[2]) < 1e-6:
            raise ValueError("Camera ray is parallel to the block-center plane")
        scale = (center_z - camera_origin[2]) / world_ray[2]
        if scale <= 0:
            raise ValueError("Block-center plane is behind the head camera")
        center = camera_origin + scale * world_ray
        if abs(center[0]) > 0.4 or abs(center[1]) > 0.3:
            raise ValueError(f"{color} estimate lies outside the ALOHA workcell: {center}")
        centers.append(center)
        evidence[color] = {"mask_pixels": len(horizontal), "median_uv": pixel[:2].tolist(), "xyz_m": center.tolist()}
    result = np.stack(centers).astype(np.float32)
    pairwise = np.linalg.norm(result[:, None, :2] - result[None, :, :2], axis=-1)
    if np.any(pairwise[np.triu_indices(3, 1)] < 0.055):
        raise ValueError("Color-mask block centers are implausibly close")
    return result, evidence


def resample_measured_qpos(qpos: np.ndarray, *, source_fps: float = 30.0, target_fps: float = 62.5) -> np.ndarray:
    """Interpolate 14-D measured qpos, then map to 16-D Isaac joint targets."""
    measured = np.asarray(qpos, dtype=np.float64)
    if measured.ndim != 2 or measured.shape[1] != 14 or len(measured) < 2 or not np.isfinite(measured).all():
        raise ValueError("Expected finite T>=2 x 14 measured joint states")
    if not 0 < source_fps < target_fps or not math.isfinite(source_fps) or not math.isfinite(target_fps):
        raise ValueError("Require finite source FPS below target FPS")
    if measured[:, [6, 13]].min() < -1e-4 or measured[:, [6, 13]].max() > 1.0001:
        raise ValueError("Public gripper state must be normalized to [0,1]")
    source_time = np.arange(len(measured), dtype=np.float64) / source_fps
    count = math.floor(source_time[-1] * target_fps)
    target_time = np.arange(1, count + 1, dtype=np.float64) / target_fps
    logical = np.stack([np.interp(target_time, source_time, measured[:, column]) for column in range(14)], axis=1)
    for column in (6, 13):
        logical[:, column] = -0.010 + 0.058 * logical[:, column]
    expanded = np.concatenate(
        (logical[:, :6], np.repeat(logical[:, 6:7], 2, axis=1),
         logical[:, 7:13], np.repeat(logical[:, 13:14], 2, axis=1)),
        axis=1,
    ).astype(np.float32)
    if expanded.shape != (count, 16) or not np.isfinite(expanded).all():
        raise ValueError("Invalid resampled 16-D target array")
    if len(expanded) > 1 and np.abs(np.diff(expanded[:, [*range(6), *range(8, 14)]], axis=0)).max() > 0.25:
        raise ValueError("Resampled public arm target jumps by more than 0.25 rad")
    return expanded


def prepare(archive_path: Path, template: Path, output_dir: Path, episode: int) -> dict:
    import h5py
    from PIL import Image

    if not archive_path.is_file() or not template.is_file():
        raise FileNotFoundError("Public archive and accepted Isaac factory template are required")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    data_member = archive_member(episode, "data")
    instruction_member = archive_member(episode, "instructions")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("Public ZIP CRC validation failed")
        raw_hdf5 = archive.read(data_member)
        instruction_bytes = archive.read(instruction_member)
    instruction = json.loads(instruction_bytes)
    if not isinstance(instruction.get("seen"), list) or not instruction["seen"]:
        raise ValueError("Public episode has no language instruction")
    with h5py.File(BytesIO(raw_hdf5), "r") as source, h5py.File(template, "r") as factory:
        qpos = source["joint_action/vector"][:]
        camera = source["observation/head_camera"]
        with Image.open(BytesIO(bytes(camera["rgb"][0]))) as frame:
            if frame.format != "JPEG" or frame.info.get("comment") == b"XPL-RGB1":
                raise ValueError("Expected legacy BGR-swapped public JPEG")
            rgb = np.asarray(frame.convert("RGB"), dtype=np.uint8)[:, :, ::-1].copy()
        center_z = float(factory["initial_state/rigid_object/block1/root_pose"][0, 2])
        centers, color_evidence = block_centers_from_head(
            rgb, camera["intrinsic_cv"][0], camera["extrinsic_cv"][0], center_z=center_z
        )
        targets = resample_measured_qpos(qpos)
        with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}-", dir=output_dir.parent) as raw_dir:
            staging = Path(raw_dir)
            (staging / "data").mkdir()
            (staging / "instructions").mkdir()
            (staging / "reports").mkdir()
            source_path = staging / "data" / f"demo_{episode}.hdf5"
            with h5py.File(source_path, "x") as output:
                factory.copy("initial_state", output)
                for index, center in enumerate(centers, 1):
                    pose = output[f"initial_state/rigid_object/block{index}/root_pose"]
                    pose[0, :3] = center
                policy = output.require_group("observations").require_group("policy")
                policy.create_dataset("actions", data=targets, compression="gzip")
                output.attrs["status"] = "candidate_public_teacher_replay_not_physically_verified"
                output.attrs["public_episode"] = episode
            (staging / "instructions" / f"demo_{episode}.json").write_bytes(instruction_bytes)
            report = {
                "status": "candidate_public_teacher_replay_not_physically_verified",
                "source": "TianxingChen/RoboTwin2.0 blocks_ranking_rgb aloha-agilex_clean_50",
                "public_episode": episode,
                "archive_member": data_member,
                "source_hdf5_sha256": hashlib.sha256(raw_hdf5).hexdigest(),
                "template": str(template),
                "source_frames": len(qpos),
                "source_fps": 30.0,
                "target_fps": 62.5,
                "replayed_action_steps": len(targets),
                "source_action_kind": "measured_qpos_interpolated_as_candidate_joint_targets",
                "initial_block_pose_source": "first_head_rgb_legacy_bgr_corrected_and_camera_ray_plane_intersection",
                "block_center_z_m": center_z,
                "block_center_evidence": color_evidence,
                "pickle_loaded": False,
                "isaac_physical_acceptance": None,
                "training_eligible": False,
            }
            (staging / "reports" / f"demo_{episode}.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
            staging.rename(output_dir)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--episode", required=True, type=int)
    args = parser.parse_args()
    print(json.dumps(prepare(args.archive, args.template, args.output_dir, args.episode), indent=2))


if __name__ == "__main__":
    main()
