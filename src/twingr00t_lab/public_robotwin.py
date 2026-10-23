"""Import RoboTwin's public clean-50 ALOHA RGB-block demonstrations.

The public HDF5 ``joint_action/vector`` is *measured joint state*, not a
command: RoboTwin's recorder fills it from ``get_left/right_arm_jointState``.
Pair observation t with measured state t+1 as a one-frame-ahead imitation
target. This is auxiliary SAPIEN-domain training data, never a successful
Isaac/GR00T run. Only exact HDF5/JSON archive members are read; pickle files
in the archive are deliberately not opened or executed.
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
from typing import Any

import numpy as np

from twingr00t_lab.conversion import (
    CAMERA_MAP,
    dataset_info,
    encode_video,
    modality_metadata,
    write_json,
    write_jsonl,
    write_parquet,
)

ARCHIVE_ROOT = "aloha-agilex_clean_50"
CAMERA_SOURCES = {
    "head": "observation/head_camera/rgb",
    "left_wrist": "observation/left_camera/rgb",
    "right_wrist": "observation/right_camera/rgb",
}
STATE_GRIPPER_CLOSED = 0.006
ACTION_GRIPPER_CLOSED = -0.010
GRIPPER_OPEN = 0.048


def canonical_vectors(qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map normalized RoboTwin grippers to ALOHA measured/command spaces."""
    values = np.asarray(qpos, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 14 or len(values) < 2:
        raise ValueError(f"Expected at least two 14-D states, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Public joint states contain NaN or Inf")
    for column in (6, 13):
        if values[:, column].min() < -1e-4 or values[:, column].max() > 1.0001:
            raise ValueError("RoboTwin gripper state must be normalized to [0, 1]")
    state = values[:-1].copy()
    action = values[1:].copy()
    for column in (6, 13):
        state[:, column] = STATE_GRIPPER_CLOSED + (GRIPPER_OPEN - STATE_GRIPPER_CLOSED) * state[:, column]
        action[:, column] = ACTION_GRIPPER_CLOSED + (GRIPPER_OPEN - ACTION_GRIPPER_CLOSED) * action[:, column]
    return state, action


def archive_member(episode: int, kind: str) -> str:
    if not 0 <= episode < 50 or kind not in {"data", "instructions"}:
        raise ValueError("Only clean-50 HDF5/JSON members are accepted")
    extension = "hdf5" if kind == "data" else "json"
    return f"{ARCHIVE_ROOT}/{kind}/episode{episode}.{extension}"


def import_clean50(archive_path: Path, output_dir: Path, *, episodes: int = 50, fps: float = 30.0) -> dict[str, Any]:
    """Write a fresh LeRobot v2 dataset with explicit provenance/audits."""
    try:
        import h5py
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - optional data dependency
        raise RuntimeError("Install the data extra: pip install -e '.[data]'") from exc

    if not archive_path.is_file() or not 1 <= episodes <= 50 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Need a clean-50 ZIP, 1-50 episodes, and a positive finite FPS")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing data: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with archive_path.open("rb") as stream:
        archive_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("Public archive CRC validation failed")
        expected = {archive_member(index, kind) for index in range(episodes) for kind in ("data", "instructions")}
        if not expected.issubset(archive.namelist()):
            raise ValueError("Public archive lacks a required exact HDF5 or instruction member")
        with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}-", dir=output_dir.parent) as staging_raw:
            staging = Path(staging_raw)
            episode_rows: list[dict[str, Any]] = []
            task_rows: list[dict[str, Any]] = []
            audit_rows: list[dict[str, Any]] = []
            global_index = 0
            image_height = image_width = 0
            for index in range(episodes):
                raw_hdf5 = archive.read(archive_member(index, "data"))
                instruction = json.loads(archive.read(archive_member(index, "instructions")))
                seen = instruction.get("seen")
                if not isinstance(seen, list) or not seen or not isinstance(seen[0], str) or not seen[0].strip():
                    raise ValueError(f"Episode {index} lacks a nonempty seen instruction")
                with h5py.File(BytesIO(raw_hdf5), "r") as handle:
                    state, action = canonical_vectors(handle["joint_action/vector"][:])
                    source_frames = len(state) + 1
                    for video_name in CAMERA_MAP:
                        source = handle[CAMERA_SOURCES[video_name]]
                        if source.shape != (source_frames,):
                            raise ValueError(f"Episode {index} {video_name} frame count mismatch")
                        # The public ZIP predates the XPL-RGB1 marker. A normal
                        # PIL/JPEG reader sees BGR-flipped colors; conversion
                        # must apply the existing legacy correction exactly once.
                        first = source[0]
                        first_bytes = (
                            bytes(first)
                            if isinstance(first, (bytes, np.bytes_))
                            else np.asarray(first).tobytes()
                        )
                        with Image.open(BytesIO(first_bytes)) as image:
                            if image.info.get("comment") == b"XPL-RGB1":
                                raise ValueError("Unexpected standard RGB JPEG; legacy BGR correction would be wrong")
                            if image.format != "JPEG":
                                raise ValueError("Expected legacy JPEG RGB camera bytes")
                        video_path = (
                            staging / "videos" / "chunk-000" / f"observation.images.{video_name}"
                            / f"episode_{index:06d}.mp4"
                        )
                        count, height, width = encode_video(
                            source, video_path, fps=fps, correct_legacy_jpeg_bgr=True, frame_limit=len(state)
                        )
                        if count != len(state):
                            raise ValueError("Video/action alignment mismatch")
                        if image_height and (height, width) != (image_height, image_width):
                            raise ValueError("Public episodes/cameras use inconsistent image dimensions")
                        image_height, image_width = height, width
                write_parquet(
                    staging / "data" / "chunk-000" / f"episode_{index:06d}.parquet",
                    state=state, action=action, episode_index=index, task_index=index,
                    global_start=global_index, fps=fps,
                )
                episode_rows.append({"episode_index": index, "tasks": [seen[0]], "length": len(state)})
                task_rows.append({"task_index": index, "task": seen[0]})
                audit_rows.append({
                    "source_episode": index,
                    "source_hdf5_sha256": hashlib.sha256(raw_hdf5).hexdigest(),
                    "source_frames": source_frames,
                    "paired_frames": len(state),
                    "first_state_left_gripper": float(state[0, 6]),
                    "first_action_left_gripper": float(action[0, 6]),
                })
                global_index += len(state)
                print(f"Imported public episode {index + 1}/{episodes}: {len(state)} pairs", flush=True)
            write_json(staging / "meta" / "modality.json", modality_metadata())
            write_json(
                staging / "meta" / "info.json",
                dataset_info(episodes=episodes, frames=global_index, height=image_height, width=image_width, fps=fps),
            )
            write_jsonl(staging / "meta" / "episodes.jsonl", episode_rows)
            write_jsonl(staging / "meta" / "tasks.jsonl", task_rows)
            manifest = {
                "status": "converted_auxiliary_source_demonstrations_not_physically_verified_in_isaac",
                "source_repo": "TianxingChen/RoboTwin2.0",
                "source_archive": str(archive_path),
                "source_archive_sha256": archive_sha256,
                "source_embodiment": "aloha-agilex_clean_50",
                "source_task": "blocks_ranking_rgb",
                "episodes": episodes,
                "frames": global_index,
                "fps": fps,
                "fps_evidence": "source ZIP video/episode0.mp4 ffprobe reports 30/1",
                "observation_action_alignment": "camera and measured joint state i -> measured joint state i+1",
                "legacy_jpeg_bgr_corrected": True,
                "state_gripper_closed_open_m": [STATE_GRIPPER_CLOSED, GRIPPER_OPEN],
                "target_gripper_closed_open_m": [ACTION_GRIPPER_CLOSED, GRIPPER_OPEN],
                "source_pickle_executed": False,
                "isaac_factory_success_claim": False,
                "episode_audit": audit_rows,
            }
            write_json(staging / "meta" / "source_audit.json", manifest)
            staging.rename(output_dir)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()
    print(json.dumps(import_clean50(args.archive, args.output_dir, episodes=args.episodes, fps=args.fps), indent=2))


if __name__ == "__main__":
    main()
