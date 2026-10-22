"""Convert RoboTwin HDF5 demonstrations into GR00T-flavored LeRobot v2."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

from twingr00t_lab.dataset import CAMERA_ROOT, POLICY_ROOT, canonical_proprio, collapse_isaac_action

CAMERA_MAP = {
    "head": "head_camera_rgb",
    "left_wrist": "left_camera_rgb",
    "right_wrist": "right_camera_rgb",
}
STATE_NAMES = [
    *(f"left_arm_{index}" for index in range(6)),
    "left_gripper",
    *(f"right_arm_{index}" for index in range(6)),
    "right_gripper",
]


def episode_number(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", maxsplit=1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Expected a demo_<number>.hdf5 filename, got {path.name}") from exc


def select_demo_paths(input_dir: Path, *, start_episode: int, limit: int | None) -> list[Path]:
    if start_episode < 0:
        raise ValueError("start_episode must be non-negative")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    paths = sorted((input_dir / "data").glob("demo_*.hdf5"), key=episode_number)
    paths = [path for path in paths if episode_number(path) >= start_episode]
    return paths if limit is None else paths[:limit]


def select_instruction(path: Path, *, index: int = 0, fallback: str | None = None) -> str:
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        seen = payload.get("seen", [])
        if seen:
            return str(seen[index % len(seen)])
    if fallback:
        return fallback
    raise ValueError(f"No seen instruction in {path} and no fallback was supplied")


def modality_metadata() -> dict[str, Any]:
    return {
        "state": {
            "left_arm": {"start": 0, "end": 6},
            "left_gripper": {"start": 6, "end": 7},
            "right_arm": {"start": 7, "end": 13},
            "right_gripper": {"start": 13, "end": 14},
        },
        "action": {
            "left_arm": {"start": 0, "end": 6},
            "left_gripper": {"start": 6, "end": 7},
            "right_arm": {"start": 7, "end": 13},
            "right_gripper": {"start": 13, "end": 14},
        },
        "video": {name: {"original_key": f"observation.images.{name}"} for name in CAMERA_MAP},
        "annotation": {
            "human.task_description": {
                "original_key": "annotation.human.task_description",
            }
        },
    }


def video_feature(height: int, width: int, fps: float) -> dict[str, Any]:
    return {
        "dtype": "video",
        "shape": [height, width, 3],
        "names": ["height", "width", "channels"],
        "info": {
            "video.height": height,
            "video.width": width,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": fps,
            "video.channels": 3,
            "has_audio": False,
        },
    }


def dataset_info(*, episodes: int, frames: int, height: int, width: int, fps: float) -> dict[str, Any]:
    vector_feature = {"dtype": "float32", "shape": [14], "names": STATE_NAMES}
    scalar_features = {
        name: {"dtype": dtype, "shape": [1], "names": None}
        for name, dtype in (
            ("timestamp", "float32"),
            ("frame_index", "int64"),
            ("episode_index", "int64"),
            ("index", "int64"),
            ("task_index", "int64"),
            ("annotation.human.task_description", "int64"),
        )
    }
    features: dict[str, Any] = {
        "action": vector_feature,
        "observation.state": vector_feature,
        **{f"observation.images.{name}": video_feature(height, width, fps) for name in CAMERA_MAP},
        **scalar_features,
    }
    return {
        "codebase_version": "v2.1",
        "robot_type": "aloha_agilex",
        "total_episodes": episodes,
        "total_frames": frames,
        "total_tasks": episodes,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
        "total_chunks": 1 if episodes else 0,
        "total_videos": episodes * len(CAMERA_MAP),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = "".join(json.dumps(row) + "\n" for row in rows)
    path.write_text(rendered, encoding="utf-8")


def encode_video(
    dataset: Any, path: Path, *, fps: float, correct_legacy_jpeg_bgr: bool = False
) -> tuple[int, int, int]:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to encode LeRobot videos")
    encoded_frames = dataset.ndim == 1 and dataset.dtype.kind in {"O", "S"}
    raw_frames = dataset.ndim == 4 and dataset.shape[-1] == 3
    if not encoded_frames and not raw_frames:
        raise ValueError(f"Expected raw RGB (T, H, W, 3) or encoded JPEG (T,), got {dataset.shape} {dataset.dtype}")

    frames = dataset.shape[0]
    if encoded_frames:
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - exercised by packaging
            raise RuntimeError("Install the data extra: pip install -e '.[data]'") from exc
        first_frame = _encoded_frame_bytes(dataset[0])
        with Image.open(BytesIO(first_frame)) as image:
            width, height = image.size
    else:
        _, height, width, _ = dataset.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
    ]
    if encoded_frames and not correct_legacy_jpeg_bgr:
        command.extend(["-f", "image2pipe", "-vcodec", "mjpeg", "-framerate", str(fps)])
    else:
        command.extend(["-f", "rawvideo", "-pix_fmt", "rgb24", "-s:v", f"{width}x{height}", "-r", str(fps)])
    command.extend(
        [
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ]
    )
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.stdin is None or process.stderr is None:  # pragma: no cover - Popen contract
        raise RuntimeError("Failed to open ffmpeg pipes")
    try:
        if encoded_frames and not correct_legacy_jpeg_bgr:
            for index in range(frames):
                process.stdin.write(_encoded_frame_bytes(dataset[index]))
        elif encoded_frames:
            for index in range(frames):
                with Image.open(BytesIO(_encoded_frame_bytes(dataset[index]))) as image:
                    restored_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)[..., ::-1]
                process.stdin.write(np.ascontiguousarray(restored_rgb).tobytes(order="C"))
        else:
            for start in range(0, frames, 16):
                batch = np.asarray(dataset[start : start + 16], dtype=np.uint8)
                process.stdin.write(batch.tobytes(order="C"))
    except BrokenPipeError:
        pass
    finally:
        process.stdin.close()
    stderr = process.stderr.read().decode("utf-8", errors="replace")
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"ffmpeg failed for {path}: {stderr.strip()}")
    return frames, height, width


def _encoded_frame_bytes(value: Any) -> bytes:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value)
    return np.asarray(value, dtype=np.uint8).tobytes(order="C")


def write_parquet(
    path: Path,
    *,
    state: np.ndarray,
    action: np.ndarray,
    episode_index: int,
    task_index: int,
    global_start: int,
    fps: float,
) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - exercised by packaging
        raise RuntimeError("Install the data extra: pip install -e '.[data]'") from exc

    frames = state.shape[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "observation.state": pa.array(state.tolist(), type=pa.list_(pa.float32(), 14)),
            "action": pa.array(action.tolist(), type=pa.list_(pa.float32(), 14)),
            "timestamp": pa.array(np.arange(frames, dtype=np.float32) / np.float32(fps)),
            "frame_index": pa.array(np.arange(frames), type=pa.int64()),
            "episode_index": pa.array(np.full(frames, episode_index), type=pa.int64()),
            "index": pa.array(np.arange(global_start, global_start + frames), type=pa.int64()),
            "task_index": pa.array(np.full(frames, task_index), type=pa.int64()),
            "annotation.human.task_description": pa.array(np.full(frames, task_index), type=pa.int64()),
            "next.reward": pa.array(np.zeros(frames, dtype=np.float32)),
            "next.done": pa.array(np.arange(frames) == frames - 1),
        }
    )
    pq.write_table(table, path, compression="zstd")


def convert_dataset(
    *,
    input_dir: Path,
    output_dir: Path,
    fps: float,
    start_episode: int,
    limit: int | None,
    instruction_index: int,
    task_fallback: str | None,
    manifest_path: Path | None,
    correct_legacy_jpeg_bgr: bool = False,
) -> dict[str, Any]:
    if fps <= 0:
        raise ValueError("fps must be positive")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to replace existing output: {output_dir}")

    demo_paths = select_demo_paths(input_dir, start_episode=start_episode, limit=limit)
    if not demo_paths:
        raise ValueError(f"No demo_*.hdf5 files found under {input_dir / 'data'}")

    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - exercised by packaging
        raise RuntimeError("Install the data extra: pip install -e '.[data]'") from exc

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}-", dir=output_dir.parent) as staging_raw:
        staging = Path(staging_raw)
        episodes: list[dict[str, Any]] = []
        tasks: list[dict[str, Any]] = []
        global_index = 0
        image_height = image_width = 0

        for episode_index, demo_path in enumerate(demo_paths):
            source_number = episode_number(demo_path)
            instruction = select_instruction(
                input_dir / "instructions" / f"demo_{source_number}.json",
                index=instruction_index,
                fallback=task_fallback,
            )
            with h5py.File(demo_path, "r") as handle:
                state = canonical_proprio(handle[POLICY_ROOT])
                action = collapse_isaac_action(handle[f"{POLICY_ROOT}/actions"][:])
                if state.shape[0] != action.shape[0]:
                    raise ValueError(f"State/action frame mismatch in {demo_path}")

                camera_group = handle[CAMERA_ROOT]
                for video_name, hdf5_name in CAMERA_MAP.items():
                    video_path = (
                        staging
                        / "videos"
                        / "chunk-000"
                        / f"observation.images.{video_name}"
                        / f"episode_{episode_index:06d}.mp4"
                    )
                    camera_frames, image_height, image_width = encode_video(
                        camera_group[hdf5_name],
                        video_path,
                        fps=fps,
                        correct_legacy_jpeg_bgr=correct_legacy_jpeg_bgr,
                    )
                    if camera_frames != state.shape[0]:
                        raise ValueError(f"Camera frame mismatch for {hdf5_name} in {demo_path}")

            write_parquet(
                staging / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet",
                state=state,
                action=action,
                episode_index=episode_index,
                task_index=episode_index,
                global_start=global_index,
                fps=fps,
            )
            episodes.append({"episode_index": episode_index, "tasks": [instruction], "length": state.shape[0]})
            tasks.append({"task_index": episode_index, "task": instruction})
            global_index += state.shape[0]

        write_json(staging / "meta" / "modality.json", modality_metadata())
        write_json(
            staging / "meta" / "info.json",
            dataset_info(
                episodes=len(episodes),
                frames=global_index,
                height=image_height,
                width=image_width,
                fps=fps,
            ),
        )
        write_jsonl(staging / "meta" / "episodes.jsonl", episodes)
        write_jsonl(staging / "meta" / "tasks.jsonl", tasks)
        Path(staging).rename(output_dir)

    manifest = {
        "status": "pass",
        "input": str(input_dir),
        "output": str(output_dir),
        "episodes": len(episodes),
        "frames": global_index,
        "fps": fps,
        "source_episode_ids": [episode_number(path) for path in demo_paths],
        "state_dim": 14,
        "action_dim": 14,
        "videos": len(episodes) * len(CAMERA_MAP),
        "video_codec": "h264",
        "correct_legacy_jpeg_bgr": correct_legacy_jpeg_bgr,
    }
    if manifest_path:
        write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=31.25)
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--instruction-index", type=int, default=0)
    parser.add_argument("--task-fallback")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--correct-legacy-jpeg-bgr", action="store_true")
    args = parser.parse_args()
    manifest = convert_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        fps=args.fps,
        start_episode=args.start_episode,
        limit=args.limit,
        instruction_index=args.instruction_index,
        task_fallback=args.task_fallback,
        manifest_path=args.manifest,
        correct_legacy_jpeg_bgr=args.correct_legacy_jpeg_bgr,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
