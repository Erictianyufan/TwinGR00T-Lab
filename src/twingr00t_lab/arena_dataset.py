"""Audit an IsaacLab-Arena Mimic HDF5 dataset for reproducible experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

REQUIRED_SIGNALS = (
    "grasp_and_idle_left",
    "grasp_and_idle_right",
    "idle_left",
    "idle_right",
    "navigate_to_bin",
    "navigate_to_table",
    "navigate_turn_inplace",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rise_frames(values: np.ndarray) -> list[int]:
    signal = np.asarray(values, dtype=bool).reshape(-1)
    return (np.flatnonzero(signal[1:].astype(np.int8) - signal[:-1].astype(np.int8) == 1) + 1).tolist()


def audit_arena_dataset(path: str | Path) -> dict:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install the data extra: pip install -e '.[data]'") from exc

    dataset_path = Path(path)
    episodes: list[dict] = []
    with h5py.File(dataset_path, "r") as handle:
        if "data" not in handle:
            raise ValueError("Missing root group: data")
        data = handle["data"]
        env_args = json.loads(data.attrs["env_args"])
        for name in sorted(data, key=lambda value: int(value.rsplit("_", 1)[1])):
            demo = data[name]
            frames = int(demo.attrs["num_samples"])
            if demo["actions"].shape != (frames, 23):
                raise ValueError(f"{name}: expected actions ({frames}, 23), got {demo['actions'].shape}")
            datagen = demo["obs/datagen_info"]
            if "brown_box" not in datagen["object_pose"]:
                raise ValueError(f"{name}: missing task-facing object pose brown_box")
            signals = datagen["subtask_term_signals"]
            missing = [key for key in REQUIRED_SIGNALS if key not in signals]
            if missing:
                raise ValueError(f"{name}: missing subtask signals {missing}")
            transitions = {}
            for key in REQUIRED_SIGNALS:
                values = np.asarray(signals[key][:], dtype=bool).reshape(-1)
                if values.size != frames:
                    raise ValueError(f"{name}: signal {key} has {values.size} values, expected {frames}")
                transitions[key] = _rise_frames(values)
            episodes.append(
                {
                    "name": name,
                    "frames": frames,
                    "success": bool(demo.attrs.get("success", False)),
                    "action_dim": int(demo["actions"].shape[1]),
                    "subtask_rise_frames": transitions,
                }
            )

    return {
        "schema_version": 1,
        "path": str(dataset_path),
        "size_bytes": dataset_path.stat().st_size,
        "sha256": _sha256(dataset_path),
        "environment": env_args,
        "episodes": len(episodes),
        "successful_episodes": sum(row["success"] for row in episodes),
        "total_frames": sum(row["frames"] for row in episodes),
        "episode_details": episodes,
    }
