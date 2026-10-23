"""CPU-side contracts for physically re-recorded factory teacher trajectories.

These are replay-derived training data, never model-inference demonstrations.
"""

from __future__ import annotations

import hashlib
import math
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

from twingr00t_lab.dataset import collapse_isaac_action

CONTROLLER_SOURCE = "teacher_replay_for_training"
ALIGNMENT = "post_step_obs_with_executed_action"
POLICY_CAMERAS = ("head_camera_rgb", "left_camera_rgb", "right_camera_rgb")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replay_plan(actions: np.ndarray, *, step_dt: float, stable_tail_s: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Return untouched source targets plus real final-target execution steps.

    This plans commands only. It must never be used to duplicate observations.
    """
    actions = np.asarray(actions, dtype=np.float32)
    collapse_isaac_action(actions)
    if not len(actions):
        raise ValueError("The source must contain at least one action")
    if not math.isfinite(step_dt) or step_dt <= 0:
        raise ValueError("step_dt must be finite and positive")
    if not math.isfinite(stable_tail_s) or stable_tail_s < 1.0:
        raise ValueError("At least one second of actual stable-tail execution is required")
    tail_steps = math.ceil(stable_tail_s / step_dt)
    indices = np.concatenate((np.arange(len(actions)), np.full(tail_steps, len(actions) - 1))).astype(np.int64)
    return np.ascontiguousarray(actions[indices]), indices


def rgb_uint8(value: Any) -> np.ndarray:
    """Accept one RGB/RGBA camera frame without RGB/BGR swapping or guessing."""
    array = np.asarray(value)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3 or array.shape[-1] not in (3, 4) or min(array.shape[:2]) < 1:
        raise ValueError(f"Expected one HxWxRGB(A) frame, got {array.shape}")
    array = array[..., :3]
    if array.dtype == np.uint8:
        return np.ascontiguousarray(array)
    if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
        raise ValueError("Camera must be uint8 RGB or finite normalized floating-point RGB")
    if array.min() < 0 or array.max() > 1:
        raise ValueError("Floating-point RGB must be explicitly normalized to [0, 1]")
    return np.ascontiguousarray(np.rint(array * 255).astype(np.uint8))


def encode_rgb_jpeg(value: Any, *, quality: int = 95) -> np.ndarray:
    """Encode RGB directly with Pillow, unlike the legacy OpenCV BGR path."""
    from PIL import Image

    if isinstance(quality, bool) or not isinstance(quality, int) or not 1 <= quality <= 100:
        raise ValueError("JPEG quality must be an integer in [1, 100]")
    buffer = BytesIO()
    Image.fromarray(rgb_uint8(value)).save(buffer, format="JPEG", quality=quality, subsampling=0)
    return np.frombuffer(buffer.getvalue(), dtype=np.uint8)


def named_eef_indices(frame_names: list[str]) -> tuple[int, int]:
    expected = ("left_end_effector", "right_end_effector")
    if any(frame_names.count(name) != 1 for name in expected):
        raise ValueError(f"Need unique left/right end-effector names, got {frame_names}")
    return tuple(frame_names.index(name) for name in expected)


def collection_metadata(
    *,
    source_demo: Path,
    source_instruction: Path,
    episode_id: int,
    source_steps: int,
    total_steps: int,
    step_dt: float,
    cameras: tuple[str, ...],
) -> dict:
    tail_steps = total_steps - source_steps
    if source_steps < 1 or tail_steps < 1 or tail_steps * step_dt < 1.0 - 1e-10:
        raise ValueError("Collection must execute the complete source and at least one second of tail")
    if not math.isfinite(step_dt) or step_dt <= 0:
        raise ValueError("step_dt must be finite and positive")
    if not set(POLICY_CAMERAS).issubset(cameras) or len(set(cameras)) != len(cameras):
        raise ValueError("Unique head/left-wrist/right-wrist cameras are required")
    return {
        "schema_version": 1,
        "controller_source": CONTROLLER_SOURCE,
        "replay_used": True,
        "model_inference_result": False,
        "training_eligible": False,
        "scene_style": "factory",
        "source_episode_id": episode_id,
        "source_demo": str(source_demo.resolve()),
        "source_demo_sha256": sha256_file(source_demo),
        "source_instruction": str(source_instruction.resolve()),
        "source_instruction_sha256": sha256_file(source_instruction),
        "instruction_sha256": sha256_file(source_instruction),
        "source_action_steps": source_steps,
        "tail_steps": tail_steps,
        "tail_duration_s": tail_steps * step_dt,
        "planned_total_steps": total_steps,
        "total_steps": 0,
        "control_fps": 1.0 / step_dt,
        "step_dt": step_dt,
        "action_dim": 16,
        "state_dim": 14,
        "observation_action_alignment": ALIGNMENT,
        "recommended_observation_action_offset": 1,
        "camera_encoding": "jpeg_rgb_pillow",
        "correct_legacy_jpeg_bgr": False,
        "cameras": list(cameras),
        "eef_source": "ee_frame.target_pos_w/target_quat_w indexed by unique frame names",
        "stable_tail_source": "real_env_step_with_last_source_target_not_duplicated_frames",
        "physics_acceptance_config": "official_GraspAcceptanceConfig_defaults",
        "direct_contact_sensor_verified": False,
    }


class TeacherHdf5Writer:
    """Stream post-step measurements/JPEGs into a diagnostic HDF5 initially.

    Promotion to the training directory happens only after physical acceptance.
    This class never copies source camera frames or source post-step states.
    """

    def __init__(self, path: Path, source: Path, metadata: dict, initial_snapshot: dict[str, np.ndarray]):
        import h5py

        self.handle = h5py.File(path, "x")
        self.rows = 0
        self.cameras = tuple(metadata["cameras"])
        self._keys: set[str] | None = None
        try:
            with h5py.File(source, "r") as original:
                original.copy("initial_state", self.handle)
            for key, value in initial_snapshot.items():
                target = f"initial_state/{key}"
                if target in self.handle:
                    del self.handle[target]
                self.handle.create_dataset(target, data=np.asarray(value)[None])
            for key in (
                "controller_source",
                "replay_used",
                "model_inference_result",
                "scene_style",
                "source_episode_id",
                "source_demo_sha256",
                "control_fps",
                "step_dt",
                "observation_action_alignment",
                "recommended_observation_action_offset",
                "camera_encoding",
                "correct_legacy_jpeg_bgr",
            ):
                self.handle.attrs[key] = metadata[key]
            self.handle.attrs["success"] = False
            self.handle.attrs["num_samples"] = 0
            for camera in self.cameras:
                dataset = self.handle.create_dataset(
                    f"observations/camera_obs/{camera}",
                    shape=(0,),
                    maxshape=(None,),
                    dtype=h5py.vlen_dtype(np.dtype("uint8")),
                    chunks=(1,),
                )
                dataset.attrs["encoding"] = "jpeg_rgb_pillow"
                dataset.attrs["color_space"] = "RGB"
                dataset.attrs["correct_legacy_jpeg_bgr"] = False
        except BaseException:
            self.handle.close()
            raise

    def append(self, values: dict[str, Any], cameras: dict[str, Any], *, jpeg_quality: int = 95) -> None:
        numeric = {key: np.asarray(value) for key, value in values.items()}
        if self._keys is not None and set(numeric) != self._keys:
            raise ValueError("Every post-step record must contain identical measurement fields")
        for key, array in numeric.items():
            if not np.issubdtype(array.dtype, np.number) and array.dtype != np.bool_:
                raise ValueError(f"Non-numeric measurement: {key}")
            if not np.isfinite(array).all():
                raise ValueError(f"Non-finite measurement: {key}")
            if key in self.handle and self.handle[key].shape[1:] != array.shape:
                raise ValueError(f"Measurement shape changed: {key}")
        encoded = {camera: encode_rgb_jpeg(cameras[camera], quality=jpeg_quality) for camera in self.cameras}
        self._keys = set(numeric)
        for key, array in numeric.items():
            if key not in self.handle:
                self.handle.create_dataset(
                    key, shape=(0, *array.shape), maxshape=(None, *array.shape), dtype=array.dtype
                )
            dataset = self.handle[key]
            dataset.resize(self.rows + 1, axis=0)
            dataset[self.rows] = array
        for camera, jpeg in encoded.items():
            dataset = self.handle[f"observations/camera_obs/{camera}"]
            dataset.resize(self.rows + 1, axis=0)
            dataset[self.rows] = jpeg
        self.rows += 1
        self.handle.attrs["num_samples"] = self.rows
        if self.rows % 32 == 0:
            self.handle.flush()

    def close(self, *, accepted: bool = False) -> None:
        self.handle.attrs["success"] = bool(accepted)
        self.handle.attrs["num_samples"] = self.rows
        self.handle.close()
