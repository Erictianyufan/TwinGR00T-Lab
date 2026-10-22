"""Translate between RoboTwin observations/actions and the GR00T policy contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

CAMERA_MAP = {
    "head": "head_camera_rgb",
    "left_wrist": "left_camera_rgb",
    "right_wrist": "right_camera_rgb",
}
STATE_KEYS = ("left_arm", "left_gripper", "right_arm", "right_gripper")
POLICY_STATE_MAP = {
    "left_arm": "left_arm_pos",
    "left_gripper": "left_gripper_pos",
    "right_arm": "right_arm_pos",
    "right_gripper": "right_gripper_pos",
}
LANGUAGE_KEY = "annotation.human.task_description"


def to_numpy(value: Any) -> np.ndarray:
    """Convert NumPy or Torch-like values without importing Torch in the client package."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _batch_time(value: Any, *, dtype: np.dtype[Any], name: str) -> np.ndarray:
    array = to_numpy(value).astype(dtype, copy=False)
    if array.ndim == 1:
        array = array[None, None, :]
    elif array.ndim == 2:
        array = array[:, None, :]
    else:
        raise ValueError(f"{name} must be shaped (D) or (B,D), got {array.shape}")
    return np.ascontiguousarray(array)


def _video_batch_time(value: Any, *, name: str) -> np.ndarray:
    array = to_numpy(value)
    if array.ndim == 3:
        array = array[None, None, ...]
    elif array.ndim == 4:
        array = array[:, None, ...]
    else:
        raise ValueError(f"{name} must be shaped (H,W,3) or (B,H,W,3), got {array.shape}")
    if array.shape[-1] != 3:
        raise ValueError(f"{name} must contain RGB channels, got {array.shape}")
    if array.dtype != np.uint8:
        if not np.isfinite(array).all() or array.min() < 0 or array.max() > 255:
            raise ValueError(f"{name} cannot be represented as uint8 RGB")
        array = array.astype(np.uint8)
    return np.ascontiguousarray(array)


def build_groot_observation(
    policy_observation: Mapping[str, Any],
    camera_observation: Mapping[str, Any],
    prompt: str,
    *,
    swap_rgb_bgr: bool = False,
) -> dict[str, Any]:
    """Build the nested, batched GR00T observation for one simulator timestep."""
    if not prompt.strip():
        raise ValueError("prompt must not be empty")

    video = {
        output_key: _video_batch_time(camera_observation[input_key], name=input_key)
        for output_key, input_key in CAMERA_MAP.items()
    }
    if swap_rgb_bgr:
        # Legacy RoboTwin recording called cv2.imencode on RGB arrays without
        # converting to BGR. Its JPEG training data therefore has R/B swapped.
        # Match that checkpoint's camera contract without recoloring the video.
        video = {key: np.ascontiguousarray(value[..., ::-1]) for key, value in video.items()}
    state = {
        output_key: _batch_time(policy_observation[input_key], dtype=np.dtype(np.float32), name=input_key)
        for output_key, input_key in POLICY_STATE_MAP.items()
    }
    batch_sizes = {array.shape[0] for array in (*video.values(), *state.values())}
    if len(batch_sizes) != 1:
        raise ValueError(f"Observation batch sizes disagree: {sorted(batch_sizes)}")
    batch_size = batch_sizes.pop()
    language = {LANGUAGE_KEY: [[prompt] for _ in range(batch_size)]}
    return {"video": video, "state": state, "language": language}


def groot_action_to_isaac(action: Mapping[str, Any]) -> np.ndarray:
    """Expand a GR00T 14-D logical action chunk into RoboTwin's 16 Isaac joints."""
    groups: dict[str, np.ndarray] = {}
    for key in STATE_KEYS:
        array = to_numpy(action[key]).astype(np.float32, copy=False)
        if array.ndim == 2:
            array = array[None, ...]
        if array.ndim != 3:
            raise ValueError(f"Action {key} must be shaped (T,D) or (B,T,D), got {array.shape}")
        groups[key] = array

    expected_dims = {"left_arm": 6, "left_gripper": 1, "right_arm": 6, "right_gripper": 1}
    batch_time = {(array.shape[0], array.shape[1]) for array in groups.values()}
    if len(batch_time) != 1:
        raise ValueError(f"Action batch/horizon disagree: {sorted(batch_time)}")
    for key, expected in expected_dims.items():
        if groups[key].shape[2] != expected:
            raise ValueError(f"Action {key} must have {expected} values, got {groups[key].shape}")

    result = np.concatenate(
        (
            groups["left_arm"],
            np.repeat(groups["left_gripper"], 2, axis=2),
            groups["right_arm"],
            np.repeat(groups["right_gripper"], 2, axis=2),
        ),
        axis=2,
    )
    if not np.isfinite(result).all():
        raise ValueError("Action contains NaN or Inf")
    return np.ascontiguousarray(result)
