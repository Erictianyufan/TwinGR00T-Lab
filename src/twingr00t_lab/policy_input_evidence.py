"""CPU-only scene labels and snapshots of the actual model-facing input."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from twingr00t_lab.policy_bridge import CAMERA_MAP, to_numpy

SCENE_STYLES = ("factory", "no-factory-dressing")


def scene_evidence(scene_style: str) -> dict[str, Any]:
    if scene_style not in SCENE_STYLES:
        raise ValueError(f"Unknown scene style: {scene_style}")
    diagnostic = scene_style == "no-factory-dressing"
    return {
        "scene_style": scene_style,
        "scene_diagnostic_only": diagnostic,
        "source_scene_equivalence_verified": False,
        "scene_description": (
            "Diagnostic: shared training HDR light, no added factory floor/walls/racks; "
            "not a verified reproduction of all source-scene settings"
            if diagnostic
            else "Factory floor/walls/racks with shared training HDR light"
        ),
    }


def scene_video_title(scene_style: str, initial_state_split: str) -> str:
    scene_evidence(scene_style)
    label = "Factory sorting" if scene_style == "factory" else "DIAGNOSTIC: no factory dressing"
    return f"TwinGR00T Lab | GR00T N1.7 online inference | {label} ({initial_state_split} initial state)"


def array_statistics(value: Any) -> dict[str, Any]:
    array = to_numpy(value)
    if not array.size or not np.isfinite(array).all():
        raise ValueError("Camera evidence requires non-empty finite values")
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def save_policy_input_evidence(
    policy_input: Mapping[str, Any],
    raw_camera_observation: Mapping[str, Any],
    output_video: Path,
    *,
    scene_style: str,
    policy_call_pending: bool,
) -> dict[str, Any]:
    """Save the first bridge result without transforming or normalizing it.

    Both pre-bridge and model-facing ranges are retained: a nice recording
    cannot establish that a separate model-input conversion was correct.
    ``policy_call_pending=False`` explicitly identifies snapshot-only runs.
    """
    from PIL import Image

    metadata = {
        **scene_evidence(scene_style),
        "control_step": 0,
        "capture_stage": "bridge_output_before_first_policy_call",
        "policy_call_pending": policy_call_pending,
        "cameras": {},
    }
    output_video = Path(output_video)
    metadata_path = output_video.with_suffix(".policy_input.json")
    prepared = []
    for camera_name, raw_key in CAMERA_MAP.items():
        frames = np.asarray(policy_input["video"][camera_name])
        if frames.dtype != np.uint8 or frames.ndim != 5 or frames.shape[:2] != (1, 1) or frames.shape[-1] != 3:
            raise ValueError(f"Expected uint8 (1,1,H,W,3) model input for {camera_name}, got {frames.shape}")
        model_stats = array_statistics(frames)
        raw_stats = array_statistics(raw_camera_observation[raw_key])
        png = output_video.with_name(f"{output_video.stem}.{camera_name}.png")
        prepared.append((camera_name, frames[0, 0], png, model_stats, raw_stats))
    for path in (metadata_path, *(item[2] for item in prepared)):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite policy input evidence: {path}")
    output_video.parent.mkdir(parents=True, exist_ok=True)
    for camera_name, frame, png, model_stats, raw_stats in prepared:
        Image.fromarray(frame).save(png)
        metadata["cameras"][camera_name] = {
            "model_input": model_stats,
            "pre_bridge": raw_stats,
            "png": str(png),
            "png_sha256": hashlib.sha256(png.read_bytes()).hexdigest(),
            "model_input_range_le_one": model_stats["max"] <= 1,
        }
    metadata["metadata_path"] = str(metadata_path)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata
