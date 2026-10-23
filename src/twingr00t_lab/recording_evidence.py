"""Content hashes and frame-accounting metadata for closed-loop recordings."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np


def action_sha256(values: np.ndarray) -> str:
    """Hash the canonical C-order float32 physical-joint action representation."""
    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.float32).tobytes()).hexdigest()


def video_evidence(path: Path, *, steps: int, record_every: int, record_fps: float, frames_written: int) -> dict:
    """Report pipe frame accounting; the independent verifier must ffprobe MP4."""
    if steps < 1 or record_every < 1 or not math.isfinite(record_fps) or record_fps <= 0:
        raise ValueError("Video timing and control-step counts must be positive")
    expected = (steps + record_every - 1) // record_every
    if frames_written != expected:
        raise ValueError(f"Wrote {frames_written} video frames but expected {expected}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "video_sha256": digest.hexdigest(),
        "video_frame_count": frames_written,
        "video_frame_count_source": "successful_pipe_writes_requires_independent_ffprobe",
        "record_every": record_every,
        "expected_video_frames": expected,
        "expected_video_duration_s": expected / record_fps,
    }
