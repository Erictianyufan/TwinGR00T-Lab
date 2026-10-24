"""Empirical joint-target guard for N1.7-to-ALOHA action adaptation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def action_bounds_from_stats(path: Path, *, arm_margin_rad: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Expand 14-D training bounds into 16 physical joint targets.

    The arm margin is capped at 0.5 rad. Gripper bounds remain unchanged so an
    arm-only diagnostic cannot silently alter grasp/release control.
    """
    if not np.isfinite(arm_margin_rad) or not 0.1 <= arm_margin_rad <= 0.5:
        raise ValueError("arm_margin_rad must be finite and between 0.1 and 0.5")
    statistics = json.loads(Path(path).read_text(encoding="utf-8"))["action"]
    lower = np.asarray(statistics["min"], dtype=np.float32)
    upper = np.asarray(statistics["max"], dtype=np.float32)
    if lower.shape != (14,) or upper.shape != (14,) or not np.isfinite(lower).all() or not np.all(lower <= upper):
        raise ValueError("Expected valid 14-D action limits from the training split")

    def expand(values: np.ndarray) -> np.ndarray:
        return np.concatenate((values[:6], np.repeat(values[6], 2), values[7:13], np.repeat(values[13], 2)))

    margin = np.asarray([arm_margin_rad] * 6 + [0.0] * 2 + [arm_margin_rad] * 6 + [0.0] * 2, dtype=np.float32)
    return expand(lower) - margin, expand(upper) + margin
