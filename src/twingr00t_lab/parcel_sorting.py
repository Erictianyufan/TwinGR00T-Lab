"""Deterministic timeline contract for the warehouse parcel-sorting demo."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class ParcelPhase(Enum):
    ESTABLISH = "warehouse overview"
    CONVEYOR_FEED = "parcels arriving on conveyor"
    DUAL_PICK = "dual-arm parcel pickup"
    SORT = "cooperative parcel sorting"
    VERIFY = "sorted parcels and workcell overview"


@dataclass(frozen=True)
class ParcelSortingTimeline:
    """Frame ranges used by both the renderer and its acceptance manifest."""

    establish_frames: int = 45
    feed_frames: int = 105
    pick_frames: int = 90
    sort_frames: int = 120
    verify_frames: int = 60

    @property
    def feed_start(self) -> int:
        return self.establish_frames

    @property
    def pick_start(self) -> int:
        return self.feed_start + self.feed_frames

    @property
    def sort_start(self) -> int:
        return self.pick_start + self.pick_frames

    @property
    def verify_start(self) -> int:
        return self.sort_start + self.sort_frames

    @property
    def total_frames(self) -> int:
        return self.verify_start + self.verify_frames

    def phase(self, frame: int) -> ParcelPhase:
        if not 0 <= frame < self.total_frames:
            raise IndexError(f"frame {frame} is outside [0, {self.total_frames})")
        if frame < self.feed_start:
            return ParcelPhase.ESTABLISH
        if frame < self.pick_start:
            return ParcelPhase.CONVEYOR_FEED
        if frame < self.sort_start:
            return ParcelPhase.DUAL_PICK
        if frame < self.verify_start:
            return ParcelPhase.SORT
        return ParcelPhase.VERIFY

    def action_index(self, frame: int, action_count: int) -> int:
        """Map the active manipulation interval monotonically onto a demo."""
        if action_count <= 0:
            raise ValueError("action_count must be positive")
        start = self.feed_start
        stop = self.verify_start - 1
        fraction = float(np.clip((frame - start) / max(stop - start, 1), 0.0, 1.0))
        return min(round(fraction * (action_count - 1)), action_count - 1)


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def interpolate(start: np.ndarray, end: np.ndarray, fraction: float) -> np.ndarray:
    return np.asarray(start, dtype=float) + smoothstep(fraction) * (
        np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    )
