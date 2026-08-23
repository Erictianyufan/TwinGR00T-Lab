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
        """Return the nearest source action for compatibility callers."""
        return round(self.action_position(frame, action_count))

    def action_position(self, frame: int, action_count: int) -> float:
        """Map a frame to a continuous source-action coordinate."""
        if action_count <= 0:
            raise ValueError("action_count must be positive")
        start = self.feed_start
        stop = self.verify_start - 1
        fraction = float(np.clip((frame - start) / max(stop - start, 1), 0.0, 1.0))
        return fraction * (action_count - 1)


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def interpolate(start: np.ndarray, end: np.ndarray, fraction: float) -> np.ndarray:
    return np.asarray(start, dtype=float) + smoothstep(fraction) * (
        np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    )


def clearance_transfer(
    start: np.ndarray,
    end: np.ndarray,
    fraction: float,
    *,
    clearance_z: float = 1.18,
) -> np.ndarray:
    """Move up, translate, then lower so a parcel cannot cut through a bin."""
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    raised_start = start.copy()
    raised_end = end.copy()
    raised_start[2] = max(start[2], clearance_z)
    raised_end[2] = max(end[2], clearance_z)
    fraction = float(np.clip(fraction, 0.0, 1.0))
    if fraction < 0.25:
        return interpolate(start, raised_start, fraction / 0.25)
    if fraction < 0.75:
        return interpolate(raised_start, raised_end, (fraction - 0.25) / 0.5)
    return interpolate(raised_end, end, (fraction - 0.75) / 0.25)


def arched_transfer(
    start: np.ndarray,
    end: np.ndarray,
    fraction: float,
    *,
    arc_height: float = 0.16,
) -> np.ndarray:
    """Blend an inter-arm handoff along one smooth, raised spatial arc."""
    phase = smoothstep(fraction)
    position = np.asarray(start, dtype=float) + phase * (
        np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    )
    position[2] += max(float(arc_height), 0.0) * np.sin(np.pi * phase)
    return position


def smooth_sequence(values: np.ndarray, radius: int = 5) -> np.ndarray:
    """Apply an edge-preserving triangular temporal filter to a trajectory."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("values must have shape (frames, channels)")
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if radius == 0 or len(values) < 2:
        return values.copy()
    ascending = np.arange(1, radius + 2, dtype=float)
    kernel = np.concatenate((ascending, ascending[-2::-1]))
    kernel /= kernel.sum()
    padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
    return np.stack(
        [np.convolve(padded[:, channel], kernel, mode="valid") for channel in range(values.shape[1])],
        axis=1,
    )
