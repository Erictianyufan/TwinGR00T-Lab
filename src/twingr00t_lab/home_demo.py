"""Timeline and orchestration primitives for the continuous home-service demo."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class DemoPhase(StrEnum):
    ESTABLISH = "establish_home"
    TRANSPORT = "humanoid_transport"
    HANDOFF = "handoff"
    STACK = "aloha_stack"
    COMPLETE = "complete"


@dataclass(frozen=True)
class HomeDemoTimeline:
    """Frame boundaries for a single uninterrupted simulator timeline."""

    establish_frames: int = 120
    transport_frames: int = 180
    handoff_frames: int = 60
    stack_frames: int = 600
    complete_frames: int = 80

    def __post_init__(self) -> None:
        values = (
            self.establish_frames,
            self.transport_frames,
            self.handoff_frames,
            self.stack_frames,
            self.complete_frames,
        )
        if any(value <= 0 for value in values):
            raise ValueError("All phase durations must be positive")

    @property
    def transport_start(self) -> int:
        return self.establish_frames

    @property
    def handoff_start(self) -> int:
        return self.transport_start + self.transport_frames

    @property
    def stack_start(self) -> int:
        return self.handoff_start + self.handoff_frames

    @property
    def complete_start(self) -> int:
        return self.stack_start + self.stack_frames

    @property
    def total_frames(self) -> int:
        return self.complete_start + self.complete_frames

    def phase(self, frame: int) -> DemoPhase:
        self._validate_frame(frame)
        if frame < self.transport_start:
            return DemoPhase.ESTABLISH
        if frame < self.handoff_start:
            return DemoPhase.TRANSPORT
        if frame < self.stack_start:
            return DemoPhase.HANDOFF
        if frame < self.complete_start:
            return DemoPhase.STACK
        return DemoPhase.COMPLETE

    def aloha_action_index(self, frame: int) -> int | None:
        phase = self.phase(frame)
        if phase is DemoPhase.STACK:
            return frame - self.stack_start
        if phase is DemoPhase.COMPLETE:
            return self.stack_frames - 1
        return None

    def cargo_position(self, frame: int) -> np.ndarray:
        """Return the visual cargo root position; zero matches the recorded bowl poses."""
        phase = self.phase(frame)
        counter = np.asarray((-2.6, -1.8, 0.0), dtype=np.float32)
        start = np.asarray((-2.6, -1.8, 0.24), dtype=np.float32)
        staging = np.asarray((0.0, 0.0, 0.24), dtype=np.float32)
        placed = np.asarray((0.0, 0.0, 0.0), dtype=np.float32)
        hidden = np.asarray((0.0, 0.0, -8.0), dtype=np.float32)

        if phase is DemoPhase.ESTABLISH:
            # Keep the tray on the counter until the hands have visibly closed.
            pickup_start = min(66, self.establish_frames * 3 // 5)
            progress = (frame - pickup_start) / max(1, self.establish_frames - pickup_start - 1)
            return _mix(counter, start, _smoothstep(progress))
        if phase is DemoPhase.TRANSPORT:
            progress = (frame - self.transport_start) / max(1, self.transport_frames - 1)
            return _mix(start, staging, _smoothstep(progress))
        if phase is DemoPhase.HANDOFF:
            progress = (frame - self.handoff_start) / max(1, self.handoff_frames - 1)
            return _mix(staging, placed, _smoothstep(progress))
        return hidden

    def humanoid_position(self, frame: int) -> np.ndarray:
        phase = self.phase(frame)
        carrier_offset = np.asarray((-0.48, -0.08, 0.0), dtype=np.float32)
        if phase in (DemoPhase.ESTABLISH, DemoPhase.TRANSPORT, DemoPhase.HANDOFF):
            cargo = self.cargo_position(frame).copy()
            cargo[2] = 0.0
            return cargo + carrier_offset

        retreat_start = np.asarray((-0.48, -0.08, 0.0), dtype=np.float32)
        retreat_end = np.asarray((-1.65, 0.65, 0.0), dtype=np.float32)
        if phase is DemoPhase.STACK:
            stack_frame = frame - self.stack_start
            progress = min(1.0, stack_frame / 80.0)
            return _mix(retreat_start, retreat_end, _smoothstep(progress))
        return retreat_end

    def instruction(self, frame: int) -> str:
        phase = self.phase(frame)
        if phase is DemoPhase.ESTABLISH:
            reach, grip, lift = self.pickup_progress(frame)
            if reach < 0.98:
                return "G1: reach toward the loaded bowl tray"
            if grip < 0.98:
                return "G1: close both hands around the tray grips"
            if lift < 0.98:
                return "G1: lift the tray with all bowls visible"
            return "G1: stabilize the loaded tray before walking"
        return {
            DemoPhase.TRANSPORT: "Humanoid: carry the bowls from the kitchen",
            DemoPhase.HANDOFF: "Humanoid: place the tray in the handoff zone",
            DemoPhase.STACK: "ALOHA: stack all three bowls",
            DemoPhase.COMPLETE: "Cooperative task complete",
        }[phase]

    def pickup_progress(self, frame: int) -> tuple[float, float, float]:
        """Return smooth reach, finger-closure, and lift progress values."""

        self._validate_frame(frame)
        phase = self.phase(frame)
        if phase is not DemoPhase.ESTABLISH:
            done = 1.0 if phase in (DemoPhase.TRANSPORT, DemoPhase.HANDOFF) else 0.0
            return done, done, done
        reach = _smoothstep((frame - 12) / 34.0)
        grip = _smoothstep((frame - 42) / 24.0)
        lift = _smoothstep((frame - 66) / max(1, self.establish_frames - 67))
        return reach, grip, lift

    def pickup_camera_active(self, frame: int) -> bool:
        """Whether the overview camera is inside the dedicated hand close-up."""

        return self.phase(frame) is DemoPhase.ESTABLISH and 18 <= frame <= 112

    def overview_camera_pose(self, frame: int) -> tuple[np.ndarray, np.ndarray]:
        """Return a cinematic eye/target pair for the narrative overview camera."""
        phase = self.phase(frame)
        follow_offset = np.asarray((2.8, 3.4, 2.2), dtype=np.float32)
        target_offset = np.asarray((0.0, 0.2, 1.02), dtype=np.float32)
        carrier = self.humanoid_position(frame)
        follow_eye = carrier + follow_offset
        follow_target = carrier + target_offset

        shared_eye = np.asarray((3.4, 4.2, 2.65), dtype=np.float32)
        shared_target = np.asarray((-0.2, -0.45, 0.92), dtype=np.float32)
        aloha_eye = np.asarray((0.0, 1.35, 1.75), dtype=np.float32)
        aloha_target = np.asarray((0.0, -0.25, 0.76), dtype=np.float32)
        complete_eye = np.asarray((2.4, 3.0, 2.2), dtype=np.float32)
        complete_target = np.asarray((0.1, -0.35, 0.82), dtype=np.float32)

        if phase is DemoPhase.ESTABLISH:
            # Start wide, move into the hands and loaded bowls for the grasp,
            # then return to the full-body carrier view before locomotion.
            cargo = self.cargo_position(frame)
            close_eye = cargo + np.asarray((0.30, 1.45, 1.58), dtype=np.float32)
            close_target = cargo + np.asarray((0.0, 0.0, 0.80), dtype=np.float32)
            enter = _smoothstep((frame - 12) / 16.0)
            leave = _smoothstep((frame - 102) / 17.0)
            amount = enter * (1.0 - leave)
            return _mix(follow_eye, close_eye, amount), _mix(follow_target, close_target, amount)
        if phase is DemoPhase.TRANSPORT:
            return follow_eye, follow_target
        if phase is DemoPhase.HANDOFF:
            last_carrier = self.humanoid_position(self.handoff_start - 1)
            start_eye = last_carrier + follow_offset
            start_target = last_carrier + target_offset
            progress = (frame - self.handoff_start) / max(1, self.handoff_frames - 1)
            amount = _smoothstep(progress)
            return _mix(start_eye, shared_eye, amount), _mix(start_target, shared_target, amount)
        if phase is DemoPhase.STACK:
            transition_frames = min(80, self.stack_frames // 4)
            progress = (frame - self.stack_start) / max(1, transition_frames - 1)
            amount = _smoothstep(progress)
            return _mix(shared_eye, aloha_eye, amount), _mix(shared_target, aloha_target, amount)

        progress = (frame - self.complete_start) / max(1, self.complete_frames - 1)
        amount = _smoothstep(progress)
        return _mix(aloha_eye, complete_eye, amount), _mix(aloha_target, complete_target, amount)

    def _validate_frame(self, frame: int) -> None:
        if not 0 <= frame < self.total_frames:
            raise IndexError(f"frame must be in [0, {self.total_frames}), got {frame}")


def _smoothstep(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value * value * (3.0 - 2.0 * value)


def _mix(start: np.ndarray, end: np.ndarray, amount: float) -> np.ndarray:
    return np.asarray(start + (end - start) * amount, dtype=np.float32)
