"""Deterministic Unitree G1 motion used by the merged cinematic demo.

The root trajectory comes from the shared home-service timeline.  GR00T owns
the semantic skill boundary, while this module supplies a stable visual gait
until a SONIC whole-body-controller checkpoint is attached.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from twingr00t_lab.home_demo import DemoPhase, HomeDemoTimeline


@dataclass(frozen=True)
class G1MotionSample:
    """One frame of root and named-joint targets for the G1 articulation."""

    root_position: np.ndarray
    root_quaternion_wxyz: np.ndarray
    joint_offsets: dict[str, float]
    gait_active: bool


@dataclass(frozen=True)
class G1MotionPlan:
    """Generate a visible, cyclic leg gait along the HomeRelay route."""

    root_height: float = 0.74
    stride_frames: float = 22.0
    hip_amplitude: float = 0.24
    knee_amplitude: float = 0.30

    def sample(self, frame: int, timeline: HomeDemoTimeline) -> G1MotionSample:
        phase = timeline.phase(frame)
        root = timeline.humanoid_position(frame).astype(np.float32)
        root[2] = self.root_height

        gait_active = phase is DemoPhase.TRANSPORT or (
            phase is DemoPhase.STACK and frame - timeline.stack_start < 80
        )
        gait_phase = 2.0 * math.pi * frame / self.stride_frames
        envelope = self._gait_envelope(frame, timeline) if gait_active else 0.0
        stride = math.sin(gait_phase) * envelope
        left_swing = max(0.0, stride)
        right_swing = max(0.0, -stride)

        if gait_active:
            root[2] += 0.012 * envelope * (1.0 - math.cos(2.0 * gait_phase))

        carry = self._carry_envelope(frame, timeline)
        grasp = self._grasp_envelope(frame, timeline)
        offsets = {
            "left_hip_pitch_joint": self.hip_amplitude * stride,
            "right_hip_pitch_joint": -self.hip_amplitude * stride,
            "left_knee_joint": self.knee_amplitude * left_swing,
            "right_knee_joint": self.knee_amplitude * right_swing,
            "left_ankle_pitch_joint": -0.12 * stride - 0.16 * left_swing,
            "right_ankle_pitch_joint": 0.12 * stride - 0.16 * right_swing,
            "left_hip_roll_joint": 0.035 * math.sin(gait_phase) * envelope,
            "right_hip_roll_joint": 0.035 * math.sin(gait_phase) * envelope,
            # Both hands stay in front of the torso while carrying the tray.
            "left_shoulder_pitch_joint": -0.88 * carry - 0.06 * stride,
            "right_shoulder_pitch_joint": -0.88 * carry + 0.06 * stride,
            "left_shoulder_roll_joint": 0.12 * carry,
            "right_shoulder_roll_joint": -0.12 * carry,
            "left_elbow_pitch_joint": -0.60 * carry,
            "right_elbow_pitch_joint": -0.60 * carry,
            # Bring the forearms toward the two tray grips during the close-up.
            "left_elbow_roll_joint": -0.20 * carry,
            "right_elbow_roll_joint": 0.20 * carry,
            "left_shoulder_yaw_joint": -0.10 * carry,
            "right_shoulder_yaw_joint": 0.10 * carry,
            # The official Isaac Lab G1 asset includes articulated three-finger
            # hands.  Opposite signs mirror the left and right mechanisms.
            "left_zero_joint": 0.24 * grasp,
            "right_zero_joint": -0.24 * grasp,
            "left_one_joint": 0.30 * grasp,
            "right_one_joint": -0.30 * grasp,
            "left_two_joint": 0.18 * grasp,
            "right_two_joint": -0.18 * grasp,
            "left_three_joint": 0.42 * grasp,
            "right_three_joint": -0.42 * grasp,
            "left_four_joint": 0.36 * grasp,
            "right_four_joint": -0.36 * grasp,
            "left_five_joint": 0.32 * grasp,
            "right_five_joint": -0.32 * grasp,
            "left_six_joint": 0.28 * grasp,
            "right_six_joint": -0.28 * grasp,
        }
        # The tray sits directly in front of G1 along +X at pickup.  Keeping the
        # torso aligned with +X makes the arm reach and finger closure readable.
        orientation = np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32)
        return G1MotionSample(root, orientation, offsets, gait_active)

    def _gait_envelope(self, frame: int, timeline: HomeDemoTimeline) -> float:
        phase = timeline.phase(frame)
        if phase is DemoPhase.TRANSPORT:
            local = frame - timeline.transport_start
            return min(1.0, local / 18.0, (timeline.transport_frames - 1 - local) / 18.0)
        if phase is DemoPhase.STACK:
            local = frame - timeline.stack_start
            return min(1.0, local / 12.0, (79 - local) / 12.0)
        return 0.0

    @staticmethod
    def _carry_envelope(frame: int, timeline: HomeDemoTimeline) -> float:
        phase = timeline.phase(frame)
        if phase is DemoPhase.ESTABLISH:
            reach, _, _ = timeline.pickup_progress(frame)
            return reach
        if phase is DemoPhase.TRANSPORT:
            return 1.0
        if phase is DemoPhase.HANDOFF:
            local = frame - timeline.handoff_start
            return 1.0 - 0.55 * _smoothstep(local / max(1, timeline.handoff_frames - 1))
        return 0.0

    @staticmethod
    def _grasp_envelope(frame: int, timeline: HomeDemoTimeline) -> float:
        phase = timeline.phase(frame)
        if phase is DemoPhase.ESTABLISH:
            _, grip, _ = timeline.pickup_progress(frame)
            return grip
        if phase is DemoPhase.TRANSPORT:
            return 1.0
        if phase is DemoPhase.HANDOFF:
            local = frame - timeline.handoff_start
            release = _smoothstep((local - timeline.handoff_frames * 0.55) / max(1, timeline.handoff_frames * 0.35))
            return 1.0 - release
        return 0.0


def apply_named_offsets(
    default_positions: np.ndarray,
    joint_names: list[str] | tuple[str, ...],
    offsets: dict[str, float],
) -> np.ndarray:
    """Apply named offsets without assuming a particular G1 joint ordering."""

    result = np.asarray(default_positions, dtype=np.float32).copy()
    if result.ndim != 1 or len(result) != len(joint_names):
        raise ValueError("default_positions and joint_names must describe one matching joint vector")
    indices = {name: index for index, name in enumerate(joint_names)}
    for name, offset in offsets.items():
        if name in indices:
            result[indices[name]] += float(offset)
    return result


def _smoothstep(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value * value * (3.0 - 2.0 * value)
