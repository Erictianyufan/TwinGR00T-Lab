"""Observation-driven language subtask routing for the three-block ALOHA demo.

The router selects a prompt; it never produces a joint action. All physical
actions must still come from the GR00T policy. A transition requires sustained
object/gripper evidence, not a timer or a saved demonstration frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

COLORS = ("red", "green", "blue")
DEFAULT_ARM_ORDER = ("right", "left", "right")


def phase_prompts(arm_order: tuple[str, str, str] = DEFAULT_ARM_ORDER) -> tuple[str, ...]:
    if len(arm_order) != 3 or any(arm not in ("left", "right") for arm in arm_order):
        raise ValueError("The three color phases need left/right arm assignments")
    return (
        f"Pick up the red block with the {arm_order[0]} arm.",
        "Place the red block on the left side of the row and release it.",
        f"Pick up the green block with the {arm_order[1]} arm.",
        "Place the green block in the middle, beside the red block, and release it.",
        f"Pick up the blue block with the {arm_order[2]} arm.",
        "Place the blue block on the far right and release it.",
    )


@dataclass
class ThreeBlockPhaseRouter:
    """Advance one subtask at a time from observed object and gripper state."""

    initial_positions: np.ndarray
    arm_order: tuple[str, str, str] = DEFAULT_ARM_ORDER
    pickup_confirmation_steps: int = 5
    place_confirmation_steps: int = 31
    phase_index: int = 0
    confirmation_count: int = 0
    events: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        initial = np.asarray(self.initial_positions, dtype=np.float32)
        if initial.shape != (3, 3) or not np.isfinite(initial).all():
            raise ValueError("initial_positions must be three finite XYZ positions")
        if min(self.pickup_confirmation_steps, self.place_confirmation_steps) < 1:
            raise ValueError("Confirmation windows must be positive")
        self.initial_positions = initial.copy()
        self.prompts = phase_prompts(self.arm_order)

    @property
    def prompt(self) -> str:
        return self.prompts[self.phase_index]

    def observe(
        self,
        *,
        block_positions: np.ndarray,
        block_linear_velocities: np.ndarray,
        eef_positions: np.ndarray,
        gripper_positions: np.ndarray,
        control_step: int,
    ) -> bool:
        """Return true only when sustained physical evidence changes the prompt."""
        if type(control_step) is not int or control_step < 0:
            raise ValueError("control_step must be nonnegative")
        blocks = np.asarray(block_positions, dtype=np.float32)
        velocities = np.asarray(block_linear_velocities, dtype=np.float32)
        eef = np.asarray(eef_positions, dtype=np.float32)
        grippers = np.asarray(gripper_positions, dtype=np.float32)
        if (
            blocks.shape != (3, 3)
            or velocities.shape != (3, 3)
            or eef.shape != (2, 3)
            or grippers.shape != (2,)
            or not all(np.isfinite(value).all() for value in (blocks, velocities, eef, grippers))
        ):
            raise ValueError("Invalid router observation shapes or non-finite values")
        color_index = self.phase_index // 2
        arm_index = 0 if self.arm_order[color_index] == "left" else 1
        block = blocks[color_index]
        lift = float(block[2] - self.initial_positions[color_index, 2])
        if self.phase_index % 2 == 0:  # pick -> place
            condition = (
                lift >= 0.025
                and grippers[arm_index] <= 0.04
                and np.linalg.norm(eef[arm_index] - block) <= 0.15
            )
            needed = self.pickup_confirmation_steps
        else:  # place -> next color; the final place stays active for task verification
            condition = (
                abs(lift) <= 0.02
                and grippers[arm_index] >= 0.03
                and np.linalg.norm(block[:2] - self.initial_positions[color_index, :2]) >= 0.003
                and np.linalg.norm(velocities[color_index]) <= 0.05
            )
            needed = self.place_confirmation_steps
        self.confirmation_count = self.confirmation_count + 1 if condition else 0
        if self.confirmation_count < needed or self.phase_index == len(self.prompts) - 1:
            return False
        old_index = self.phase_index
        self.phase_index += 1
        self.confirmation_count = 0
        self.events.append(
            {
                "control_step": control_step,
                "from_phase": old_index,
                "to_phase": self.phase_index,
                "from_prompt": self.prompts[old_index],
                "to_prompt": self.prompt,
                "evidence": {
                    "object_color": COLORS[color_index],
                    "object_lift_m": round(lift, 4),
                    "gripper_position_m": round(float(grippers[arm_index]), 4),
                    "object_xyz_m": block.round(4).tolist(),
                },
            }
        )
        return True
