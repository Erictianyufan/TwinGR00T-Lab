"""Dependency-light contracts shared by converters and policy clients."""

from dataclasses import dataclass

import numpy as np

ARM_DIM = 6
CANONICAL_ACTION_DIM = 14
ISAAC_ACTION_DIM = 16


@dataclass(frozen=True)
class CanonicalAlohaAction:
    """Logical dual-arm action with one scalar command per gripper."""

    left_arm: np.ndarray
    left_gripper: float
    right_arm: np.ndarray
    right_gripper: float

    def __post_init__(self) -> None:
        left = np.asarray(self.left_arm, dtype=np.float32)
        right = np.asarray(self.right_arm, dtype=np.float32)
        if left.shape != (ARM_DIM,) or right.shape != (ARM_DIM,):
            raise ValueError("Each Aloha arm must contain exactly six values")
        values = np.concatenate((left, [self.left_gripper], right, [self.right_gripper]))
        if not np.isfinite(values).all():
            raise ValueError("Action values must be finite")
        object.__setattr__(self, "left_arm", left)
        object.__setattr__(self, "right_arm", right)

    @classmethod
    def from_vector(cls, value: np.ndarray) -> "CanonicalAlohaAction":
        vector = np.asarray(value, dtype=np.float32)
        if vector.shape != (CANONICAL_ACTION_DIM,):
            raise ValueError(f"Expected shape ({CANONICAL_ACTION_DIM},), got {vector.shape}")
        return cls(vector[:6], float(vector[6]), vector[7:13], float(vector[13]))

    def to_vector(self) -> np.ndarray:
        return np.concatenate(
            (
                self.left_arm,
                np.asarray([self.left_gripper], dtype=np.float32),
                self.right_arm,
                np.asarray([self.right_gripper], dtype=np.float32),
            )
        )

    def to_isaac(self) -> np.ndarray:
        """Expand logical grippers to the two physical joints expected by Isaac."""
        return np.concatenate(
            (
                self.left_arm,
                np.repeat(np.float32(self.left_gripper), 2),
                self.right_arm,
                np.repeat(np.float32(self.right_gripper), 2),
            )
        )
