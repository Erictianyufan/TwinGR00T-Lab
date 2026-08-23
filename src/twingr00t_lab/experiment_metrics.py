"""Auditable episode metrics for the G1 loco-manipulation experiment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class G1EpisodeMetrics:
    """Accumulate one episode without relying on cinematic or replay state."""

    seed: int | None
    requested_steps: int
    completed_steps: int = 0
    terminal_step: int | None = None
    success_step: int | None = None
    truncated_step: int | None = None
    initial_robot_position: np.ndarray | None = None
    final_robot_position: np.ndarray | None = None
    initial_cargo_position: np.ndarray | None = None
    final_cargo_position: np.ndarray | None = None
    destination_position: np.ndarray | None = None
    max_cargo_lift_m: float = 0.0
    min_destination_l2_m: float = float("inf")
    inference_latency_ms: list[float] = field(default_factory=list)
    action_delta_l2: list[float] = field(default_factory=list)
    _previous_action: np.ndarray | None = field(default=None, repr=False)

    def update(
        self,
        *,
        step: int,
        action: np.ndarray,
        robot_position: np.ndarray,
        cargo_position: np.ndarray,
        destination_position: np.ndarray,
        inference_latency_ms: float,
        terminated: bool,
        truncated: bool,
        state_valid: bool = True,
    ) -> None:
        robot_position = np.asarray(robot_position, dtype=np.float64)
        cargo_position = np.asarray(cargo_position, dtype=np.float64)
        destination_position = np.asarray(destination_position, dtype=np.float64)
        action = np.asarray(action, dtype=np.float64)

        if self.initial_robot_position is None:
            self.initial_robot_position = robot_position.copy()
            self.initial_cargo_position = cargo_position.copy()
            self.destination_position = destination_position.copy()
        self.completed_steps = step + 1
        self.inference_latency_ms.append(float(inference_latency_ms))

        if self._previous_action is not None:
            self.action_delta_l2.append(float(np.linalg.norm(action - self._previous_action)))
        self._previous_action = action.copy()

        if state_valid:
            self.final_robot_position = robot_position.copy()
            self.final_cargo_position = cargo_position.copy()
            assert self.initial_cargo_position is not None
            lift = float(cargo_position[2] - self.initial_cargo_position[2])
            self.max_cargo_lift_m = max(self.max_cargo_lift_m, lift)
            distance = float(np.linalg.norm(cargo_position - destination_position))
            self.min_destination_l2_m = min(self.min_destination_l2_m, distance)

            delta = np.abs(cargo_position - destination_position)
            success = bool(delta[0] <= 0.26 and delta[1] <= 0.13 and delta[2] <= 0.15)
            if success and self.success_step is None:
                self.success_step = step
        if terminated and self.terminal_step is None:
            self.terminal_step = step
        if truncated and self.truncated_step is None:
            self.truncated_step = step

    def summary(self) -> dict[str, Any]:
        assert self.initial_robot_position is not None
        assert self.final_robot_position is not None
        assert self.initial_cargo_position is not None
        assert self.final_cargo_position is not None
        assert self.destination_position is not None
        latency = np.asarray(self.inference_latency_ms, dtype=np.float64)
        action_delta = np.asarray(self.action_delta_l2, dtype=np.float64)
        return {
            "schema_version": 1,
            "seed": self.seed,
            "requested_steps": self.requested_steps,
            "completed_steps": self.completed_steps,
            "terminal_step": self.terminal_step,
            "truncated_step": self.truncated_step,
            "success": self.success_step is not None,
            "success_step": self.success_step,
            "initial_robot_position_w": self.initial_robot_position.round(6).tolist(),
            "final_robot_position_w": self.final_robot_position.round(6).tolist(),
            "robot_displacement_m": round(
                float(np.linalg.norm(self.final_robot_position - self.initial_robot_position)), 6
            ),
            "initial_cargo_position_w": self.initial_cargo_position.round(6).tolist(),
            "final_cargo_position_w": self.final_cargo_position.round(6).tolist(),
            "destination_position_w": self.destination_position.round(6).tolist(),
            "max_cargo_lift_m": round(self.max_cargo_lift_m, 6),
            "min_destination_l2_m": round(self.min_destination_l2_m, 6),
            "mean_policy_latency_ms": round(float(latency.mean()), 3),
            "p95_policy_latency_ms": round(float(np.percentile(latency, 95)), 3),
            "mean_action_delta_l2": round(float(action_delta.mean()), 6) if action_delta.size else 0.0,
        }
