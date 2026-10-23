"""Inference-only sampling controls for a loaded GR00T N1.7 policy.

The policy architecture and weights are unchanged. A larger flow integration
budget can reduce discretization error; averaging independent initial-noise
samples trades latency for lower stochastic variance in a unimodal task.
"""

from __future__ import annotations

from typing import Any


def configure_n17_sampling(policy: Any, *, flow_steps: int = 4, sample_average: int = 1) -> None:
    if flow_steps < 1 or sample_average < 1:
        raise ValueError("flow_steps and sample_average must be positive")
    policy.model.action_head.num_inference_timesteps = flow_steps
    if sample_average == 1:
        return

    import torch

    original_get_action = policy.model.get_action

    def averaged_get_action(*args: Any, **kwargs: Any) -> Any:
        first = original_get_action(*args, **kwargs)
        predictions = [first["action_pred"]]
        for _ in range(sample_average - 1):
            predictions.append(original_get_action(*args, **kwargs)["action_pred"])
        first["action_pred"] = torch.stack(predictions).mean(dim=0)
        return first

    policy.model.get_action = averaged_get_action
