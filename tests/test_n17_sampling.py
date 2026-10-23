"""Inference-only sampling changes must preserve N1.7 model outputs."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from twingr00t_lab.n17_sampling import configure_n17_sampling


def test_flow_steps_update_only_action_head() -> None:
    policy = SimpleNamespace(model=SimpleNamespace(action_head=SimpleNamespace(num_inference_timesteps=4)))
    configure_n17_sampling(policy, flow_steps=8)
    assert policy.model.action_head.num_inference_timesteps == 8


@pytest.mark.parametrize("steps,samples", [(0, 1), (4, 0), (-1, 2)])
def test_invalid_sampling_rejected(steps: int, samples: int) -> None:
    policy = SimpleNamespace(model=SimpleNamespace(action_head=SimpleNamespace(num_inference_timesteps=4)))
    with pytest.raises(ValueError):
        configure_n17_sampling(policy, flow_steps=steps, sample_average=samples)


def test_sample_average_uses_independent_model_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeStack:
        def __init__(self, values: list[np.ndarray]) -> None:
            self.values = values

        def mean(self, *, dim: int) -> np.ndarray:
            assert dim == 0
            return np.mean(self.values, axis=0)

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(stack=lambda values: FakeStack(values)))
    calls = 0

    def predict() -> dict[str, np.ndarray]:
        nonlocal calls
        calls += 1
        return {"action_pred": np.asarray([calls], dtype=np.float32)}

    policy = SimpleNamespace(
        model=SimpleNamespace(action_head=SimpleNamespace(num_inference_timesteps=4), get_action=predict)
    )
    configure_n17_sampling(policy, flow_steps=8, sample_average=3)
    result = policy.model.get_action()
    assert calls == 3
    assert result["action_pred"].tolist() == [2.0]
