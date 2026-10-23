import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from twingr00t_lab.seeding import configure_policy_seed, validate_seed


@pytest.mark.parametrize("cuda_available", [False, True])
def test_policy_rngs_repeat_without_changing_deterministic_kernel_flags(monkeypatch, cuda_available):
    calls = []
    torch_stub = SimpleNamespace(
        manual_seed=lambda seed: calls.append(("torch", seed)),
        cuda=SimpleNamespace(
            is_available=lambda: cuda_available,
            manual_seed_all=lambda seed: calls.append(("cuda", seed)),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    metadata = configure_policy_seed(42)
    first = (random.random(), np.random.random())
    configure_policy_seed(42)
    assert first == (random.random(), np.random.random())
    assert calls == ([("torch", 42), ("cuda", 42)] * 2 if cuda_available else [("torch", 42)] * 2)
    assert metadata["policy_seed"] == 42
    assert "does not guarantee" in metadata["seed_determinism_note"]


@pytest.mark.parametrize("seed", [-1, 2**32, 0.5, True, None])
def test_invalid_seeds_fail_before_importing_torch(seed):
    with pytest.raises(ValueError, match="integer between"):
        configure_policy_seed(seed)


@pytest.mark.parametrize("seed", [0, 42, 2**32 - 1])
def test_seed_boundaries(seed):
    assert validate_seed(seed) == seed
