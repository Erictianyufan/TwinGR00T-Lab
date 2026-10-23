"""Explicit RNG seeding for paired diagnostics, not a determinism guarantee."""

from __future__ import annotations

import random
from typing import Any

import numpy as np

SEED_NOTE = (
    "Python/NumPy/Torch policy RNGs and the simulation seed are configured explicitly. "
    "This does not guarantee bitwise determinism across devices, versions, or kernels; "
    "identical seeds alone do not guarantee identical later RNG streams if call counts differ."
)


def validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an integer between 0 and 2**32-1")
    return seed


def configure_policy_seed(seed: int) -> dict[str, Any]:
    """Set all policy RNGs before constructing the model; preserve kernel settings."""
    seed = validate_seed(seed)
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return {"policy_seed": seed, "seed_determinism_note": SEED_NOTE}
