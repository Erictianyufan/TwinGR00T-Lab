"""Aggregate frozen episode manifests without importing simulator dependencies."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from statistics import fmean
from typing import Any


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("total must be positive")
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    center = (proportion + z**2 / (2.0 * total)) / denominator
    margin = z * math.sqrt(proportion * (1.0 - proportion) / total + z**2 / (4.0 * total**2))
    margin /= denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def aggregate_g1_manifests(manifests: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(manifests)
    if not rows:
        raise ValueError("at least one manifest is required")
    invalid = [row.get("seed") for row in rows if not row.get("protocol_valid", False)]
    if invalid:
        raise ValueError(f"protocol-invalid manifests are not reportable: seeds={invalid}")
    seeds = [int(row["seed"]) for row in rows]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"duplicate seeds: {seeds}")

    successes = sum(bool(row["success"]) for row in rows)
    low, high = wilson_interval(successes, len(rows))
    metric_names = (
        "completed_steps",
        "max_cargo_lift_m",
        "min_destination_l2_m",
        "robot_displacement_m",
        "mean_policy_latency_ms",
        "p95_policy_latency_ms",
        "mean_action_delta_l2",
    )
    means = {f"mean_{name}": round(fmean(float(row[name]) for row in rows), 6) for name in metric_names}
    return {
        "schema_version": 1,
        "condition": rows[0].get("condition"),
        "seeds": seeds,
        "episodes": len(rows),
        "successes": successes,
        "success_rate": round(successes / len(rows), 6),
        "success_rate_wilson95": [round(low, 6), round(high, 6)],
        "gate_4_of_5_pass": len(rows) == 5 and successes >= 4,
        **means,
    }
