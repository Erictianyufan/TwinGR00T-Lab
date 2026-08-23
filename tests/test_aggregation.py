import pytest

from twingr00t_lab.aggregation import aggregate_g1_manifests, wilson_interval


def _row(seed: int, success: bool) -> dict:
    return {
        "seed": seed,
        "success": success,
        "protocol_valid": True,
        "condition": "unassisted",
        "completed_steps": 100,
        "max_cargo_lift_m": 0.2,
        "min_destination_l2_m": 0.3,
        "robot_displacement_m": 1.0,
        "mean_policy_latency_ms": 10.0,
        "p95_policy_latency_ms": 20.0,
        "mean_action_delta_l2": 0.1,
    }


def test_aggregate_g1_manifests_passes_four_of_five_gate() -> None:
    result = aggregate_g1_manifests([_row(seed, seed != 14) for seed in range(10, 15)])
    assert result["successes"] == 4
    assert result["success_rate"] == 0.8
    assert result["gate_4_of_5_pass"] is True


def test_aggregate_rejects_invalid_or_duplicate_runs() -> None:
    invalid = _row(10, True)
    invalid["protocol_valid"] = False
    with pytest.raises(ValueError, match="protocol-invalid"):
        aggregate_g1_manifests([invalid])
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_g1_manifests([_row(10, True), _row(10, False)])


def test_wilson_interval_is_bounded() -> None:
    low, high = wilson_interval(4, 5)
    assert 0.0 < low < 0.8 < high <= 1.0
