import json

import numpy as np
import pytest

from twingr00t_lab.action_safety import action_bounds_from_stats


def test_arm_margin_changes_only_arm_bounds(tmp_path):
    stats = tmp_path / "stats.json"
    stats.write_text(
        json.dumps({"action": {"min": list(range(14)), "max": list(range(1, 15))}}), encoding="utf-8"
    )
    default_lower, default_upper = action_bounds_from_stats(stats)
    wider_lower, wider_upper = action_bounds_from_stats(stats, arm_margin_rad=0.3)
    assert default_lower.shape == (16,)
    assert default_upper.shape == (16,)
    arm = np.array([*range(6), *range(8, 14)])
    grippers = [6, 7, 14, 15]
    np.testing.assert_allclose(wider_lower[arm], default_lower[arm] - 0.2, atol=1e-6)
    np.testing.assert_allclose(wider_upper[arm], default_upper[arm] + 0.2, atol=1e-6)
    np.testing.assert_array_equal(wider_lower[grippers], default_lower[grippers])
    np.testing.assert_array_equal(wider_upper[grippers], default_upper[grippers])


@pytest.mark.parametrize("margin", [0.0, 0.09, 0.51, float("nan"), float("inf")])
def test_arm_margin_rejects_unsafe_or_nonfinite_values(tmp_path, margin):
    with pytest.raises(ValueError, match="arm_margin_rad"):
        action_bounds_from_stats(tmp_path / "unused.json", arm_margin_rad=margin)
