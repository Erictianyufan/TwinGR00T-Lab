import numpy as np
import pytest

from twingr00t_lab.dataset import collapse_isaac_action


def test_collapse_isaac_action():
    physical = np.asarray([[0, 1, 2, 3, 4, 5, 6, 6, 7, 8, 9, 10, 11, 12, 13, 13]], dtype=np.float32)
    expected = np.arange(14, dtype=np.float32)[None, :]
    np.testing.assert_array_equal(collapse_isaac_action(physical), expected)


def test_collapse_rejects_disagreeing_grippers():
    physical = np.zeros((2, 16), dtype=np.float32)
    physical[0, 7] = 0.5
    with pytest.raises(ValueError, match="Left physical gripper"):
        collapse_isaac_action(physical)
