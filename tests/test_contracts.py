import numpy as np
import pytest

from twingr00t_lab.contracts import CanonicalAlohaAction


def test_canonical_roundtrip_and_isaac_expansion():
    vector = np.arange(14, dtype=np.float32)
    action = CanonicalAlohaAction.from_vector(vector)

    np.testing.assert_array_equal(action.to_vector(), vector)
    np.testing.assert_array_equal(
        action.to_isaac(),
        np.asarray([0, 1, 2, 3, 4, 5, 6, 6, 7, 8, 9, 10, 11, 12, 13, 13], dtype=np.float32),
    )


def test_contract_rejects_wrong_shape_and_nonfinite_values():
    with pytest.raises(ValueError):
        CanonicalAlohaAction.from_vector(np.zeros(13, dtype=np.float32))
    with pytest.raises(ValueError):
        CanonicalAlohaAction.from_vector(np.full(14, np.nan, dtype=np.float32))
