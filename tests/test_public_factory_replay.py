"""Guard the geometry and cadence of proposed public-to-Isaac replay data."""

from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pytest

module = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/prepare_public_factory_replay.py"))
block_centers_from_head = module["block_centers_from_head"]
resample_measured_qpos = module["resample_measured_qpos"]


def test_calibrated_color_centers_are_red_green_blue() -> None:
    image = np.zeros((60, 60, 3), dtype=np.uint8)
    for column, channel in ((5, 0), (25, 1), (45, 2)):
        image[10:18, column : column + 8, channel] = 240
    intrinsic = np.array([[100, 0, 20], [0, 100, 20], [0, 0, 1]], dtype=np.float32)
    extrinsic = np.column_stack((np.eye(3), np.zeros(3))).astype(np.float32)
    centers, evidence = block_centers_from_head(image, intrinsic, extrinsic, center_z=1.0)
    np.testing.assert_allclose(centers[:, 0], [-0.115, 0.085, 0.285], atol=1e-6)
    np.testing.assert_allclose(centers[:, 1:], [[-0.065, 1.0]] * 3, atol=1e-6)
    assert [evidence[color]["mask_pixels"] for color in ("red", "green", "blue")] == [64] * 3


def test_missing_color_rejected() -> None:
    image = np.zeros((60, 60, 3), dtype=np.uint8)
    intrinsic = np.eye(3, dtype=np.float32)
    extrinsic = np.column_stack((np.eye(3), np.zeros(3))).astype(np.float32)
    with pytest.raises(ValueError, match="mask area"):
        block_centers_from_head(image, intrinsic, extrinsic, center_z=1.0)


def test_resampling_preserves_causal_16d_joint_contract() -> None:
    qpos = np.zeros((3, 14), dtype=np.float32)
    qpos[:, 0] = [0, 0.2, 0.4]
    qpos[:, 6] = 1.0
    qpos[:, 13] = 0.0
    actions = resample_measured_qpos(qpos)
    assert actions.shape == (4, 16)
    assert actions[0, 0] == pytest.approx(0.096)
    assert actions[-1, 0] == pytest.approx(0.384)
    np.testing.assert_allclose(actions[:, 6:8], 0.048)
    np.testing.assert_allclose(actions[:, 14:16], -0.010)


def test_bad_public_gripper_rejected() -> None:
    qpos = np.zeros((3, 14), dtype=np.float32)
    qpos[1, 6] = 2
    with pytest.raises(ValueError, match="normalized"):
        resample_measured_qpos(qpos)
