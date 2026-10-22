from __future__ import annotations

import json

import h5py
import numpy as np

from twingr00t_lab.xpolicylab_audit import audit_dataset


def _write_episode(path, *, with_state: bool) -> None:
    with h5py.File(path, "w") as episode:
        episode.create_dataset("joint_action/vector", data=np.zeros((3, 14)))
        for camera in ("head_camera", "left_camera", "right_camera"):
            episode.create_dataset(f"observation/{camera}/rgb", data=np.zeros((3, 2), dtype=np.uint8))
        if with_state:
            episode.create_dataset("observation/joint_state/vector", data=np.ones((3, 14)))


def test_actions_do_not_count_as_measured_state(tmp_path) -> None:
    data = tmp_path / "data"
    instructions = tmp_path / "instructions"
    data.mkdir()
    instructions.mkdir()
    _write_episode(data / "episode0.hdf5", with_state=False)
    (instructions / "episode0.json").write_text(json.dumps({"seen": ["sort blocks"]}))

    result = audit_dataset(tmp_path)

    assert not result["data_contract_ready"]
    assert result["action_is_not_measured_state"]
    assert "missing measured joint state" in result["episode_details"][0]["problems"][0]


def test_measured_state_and_cameras_pass(tmp_path) -> None:
    data = tmp_path / "data"
    instructions = tmp_path / "instructions"
    data.mkdir()
    instructions.mkdir()
    _write_episode(data / "episode0.hdf5", with_state=True)
    (instructions / "episode0.json").write_text(json.dumps({"seen": ["sort blocks"]}))

    result = audit_dataset(tmp_path)

    assert result["data_contract_ready"]
    assert result["frames"] == 3
