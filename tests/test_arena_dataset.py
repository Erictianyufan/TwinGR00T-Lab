from __future__ import annotations

import json

import h5py
import numpy as np

from twingr00t_lab.arena_dataset import REQUIRED_SIGNALS, audit_arena_dataset


def test_audit_arena_dataset_records_subtask_boundaries(tmp_path) -> None:
    path = tmp_path / "arena.hdf5"
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        data.attrs["env_args"] = json.dumps({"env_name": "g1", "sim_args": {"dt": 0.005}})
        demo = data.create_group("demo_0")
        demo.attrs["num_samples"] = 4
        demo.attrs["success"] = True
        demo.create_dataset("actions", data=np.zeros((4, 23), dtype=np.float32))
        datagen = demo.create_group("obs/datagen_info")
        datagen.create_group("object_pose").create_dataset("brown_box", data=np.zeros((4, 4, 4)))
        signals = datagen.create_group("subtask_term_signals")
        for name in REQUIRED_SIGNALS:
            signals.create_dataset(name, data=np.array([[False, False, True, True]]))

    result = audit_arena_dataset(path)
    assert result["episodes"] == 1
    assert result["successful_episodes"] == 1
    assert result["total_frames"] == 4
    assert result["episode_details"][0]["subtask_rise_frames"]["navigate_to_table"] == [2]
