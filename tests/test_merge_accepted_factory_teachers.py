"""Only independently accepted Isaac episodes may enter the N1.7 training set."""

from __future__ import annotations

import json
import runpy
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import pytest

from twingr00t_lab.grasp_acceptance import GraspAcceptanceConfig

module = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/merge_accepted_factory_teachers.py"))
merge = module["merge"]
sha256 = module["sha256"]


def make_episode(root: Path, episode: int, *, accepted: bool) -> None:
    (root / "reports").mkdir(parents=True, exist_ok=True)
    report = {
        "status": "accepted_for_training" if accepted else "rejected_physics",
        "training_eligible": accepted,
    }
    if accepted:
        hdf = root / "data" / f"demo_{episode}.hdf5"
        instruction = root / "instructions" / f"demo_{episode}.json"
        hdf.parent.mkdir(parents=True, exist_ok=True)
        instruction.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(hdf, "w") as handle:
            handle.attrs["success"] = True
            handle.create_dataset("observations/policy/actions", data=np.zeros((17, 16), dtype=np.float32))
            for camera in ("head_camera_rgb", "left_camera_rgb", "right_camera_rgb"):
                handle.create_dataset(f"observations/camera_obs/{camera}", data=np.zeros((17, 1), dtype=np.uint8))
        instruction.write_text('{"seen": ["sort blocks"]}\n', encoding="utf-8")
        report.update({
            "scene_style": "factory",
            "control_fps": 62.5,
            "correct_legacy_jpeg_bgr": False,
            "model_inference_result": False,
            "replay_used": True,
            "acceptance": {
                "config": asdict(GraspAcceptanceConfig()),
                "success": True,
                "all_objects_grasped": True,
                "both_arms_grasp_evidence": True,
                "stable_placement": True,
                "sorting_stable_window": True,
            },
            "hdf5": f"data/demo_{episode}.hdf5",
            "hdf5_sha256": sha256(hdf),
            "instruction_sha256": sha256(instruction),
        })
    (root / "reports" / f"demo_{episode}.json").write_text(json.dumps(report), encoding="utf-8")


def sources(tmp_path: Path) -> tuple[Path, Path, Path]:
    original, pilot, batch = (tmp_path / name for name in ("original", "pilot", "batch"))
    for episode in range(9):
        make_episode(original, episode, accepted=True)
    make_episode(pilot, 0, accepted=True)
    make_episode(batch, 1, accepted=True)
    make_episode(batch, 2, accepted=False)
    return original, pilot, batch


def test_merge_retains_only_accepted_isaac_replays(tmp_path: Path) -> None:
    output = tmp_path / "combined"
    result = merge(*sources(tmp_path), output)
    assert len(result["episodes"]) == 11
    assert result["rejected_public_episodes_not_used"] == [
        {"source": "public_isaac_batch", "episode": 2, "status": "rejected_physics"}
    ]
    assert len(list((output / "data").glob("demo_*.hdf5"))) == 11
    assert not result["model_inference_success_claim"]
    assert result["controller_source"] == "teacher_replay_for_training_not_N1.7_inference"


def test_merge_rejects_tampered_instruction(tmp_path: Path) -> None:
    original, pilot, batch = sources(tmp_path)
    (pilot / "instructions/demo_0.json").write_text('{"seen": ["different"]}', encoding="utf-8")
    output = tmp_path / "combined"
    with pytest.raises(ValueError, match="changed after verification"):
        merge(original, pilot, batch, output)
    assert not output.exists()


def test_merge_rejects_model_inference_mislabeled_as_teacher(tmp_path: Path) -> None:
    original, pilot, batch = sources(tmp_path)
    path = batch / "reports/demo_1.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["model_inference_result"] = True
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="teacher data contract"):
        merge(original, pilot, batch, tmp_path / "combined")
