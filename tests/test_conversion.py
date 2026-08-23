import json
from pathlib import Path

from twingr00t_lab.conversion import (
    dataset_info,
    episode_number,
    modality_metadata,
    select_demo_paths,
    select_instruction,
)


def test_episode_number() -> None:
    assert episode_number(Path("demo_42.hdf5")) == 42


def test_select_demo_paths_uses_source_episode_boundaries(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for number in (2, 10, 80, 81, 90):
        (data_dir / f"demo_{number}.hdf5").touch()
    assert [episode_number(path) for path in select_demo_paths(tmp_path, start_episode=80, limit=2)] == [80, 81]


def test_instruction_selection_wraps(tmp_path) -> None:
    path = tmp_path / "demo_0.json"
    path.write_text(json.dumps({"seen": ["first", "second"]}), encoding="utf-8")
    assert select_instruction(path, index=3) == "second"


def test_metadata_contract() -> None:
    modality = modality_metadata()
    info = dataset_info(episodes=5, frames=100, height=480, width=640, fps=31.25)
    assert modality["state"]["right_gripper"] == {"start": 13, "end": 14}
    assert info["features"]["action"]["shape"] == [14]
    assert info["features"]["observation.images.head"]["info"]["video.fps"] == 31.25
