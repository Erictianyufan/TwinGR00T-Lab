import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

from twingr00t_lab.conversion import (
    CAMERA_MAP,
    convert_dataset,
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


@pytest.mark.parametrize("offset", [0, 1, 2])
def test_conversion_aligns_future_actions_without_crossing_episodes(tmp_path, monkeypatch, offset) -> None:
    h5py = pytest.importorskip("h5py")
    pq = pytest.importorskip("pyarrow.parquet")
    from twingr00t_lab import conversion

    source = tmp_path / "source"
    (source / "data").mkdir(parents=True)
    for episode, length in enumerate((4, 3)):
        with h5py.File(source / "data" / f"demo_{episode}.hdf5", "w") as handle:
            policy = handle.create_group("observations/policy")
            for key, dim in (
                ("left_arm_pos", 6),
                ("left_gripper_pos", 1),
                ("right_arm_pos", 6),
                ("right_gripper_pos", 1),
            ):
                policy[key] = np.repeat((episode * 10 + np.arange(length))[:, None], dim, axis=1)
            policy["actions"] = np.repeat((1000 + episode * 100 + np.arange(length))[:, None], 16, axis=1)
            cameras = handle.create_group("observations/camera_obs")
            for key in CAMERA_MAP.values():
                cameras[key] = np.broadcast_to(
                    (episode * 10 + np.arange(length, dtype=np.uint8))[:, None, None, None], (length, 2, 2, 3)
                )

    # Run the real encoder's slicing/streaming with a byte sink instead of an
    # external ffmpeg dependency; this catches accidental untrimmed video tails.
    streams = []

    class Sink:
        def __init__(self):
            self.data = bytearray()

        def write(self, data):
            self.data.extend(data)

        def close(self):
            pass

    class Process:
        def __init__(self, command, **kwargs):
            self.stdin = Sink()
            self.stderr = BytesIO()
            streams.append(self.stdin)

        def wait(self):
            return 0

    monkeypatch.setattr(conversion.shutil, "which", lambda name: "/fake/ffmpeg")
    monkeypatch.setattr(conversion.subprocess, "Popen", Process)
    destination = tmp_path / "converted"
    manifest = convert_dataset(
        input_dir=source,
        output_dir=destination,
        fps=31.25,
        start_episode=0,
        limit=None,
        instruction_index=0,
        task_fallback="sort blocks",
        manifest_path=None,
        observation_action_offset=offset,
    )
    assert manifest["frames"] == 7 - 2 * offset
    assert manifest["observation_action_offset"] == offset
    assert manifest["discarded_tail_frames_per_episode"] == offset
    episodes = [json.loads(row) for row in (destination / "meta/episodes.jsonl").read_text().splitlines()]
    global_start = 0
    for episode, length in enumerate((4, 3)):
        kept = length - offset
        assert episodes[episode]["length"] == kept
        table = pq.read_table(destination / "data/chunk-000" / f"episode_{episode:06d}.parquet").to_pydict()
        np.testing.assert_array_equal(np.asarray(table["observation.state"])[:, 0], episode * 10 + np.arange(kept))
        np.testing.assert_array_equal(
            np.asarray(table["action"])[:, 0], 1000 + episode * 100 + np.arange(offset, length)
        )
        assert table["episode_index"] == [episode] * kept
        assert table["frame_index"] == list(range(kept))
        assert table["index"] == list(range(global_start, global_start + kept))
        np.testing.assert_allclose(table["timestamp"], np.arange(kept) / 31.25)
        assert table["next.done"] == [False] * (kept - 1) + [True]
        global_start += kept
        for stream in streams[episode * 3 : episode * 3 + 3]:
            video = np.frombuffer(stream.data, dtype=np.uint8).reshape(kept, 2, 2, 3)
            np.testing.assert_array_equal(video[:, 0, 0, 0], episode * 10 + np.arange(kept))


def test_negative_action_offset_is_rejected_before_creating_output(tmp_path) -> None:
    with pytest.raises(ValueError, match="must be non-negative"):
        convert_dataset(
            input_dir=tmp_path / "source",
            output_dir=tmp_path / "out",
            fps=31.25,
            start_episode=0,
            limit=None,
            instruction_index=0,
            task_fallback="sort blocks",
            manifest_path=None,
            observation_action_offset=-1,
        )
    assert not (tmp_path / "out").exists()


def test_offset_cannot_consume_entire_episode(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    source = tmp_path / "source"
    (source / "data").mkdir(parents=True)
    with h5py.File(source / "data/demo_0.hdf5", "w") as handle:
        policy = handle.create_group("observations/policy")
        for side in ("left", "right"):
            policy[f"{side}_arm_pos"] = np.zeros((1, 6), dtype=np.float32)
            policy[f"{side}_gripper_pos"] = np.zeros((1, 1), dtype=np.float32)
        policy["actions"] = np.zeros((1, 16), dtype=np.float32)
    with pytest.raises(ValueError, match="no paired frames"):
        convert_dataset(
            input_dir=source,
            output_dir=tmp_path / "out",
            fps=31.25,
            start_episode=0,
            limit=None,
            instruction_index=0,
            task_fallback="sort blocks",
            manifest_path=None,
            observation_action_offset=1,
        )
    assert not (tmp_path / "out").exists()
