"""CPU checks of real shard-table filtering without importing GR00T or Torch."""

import copy
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from twingr00t_lab.training_selection import (
    filter_training_shards,
    focus_window_plan,
    install_factory_selection,
    selection_plan,
    single_episode_focus_plan,
)


class FakeUpstreamDataset:
    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path
        # Noncontiguous IDs deliberately differ from loader positions.
        self.episode_loader = SimpleNamespace(
            episodes_metadata=[{"episode_index": 8, "length": 6}, {"episode_index": 0, "length": 5}]
        )
        self.sharded_episodes = [
            [(0, np.array([0, 1])), (1, np.array([3, 1]))],
            [(0, np.array([2, 3, 4]))],
            [(1, np.array([0, 2]))],
        ]
        self.shard_lengths = np.array([4, 3, 2])
        self.statistics = {"all_episode_mean": np.array([42.0])}

    def get_effective_episode_length(self, index):
        return self.episode_loader.episodes_metadata[index]["length"] - 1

    def get_dataset_statistics(self):
        return self.statistics


@pytest.fixture
def sources(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    payloads = {
        "info.json": {"total_episodes": 2},
        "modality.json": {},
        "stats.json": {"state": {"mean": [42]}},
        "relative_stats.json": {"left_arm": {"mean": [0]}},
    }
    for filename, payload in payloads.items():
        (meta / filename).write_text(json.dumps(payload))
    (meta / "episodes.jsonl").write_text(
        "\n".join(json.dumps(x) for x in FakeUpstreamDataset(tmp_path).episode_loader.episodes_metadata)
    )
    (meta / "tasks.jsonl").write_text('{"task_index": 0, "task": "complete all three objects"}\n')
    return tmp_path


def test_mapping_coverage_statistics_and_original_type_preserved(sources):
    dataset = FakeUpstreamDataset(sources)
    metadata = copy.deepcopy(dataset.episode_loader.episodes_metadata)
    stats = dataset.statistics
    audit = filter_training_shards(dataset, selection_plan(sources, 0))
    assert audit["loader_index"] == 1
    assert audit["effective_start_count"] == 4
    assert audit["shard_lengths"] == [2, 2]
    assert audit["each_start_exactly_once"] is True
    assert audit["generalization_evidence"] is False
    assert dataset.statistics is stats
    assert dataset.episode_loader.episodes_metadata == metadata
    restored = pickle.loads(pickle.dumps(dataset))
    assert type(restored) is FakeUpstreamDataset
    assert all(i == 1 for shard in restored.sharded_episodes for i, _ in shard)


@pytest.mark.parametrize("episode_id", [-1, True, 99])
def test_missing_or_invalid_episode_fails(sources, episode_id):
    with pytest.raises(ValueError):
        selection_plan(sources, episode_id)


@pytest.mark.parametrize("steps", [[], [0, 1, 1, 3], [0, 1, 2], [0, 1, 2, 4], [0.0, 1.0, 2.0, 3.0]])
def test_missing_duplicate_out_of_range_and_noninteger_steps_fail(sources, steps):
    dataset = FakeUpstreamDataset(sources)
    dataset.sharded_episodes = [[(1, np.asarray(steps))]]
    with pytest.raises(ValueError, match="timesteps|coverage"):
        filter_training_shards(dataset, selection_plan(sources, 0))


@pytest.mark.parametrize("filename", ["stats.json", "relative_stats.json", "episodes.jsonl", "modality.json"])
def test_source_changes_before_construction_fail(sources, filename):
    plan = selection_plan(sources, 0)
    (sources / "meta" / filename).write_text('{"changed": true}\n')
    with pytest.raises(ValueError, match="changed before selection"):
        filter_training_shards(FakeUpstreamDataset(sources), plan)


def test_duplicate_metadata_id_fails(sources):
    with (sources / "meta/episodes.jsonl").open("a") as stream:
        stream.write('\n{"episode_index": 0, "length": 5}\n')
    with pytest.raises(ValueError, match="exactly one"):
        selection_plan(sources, 0)


def test_effective_empty_episode_fails(sources):
    dataset = FakeUpstreamDataset(sources)
    dataset.get_effective_episode_length = lambda _: 0
    with pytest.raises(ValueError, match="no valid"):
        filter_training_shards(dataset, selection_plan(sources, 0))


def test_loader_metadata_or_path_mismatch_fails(sources):
    plan = selection_plan(sources, 0)
    dataset = FakeUpstreamDataset(sources)
    dataset.episode_loader.episodes_metadata.reverse()
    with pytest.raises(ValueError, match="metadata does not match"):
        filter_training_shards(dataset, plan)
    dataset = FakeUpstreamDataset(sources)
    dataset.dataset_path = sources / "different-dataset"
    with pytest.raises(ValueError, match="unexpected dataset path"):
        filter_training_shards(dataset, plan)


def test_upstream_reserialization_does_not_change_statistics_semantics(sources):
    plan = selection_plan(sources, 0)
    path = sources / "meta/relative_stats.json"
    path.write_text(json.dumps(json.loads(path.read_text()), indent=2) + "\n")
    audit = filter_training_shards(FakeUpstreamDataset(sources), plan)
    before = audit["source_content"]["relative_stats.json"]
    after = audit["effective_source_content"]["relative_stats.json"]
    assert before["sha256"] != after["sha256"]
    assert before["content_sha256"] == after["content_sha256"]


def test_factory_alias_returns_original_pickleable_class_and_restores(sources):
    factory = SimpleNamespace(ShardedSingleStepDataset=FakeUpstreamDataset)
    audit_path = sources / "audit.json"
    restore = install_factory_selection(factory, selection_plan(sources, 0), audit_path)
    dataset = factory.ShardedSingleStepDataset(dataset_path=sources)
    assert type(pickle.loads(pickle.dumps(dataset))) is FakeUpstreamDataset
    assert json.loads(audit_path.read_text())["status"] == "verified_sampling_subset"
    with pytest.raises(RuntimeError, match="exactly one dataset"):
        factory.ShardedSingleStepDataset(dataset_path=sources)
    restore()
    assert factory.ShardedSingleStepDataset is FakeUpstreamDataset


def test_factory_refuses_overwriting_audit(sources):
    audit_path = sources / "audit.json"
    audit_path.write_text("existing")
    factory = SimpleNamespace(ShardedSingleStepDataset=FakeUpstreamDataset)
    restore = install_factory_selection(factory, selection_plan(sources, 0), audit_path)
    with pytest.raises(FileExistsError):
        factory.ShardedSingleStepDataset(sources)
    restore()
    assert audit_path.read_text() == "existing"


def test_focus_window_retains_all_starts_and_upweights_only_requested_window(sources):
    dataset = FakeUpstreamDataset(sources)
    before_stats = dataset.statistics
    factory = SimpleNamespace(ShardedSingleStepDataset=lambda dataset_path: dataset)
    audit_path = sources / "focus_audit.json"
    restore = install_factory_selection(factory, focus_window_plan(sources, 1, 3, 3), audit_path)
    assert factory.ShardedSingleStepDataset(sources) is dataset
    restore()
    audit = json.loads(audit_path.read_text())
    assert audit["status"] == "verified_phase_focus"
    assert audit["original_start_count"] == 9
    assert audit["extra_start_count"] == 12
    assert audit["effective_start_count"] == 21
    assert dataset.statistics is before_stats
    assert np.array_equal(dataset.shard_lengths, np.array([10, 6, 5]))
    counts = {i: {} for i in (0, 1)}
    for shard in dataset.sharded_episodes:
        for episode, steps in shard:
            for step in steps:
                counts[episode][int(step)] = counts[episode].get(int(step), 0) + 1
    assert counts == {0: {0: 1, 1: 4, 2: 4, 3: 1, 4: 1}, 1: {0: 1, 1: 4, 2: 4, 3: 1}}


@pytest.mark.parametrize("start,end,copies", [(0, 0, 3), (2, 1, 3), (0, 3, 0), (0, 3, 9), (True, 3, 2)])
def test_focus_window_rejects_invalid_bounds(sources, start, end, copies):
    with pytest.raises(ValueError):
        focus_window_plan(sources, start, end, copies)


def test_focus_window_rejects_incomplete_source_coverage(sources):
    dataset = FakeUpstreamDataset(sources)
    dataset.sharded_episodes[0][0] = (0, np.array([0, 0]))
    factory = SimpleNamespace(ShardedSingleStepDataset=lambda dataset_path: dataset)
    restore = install_factory_selection(factory, focus_window_plan(sources, 1, 3, 2), sources / "focus_audit.json")
    with pytest.raises(ValueError, match="coverage"):
        factory.ShardedSingleStepDataset(sources)
    restore()


def test_single_episode_phase_focus_preserves_all_selected_starts_and_stats(sources):
    dataset = FakeUpstreamDataset(sources)
    before_stats = dataset.statistics
    plan = single_episode_focus_plan(sources, 0, 1, 3, 3)
    factory = SimpleNamespace(ShardedSingleStepDataset=lambda dataset_path: dataset)
    restore = install_factory_selection(factory, plan, sources / "single_focus_audit.json")
    assert factory.ShardedSingleStepDataset(sources) is dataset
    restore()
    audit = json.loads((sources / "single_focus_audit.json").read_text())
    assert audit["status"] == "verified_single_episode_phase_focus"
    assert audit["training_episode_id"] == 0
    assert audit["original_start_count"] == 4
    assert audit["extra_start_count"] == 6
    assert audit["effective_start_count"] == 10
    assert audit["original_each_start_exactly_once"] is True
    assert dataset.statistics is before_stats
    counts = {}
    for shard in dataset.sharded_episodes:
        for episode, steps in shard:
            assert episode == 1  # loader index 1 is metadata episode_index 0
            for step in steps:
                counts[int(step)] = counts.get(int(step), 0) + 1
    assert counts == {0: 1, 1: 4, 2: 4, 3: 1}
