"""CPU-only episode sampling selection; never rewrite source data or statistics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _digest(value: Any) -> str:
    def convert(item: Any) -> Any:
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"Unsupported audit value: {type(item).__name__}")

    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=convert)
    return hashlib.sha256(text.encode()).hexdigest()


def inspect_selection_sources(dataset_path: Path) -> dict:
    """Record content identity; JSON formatting changes do not change its meaning."""
    result = {}
    for filename in (
        "info.json",
        "episodes.jsonl",
        "modality.json",
        "tasks.jsonl",
        "stats.json",
        "relative_stats.json",
    ):
        raw = (dataset_path / "meta" / filename).read_bytes()
        text = raw.decode("utf-8")
        value = (
            [json.loads(line) for line in text.splitlines() if line.strip()]
            if filename.endswith("jsonl")
            else json.loads(text)
        )
        result[filename] = {"sha256": hashlib.sha256(raw).hexdigest(), "content_sha256": _digest(value)}
    return result


def selection_plan(dataset_path: Path, episode_id: int) -> dict:
    if isinstance(episode_id, bool) or not isinstance(episode_id, int) or episode_id < 0:
        raise ValueError("training-episode-id must be a nonnegative integer")
    root = dataset_path.resolve()
    sources = inspect_selection_sources(root)
    metadata = [
        json.loads(line)
        for line in (root / "meta/episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    matches = [entry for entry in metadata if entry.get("episode_index") == episode_id]
    if len(matches) != 1 or type(matches[0].get("episode_index")) is not int:
        raise ValueError("Requested episode must match exactly one metadata episode_index")
    return {
        "mode": "single_episode_overfit",
        "dataset_path": str(root),
        "training_episode_id": episode_id,
        "source_episode_count": len(metadata),
        "normalization_source": "unchanged_full_dataset_statistics",
        "source_content": sources,
        "generalization_evidence": False,
        "status": "planned_not_yet_constructed",
    }


def filter_training_shards(dataset: Any, plan: dict) -> dict:
    """Filter the real upstream shard table, keeping original loader indices."""
    root = Path(plan["dataset_path"])
    if Path(dataset.dataset_path).resolve() != root:
        raise ValueError("Episode selection received an unexpected dataset path")
    sources = inspect_selection_sources(root)
    for name, original in plan["source_content"].items():
        if sources[name]["content_sha256"] != original["content_sha256"]:
            raise ValueError(f"Source metadata/statistics changed before selection: {name}")
    loader = dataset.episode_loader
    before_metadata = _digest(loader.episodes_metadata)
    disk_metadata = [
        json.loads(line)
        for line in (root / "meta/episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if before_metadata != _digest(disk_metadata):
        raise ValueError("Loaded episode metadata does not match the audited dataset")
    indices = [
        index
        for index, metadata in enumerate(loader.episodes_metadata)
        if metadata.get("episode_index") == plan["training_episode_id"]
    ]
    if len(indices) != 1:
        raise ValueError("Requested episode must match exactly one loader index")
    selected_index = indices[0]
    expected_length = int(dataset.get_effective_episode_length(selected_index))
    if expected_length < 1:
        raise ValueError("Selected episode has no valid action-horizon starting points")
    stats_digest = _digest(dataset.get_dataset_statistics())
    selected_shards = []
    all_steps = []
    for shard in dataset.sharded_episodes:
        selected = []
        for index, steps in shard:
            if index != selected_index:
                continue
            steps = np.asarray(steps)
            if steps.ndim != 1 or not np.issubdtype(steps.dtype, np.integer) or not len(steps):
                raise ValueError("Selected shard timesteps must be a nonempty 1D integer array")
            selected.append((index, steps))
            all_steps.append(steps)
        if selected:
            selected_shards.append(selected)
    actual = np.concatenate(all_steps) if all_steps else np.empty(0, dtype=np.int64)
    if not np.array_equal(np.sort(actual), np.arange(expected_length)):
        raise ValueError("Selected shard coverage must contain every valid starting point exactly once")
    lengths = np.asarray([sum(len(steps) for _, steps in shard) for shard in selected_shards], dtype=np.int64)
    if not len(lengths) or np.any(lengths <= 0):
        raise ValueError("Episode selection produced empty shards")
    # No loader/statistics/class mutation: workers still serialize the upstream class.
    dataset.sharded_episodes = selected_shards
    dataset.shard_lengths = lengths
    if (
        _digest(loader.episodes_metadata) != before_metadata
        or _digest(dataset.get_dataset_statistics()) != stats_digest
    ):
        raise ValueError("Episode filtering changed loader metadata or normalization statistics")
    if inspect_selection_sources(root) != sources:
        raise ValueError("Source files changed during episode filtering")
    return {
        **plan,
        "status": "verified_sampling_subset",
        "loader_index": selected_index,
        "effective_start_count": expected_length,
        "start_index_range": [0, expected_length - 1],
        "each_start_exactly_once": True,
        "shard_lengths": lengths.tolist(),
        "dataset_statistics_content_sha256": stats_digest,
        "effective_source_content": sources,
        "source_metadata_and_statistics_unchanged": True,
        "dataset_class": f"{type(dataset).__module__}.{type(dataset).__qualname__}",
    }


def install_factory_selection(factory_module: Any, plan: dict, audit_path: Path) -> Any:
    """Wrap only the factory alias, returning original, pickleable dataset objects.

    The closure lives in the constructing process, never on a dataset or worker.
    Returning a restore callback also prevents the hook leaking across experiments.
    """
    original = factory_module.ShardedSingleStepDataset
    calls = 0

    def construct(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        if calls:
            raise RuntimeError("Single-episode launcher expected exactly one dataset construction")
        calls += 1
        dataset = original(*args, **kwargs)
        audit = filter_training_shards(dataset, plan)
        with audit_path.open("x", encoding="utf-8") as stream:
            json.dump(audit, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"episode_selection_audit": audit}, allow_nan=False), flush=True)
        return dataset

    factory_module.ShardedSingleStepDataset = construct

    def restore() -> None:
        factory_module.ShardedSingleStepDataset = original

    return restore
