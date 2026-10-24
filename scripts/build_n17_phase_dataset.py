"""Relabel one accepted teacher episode into six language-conditioned phases.

Joint targets, proprioception and all RGB videos remain bit-for-bit unchanged.
The output is a *training* dataset, not a policy result. A separate online
observation-driven router is required to use these prompts at inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from itertools import pairwise
from pathlib import Path

import numpy as np

from twingr00t_lab.phase_router import DEFAULT_ARM_ORDER, phase_prompts


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def phase_segments(
    block_z: np.ndarray,
    gripper_actions: np.ndarray,
    *,
    arm_order: tuple[str, str, str] = DEFAULT_ARM_ORDER,
    frame_count: int,
) -> list[tuple[int, int]]:
    """Locate observed lift and release boundaries in one accepted teacher."""
    z = np.asarray(block_z, dtype=np.float32)
    grippers = np.asarray(gripper_actions, dtype=np.float32)
    if z.ndim != 2 or z.shape[1] != 3 or grippers.shape != (len(z), 2):
        raise ValueError("Expected synchronized (T,3) object heights and (T,2) gripper targets")
    if frame_count != len(z) - 1 or not np.isfinite(z).all() or not np.isfinite(grippers).all():
        raise ValueError("Invalid one-step action alignment or non-finite teacher data")
    phase_prompts(arm_order)
    first_lifts, releases = [], []
    for color_index, arm in enumerate(arm_order):
        lifted = np.flatnonzero(z[:, color_index] - z[0, color_index] >= 0.025)
        if len(lifted) < 5:
            raise ValueError(f"Object {color_index} has no sustained teacher lift")
        first_lift, last_lift = int(lifted[0]), int(lifted[-1])
        arm_index = 0 if arm == "left" else 1
        grip = grippers[:, arm_index]
        openings = np.flatnonzero((grip[:-1] <= 0.03) & (grip[1:] > 0.03)) + 1
        after_lift = openings[openings > last_lift]
        if not len(after_lift):
            raise ValueError(f"Object {color_index} has no teacher gripper release")
        first_lifts.append(first_lift)
        releases.append(int(after_lift[0]))
    boundaries = [0, first_lifts[0], releases[0], first_lifts[1], releases[1], first_lifts[2], frame_count]
    if any(start >= end for start, end in pairwise(boundaries)) or releases[2] >= frame_count:
        raise ValueError(f"Teacher pick/place milestones are not strictly ordered: {boundaries}")
    return list(pairwise(boundaries))


def build(args: argparse.Namespace) -> dict:
    import h5py
    import pyarrow as pa
    import pyarrow.parquet as pq

    source, output, demo, report = (path.resolve() for path in (args.source, args.output, args.demo, args.report))
    if output.exists() or not all(path.is_file() for path in (demo, report)):
        raise ValueError("Output must be new and accepted teacher HDF5/report must exist")
    audit_path = demo.parent.parent / "source_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    match = [entry for entry in audit["episodes"] if entry.get("new_episode") == args.episode_id]
    if len(match) != 1 or match[0]["hdf5_sha256"] != sha256(demo):
        raise ValueError("Teacher HDF5 does not match the accepted merged-source audit")
    if match[0]["source_report_sha256"] != sha256(report):
        raise ValueError("Physical acceptance report does not match the source audit")
    acceptance = json.loads(report.read_text(encoding="utf-8"))["acceptance"]
    if not all(
        acceptance.get(key) is True
        for key in ("success", "all_objects_grasped", "both_arms_grasp_evidence", "stable_placement")
    ):
        raise ValueError("Only a physically accepted complete teacher may be phase-relabelled")

    parquet_name = f"data/chunk-{args.episode_id // 1000:03d}/episode_{args.episode_id:06d}.parquet"
    table = pq.read_table(source / parquet_name)
    with h5py.File(demo) as handle:
        z = np.stack(
            [np.asarray(handle[f"states/rigid_object/block{index}/root_pose"][:, 2]) for index in (1, 2, 3)],
            axis=1,
        )
        actions = np.asarray(handle["observations/policy/actions"][:], dtype=np.float32)
    segments = phase_segments(z, actions[:, [6, 14]], frame_count=table.num_rows)
    frame_indices = table["frame_index"].to_numpy()
    if not np.array_equal(frame_indices, np.arange(table.num_rows)):
        raise ValueError("Parquet frame indices are not contiguous from zero")
    if set(table["episode_index"].to_pylist()) != {args.episode_id}:
        raise ValueError("Parquet episode index does not match the requested teacher")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as raw:
        temporary = Path(raw) / "dataset"
        for item in source.rglob("*"):
            target = temporary / item.relative_to(source)
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif item.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                if item.suffix == ".mp4":
                    os.link(item, target)
                else:
                    shutil.copy2(item, target)
        info_path = temporary / "meta/info.json"
        tasks_path = temporary / "meta/tasks.jsonl"
        episodes_path = temporary / "meta/episodes.jsonl"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        tasks = [json.loads(line) for line in tasks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        prompts = phase_prompts(tuple(args.arm_order))
        first_task = max(task["task_index"] for task in tasks) + 1
        tasks.extend({"task_index": first_task + index, "task": text} for index, text in enumerate(prompts))
        tasks_path.write_text("".join(json.dumps(task) + "\n" for task in tasks), encoding="utf-8")
        info["total_tasks"] = len(tasks)
        info_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
        episodes = [json.loads(line) for line in episodes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        episode_matches = [entry for entry in episodes if entry["episode_index"] == args.episode_id]
        if len(episode_matches) != 1 or episode_matches[0]["length"] != table.num_rows:
            raise ValueError("Episode metadata disagrees with the source parquet")
        episode_matches[0]["tasks"] = list(prompts)
        episodes_path.write_text("".join(json.dumps(entry) + "\n" for entry in episodes), encoding="utf-8")

        labels = np.empty(table.num_rows, dtype=np.int64)
        for index, (start, end) in enumerate(segments):
            labels[start:end] = first_task + index
        for name in ("task_index", "annotation.human.task_description"):
            index = table.schema.get_field_index(name)
            if index < 0:
                raise ValueError(f"Source parquet lacks language column {name}")
            table = table.set_column(index, name, pa.array(labels))
        replacement = temporary / parquet_name
        pq.write_table(table, replacement, compression="zstd")
        reread = pq.read_table(replacement)
        for name in ("action", "observation.state", "timestamp", "frame_index", "episode_index", "index"):
            if not reread[name].equals(pq.read_table(source / parquet_name, columns=[name])[name]):
                raise ValueError(f"Phase relabelling changed non-language feature {name}")
        if not np.array_equal(reread["task_index"].to_numpy(), labels) or not np.array_equal(
            reread["annotation.human.task_description"].to_numpy(), labels
        ):
            raise ValueError("Phase prompt labels did not survive parquet serialization")
        result = {
            "status": "phase_labelled_teacher_training_data_not_model_inference",
            "source_dataset": str(source),
            "source_demo_sha256": sha256(demo),
            "source_report_sha256": sha256(report),
            "episode_id": args.episode_id,
            "arm_order": list(args.arm_order),
            "prompt_ids": list(range(first_task, first_task + 6)),
            "prompts": list(prompts),
            "segments_start_inclusive_end_exclusive": [list(pair) for pair in segments],
            "observation_action_offset": 1,
            "actions_state_timestamp_and_video_unchanged": True,
            "training_eligible": True,
            "generalization_evidence": False,
            "statistics": "copied temporarily; recompute against output parquet before training",
        }
        (temporary / "phase_dataset_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        temporary.rename(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Audited original LeRobot v2 dataset")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--demo", required=True, type=Path, help="Accepted HDF5 for the selected episode")
    parser.add_argument("--report", required=True, type=Path, help="Matching physical replay report")
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--arm-order", nargs=3, choices=("left", "right"), default=DEFAULT_ARM_ORDER)
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
