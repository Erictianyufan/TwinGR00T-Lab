"""Merge only independently accepted Isaac factory teachers for N1.7 training.

This creates a *training* dataset, never an inference result. Rejected public
replays remain in their source diagnostics and are omitted, not relabeled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

from twingr00t_lab.conversion import episode_number
from twingr00t_lab.grasp_acceptance import GraspAcceptanceConfig


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def accepted_rows(root: Path, source_label: str, *, require_all: bool) -> tuple[list[dict], list[dict]]:
    import h5py

    reports = sorted((root / "reports").glob("demo_*.json"), key=episode_number)
    if not reports:
        raise ValueError(f"No Isaac physical reports in {root}")
    accepted, rejected = [], []
    for report_path in reports:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        episode = episode_number(report_path)
        status = report.get("status")
        if status == "rejected_physics" and not require_all:
            rejected.append({"source": source_label, "episode": episode, "status": status})
            continue
        if status != "accepted_for_training" or report.get("training_eligible") is not True:
            raise ValueError(f"{source_label}/demo_{episode} has unresolved collection status: {status}")
        if (
            report.get("scene_style") != "factory"
            or report.get("control_fps") != 62.5
            or report.get("correct_legacy_jpeg_bgr") is not False
            or report.get("model_inference_result") is not False
            or report.get("replay_used") is not True
        ):
            raise ValueError(f"{source_label}/demo_{episode} violates factory teacher data contract")
        acceptance = report.get("acceptance", {})
        if acceptance.get("config") != asdict(GraspAcceptanceConfig()) or any(
            acceptance.get(key) is not True
            for key in (
                "success", "all_objects_grasped", "both_arms_grasp_evidence",
                "stable_placement", "sorting_stable_window",
            )
        ):
            raise ValueError(f"{source_label}/demo_{episode} did not pass unchanged physical acceptance")
        hdf_relative = Path(report["hdf5"])
        if hdf_relative.is_absolute() or ".." in hdf_relative.parts:
            raise ValueError(f"Unsafe HDF5 path in {report_path}")
        hdf = root / hdf_relative
        instruction = root / "instructions" / f"demo_{episode}.json"
        if not hdf.is_file() or not instruction.is_file():
            raise FileNotFoundError(f"Missing accepted HDF5/instruction for {source_label}/demo_{episode}")
        if sha256(hdf) != report["hdf5_sha256"] or sha256(instruction) != report["instruction_sha256"]:
            raise ValueError(f"Accepted HDF5/instruction changed after verification: {source_label}/demo_{episode}")
        with h5py.File(hdf, "r") as handle:
            # Eligibility lives in the signed report; the collector records
            # only physical success in the HDF5 attributes.
            if not bool(handle.attrs.get("success", False)):
                raise ValueError(f"Accepted HDF5 lacks physical success: {hdf}")
            policy = handle["observations/policy"]
            count = len(policy["actions"])
            if count < 17 or any(
                len(handle[f"observations/camera_obs/{camera}"]) != count
                for camera in ("head_camera_rgb", "left_camera_rgb", "right_camera_rgb")
            ):
                raise ValueError(f"Accepted HDF5 has incomplete observation/action cameras: {hdf}")
        accepted.append({
            "source": source_label,
            "source_episode": episode,
            "report_path": report_path,
            "hdf_path": hdf,
            "instruction_path": instruction,
            "hdf5_sha256": report["hdf5_sha256"],
            "instruction_sha256": report["instruction_sha256"],
            "frames": count,
        })
    return accepted, rejected


def merge(original: Path, public_pilot: Path, public_batch: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    accepted, rejected = [], []
    for label, path, required in (
        ("original_factory_isaac", original, True),
        ("public_isaac_pilot", public_pilot, True),
        ("public_isaac_batch", public_batch, False),
    ):
        rows, skipped = accepted_rows(path, label, require_all=required)
        accepted.extend(rows)
        rejected.extend(skipped)
    original_count = sum(row["source"] == "original_factory_isaac" for row in accepted)
    public_count = sum(row["source"].startswith("public_") for row in accepted)
    if original_count != 9 or public_count < 1:
        raise ValueError("Need exactly nine original teachers and at least one accepted public Isaac replay")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as raw:
        staging = Path(raw) / "merged"
        for folder in ("data", "instructions", "provenance"):
            (staging / folder).mkdir(parents=True)
        manifest_rows = []
        for new_episode, row in enumerate(accepted):
            hdf_target = staging / "data" / f"demo_{new_episode}.hdf5"
            instruction_target = staging / "instructions" / f"demo_{new_episode}.json"
            shutil.copy2(row["hdf_path"], hdf_target)
            shutil.copy2(row["instruction_path"], instruction_target)
            if sha256(hdf_target) != row["hdf5_sha256"] or sha256(instruction_target) != row["instruction_sha256"]:
                raise ValueError(f"Copied accepted episode changed: {row['source']} {row['source_episode']}")
            report_target = staging / "provenance" / f"{row['source']}_demo_{row['source_episode']}.json"
            shutil.copy2(row["report_path"], report_target)
            manifest_rows.append({
                "new_episode": new_episode,
                "source": row["source"],
                "source_episode": row["source_episode"],
                "frames": row["frames"],
                "hdf5_sha256": row["hdf5_sha256"],
                "instruction_sha256": row["instruction_sha256"],
                "source_report_sha256": sha256(row["report_path"]),
            })
        manifest = {
            "status": "merged_only_physically_accepted_factory_teacher_episodes_for_training",
            "controller_source": "teacher_replay_for_training_not_N1.7_inference",
            "fps": 62.5,
            "observation_action_offset": 1,
            "legacy_jpeg_bgr_correction": False,
            "episodes": manifest_rows,
            "rejected_public_episodes_not_used": rejected,
            "model_inference_success_claim": False,
        }
        (staging / "source_audit.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        staging.rename(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True, type=Path)
    parser.add_argument("--public-pilot", required=True, type=Path)
    parser.add_argument("--public-batch", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = merge(args.original, args.public_pilot, args.public_batch, args.output_dir)
    print(json.dumps({"status": result["status"], "accepted": len(result["episodes"]),
                      "rejected": len(result["rejected_public_episodes_not_used"])}, indent=2))


if __name__ == "__main__":
    main()
