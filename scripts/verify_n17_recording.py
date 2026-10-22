"""Verify that a recorded episode was executed from saved N1.7 action chunks."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


def verify_recording(manifest_path: Path, *, require_success: bool = True) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if require_success and manifest.get("status") != "success":
        raise ValueError(f"Simulator did not report task success: {manifest.get('status')}")
    if manifest.get("controller_source") != "GR00T_N1.7_model_inference":
        raise ValueError("Controller source is not GR00T N1.7 inference")
    if manifest.get("replay_used") is not False or manifest.get("recorded_actions_used_after_reset") is not False:
        raise ValueError("A recorded-action replay cannot be accepted as online inference")

    trace_path = manifest_path.with_suffix(".npz")
    video_path = manifest_path.with_suffix(".mp4")
    if not trace_path.is_file() or not video_path.is_file():
        raise FileNotFoundError("The action trace and MP4 must accompany the manifest")
    with np.load(trace_path) as trace:
        actions = trace["actions"]
        chunks = trace["policy_chunks"]
        initial = trace["initial_action"]
        positions = trace["block_positions"] if "block_positions" in trace else None
        block_names = trace["block_names"] if "block_names" in trace else None
    if actions.ndim != 2 or actions.shape[1] != 16 or chunks.ndim != 3 or chunks.shape[2] != 16:
        raise ValueError("Expected 16-D executed actions and predicted chunks")
    if initial.shape != (16,) or len(actions) != manifest["steps_executed"]:
        raise ValueError("Initial or executed action count does not match the manifest")
    if positions is not None:
        if positions.shape != (len(actions), 3, 3) or block_names is None or block_names.shape != (3,):
            raise ValueError("Block-position trace must cover each action and all three blocks")
        if not np.isfinite(positions).all():
            raise ValueError("Block-position trace contains NaN or Inf")
        for index, name in enumerate(block_names):
            expected = manifest["final_block_positions_m"][str(name)]
            if not np.allclose(positions[-1, index], expected, atol=1e-4, rtol=0):
                raise ValueError(f"Final pose for {name} does not match the saved trace")
    calls = manifest["policy_calls"]
    if len(calls) != len(chunks) or len(calls) < 1:
        raise ValueError("Policy call count does not match saved model chunks")
    for call, chunk in zip(calls, chunks, strict=True):
        if int(call["executed_horizon"]) != len(chunk):
            raise ValueError("A policy chunk has the wrong horizon")
        if hashlib.sha256(np.ascontiguousarray(chunk).tobytes()).hexdigest() != call["action_sha256"]:
            raise ValueError("A saved policy chunk does not match its recorded SHA-256")

    settings = manifest["action_safety"]
    alpha = float(settings["low_pass_alpha"])
    limits = np.asarray(
        [settings["max_joint_step"]] * 6
        + [settings["max_gripper_step"]] * 2
        + [settings["max_joint_step"]] * 6
        + [settings["max_gripper_step"]] * 2,
        dtype=np.float32,
    )
    targets = chunks.reshape(-1, 16)[: len(actions)]
    if len(targets) != len(actions):
        raise ValueError("The model produced fewer actions than the simulator executed")
    previous = initial.astype(np.float32)
    for index, target in enumerate(targets):
        previous = previous + np.clip(alpha * (target - previous), -limits, limits)
        if not np.allclose(previous, actions[index], atol=1e-5, rtol=0):
            raise ValueError(f"Executed action {index} is not derived from the model chunk")

    arm_motion = {
        "left": float(np.max(np.ptp(actions[:, :6], axis=0))),
        "right": float(np.max(np.ptp(actions[:, 8:14], axis=0))),
    }
    if min(arm_motion.values()) < 0.1:
        raise ValueError(f"Both arms did not move meaningfully: {arm_motion}")
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(video_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    duration = float(json.loads(probe.stdout)["format"]["duration"])
    if duration <= 0:
        raise ValueError("The MP4 is empty")
    return {
        "status": manifest["status"],
        "controller_source": manifest["controller_source"],
        "steps_executed": len(actions),
        "policy_calls": len(calls),
        "both_arms_motion_rad": arm_motion,
        "action_trace_reconstructed": True,
        "video_duration_s": duration,
        "video_bytes": video_path.stat().st_size,
        "initial_state_split": manifest.get("initial_state_split", "unspecified"),
        "block_displacement_m": (
            {
                str(name): float(np.linalg.norm(positions[-1, index] - positions[0, index]))
                for index, name in enumerate(block_names)
            }
            if positions is not None
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--allow-failed", action="store_true", help="Audit an unsuccessful diagnostic episode")
    args = parser.parse_args()
    print(json.dumps(verify_recording(args.manifest, require_success=not args.allow_failed), indent=2))


if __name__ == "__main__":
    main()
