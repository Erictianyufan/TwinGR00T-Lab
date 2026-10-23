"""Verify that a recorded episode was executed from saved N1.7 action chunks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np

from twingr00t_lab.grasp_acceptance import GraspAcceptanceConfig, evaluate_grasp_sorting


def expand_clip_bounds(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values)
    if values.shape != (14,) or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain 14 finite joint limits")
    return np.concatenate((values[:6], np.repeat(values[6:7], 2), values[7:13], np.repeat(values[13:14], 2)))


def verify_video(video_path: Path, manifest: dict, steps: int, video_control_steps: np.ndarray | None) -> dict:
    strict = manifest.get("recording_schema_version", 1) >= 2
    with video_path.open("rb") as video:
        digest = hashlib.file_digest(video, "sha256").hexdigest()
    if strict and digest != manifest.get("video_sha256"):
        raise ValueError("Video SHA-256 does not match the recording manifest")
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "format=duration:stream=nb_read_frames,avg_frame_rate", "-of", "json", str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(probe.stdout)
    duration = float(payload["format"]["duration"])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("The MP4 has no valid positive duration")
    result = {"duration_s": duration, "sha256": digest, "complete": False, "frame_count": None}
    if not strict:
        return result
    record_every = manifest.get("record_every")
    if type(record_every) is not int or record_every < 1:
        raise ValueError("record_every must be a positive integer")
    step_dt, fps = float(manifest["step_dt"]), float(manifest["record_fps"])
    if not math.isfinite(step_dt) or step_dt <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Recording cadence must be finite and positive")
    if not math.isclose(fps, 1.0 / step_dt / record_every, rel_tol=1e-5):
        raise ValueError("Recording FPS disagrees with simulator cadence")
    expected_steps = np.arange(1, steps + 1, record_every)
    expected_frames = len(expected_steps)
    if video_control_steps is None or not np.array_equal(video_control_steps, expected_steps):
        raise ValueError("Video control-step indices do not match recorded simulator steps")
    if manifest.get("expected_video_frames") != expected_frames or manifest.get("video_frame_count") != expected_frames:
        raise ValueError("Manifest video frame count does not cover the executed episode")
    expected_duration = expected_frames / fps
    if not math.isclose(float(manifest["expected_video_duration_s"]), expected_duration, rel_tol=1e-5):
        raise ValueError("Manifest video duration disagrees with recording cadence")
    try:
        stream = payload["streams"][0]
        decoded_frames = int(stream["nb_read_frames"])
        actual_fps = float(Fraction(stream["avg_frame_rate"]))
    except (KeyError, IndexError, ValueError, ZeroDivisionError) as exc:
        raise ValueError("Could not decode video frame count and frame rate") from exc
    if decoded_frames != expected_frames:
        raise ValueError("Decoded video frame count is incomplete or mismatched")
    if not math.isclose(actual_fps, fps, rel_tol=1e-3):
        raise ValueError("Decoded video frame rate disagrees with the manifest")
    if abs(duration - expected_duration) > max(0.01, 1.0 / fps):
        raise ValueError("Decoded video duration does not cover the executed episode")
    return {**result, "complete": True, "frame_count": decoded_frames}


def verify_recording(manifest_path: Path, *, require_success: bool = True, allow_unreviewed: bool = False) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if require_success and manifest.get("status") != "success":
        raise ValueError(f"Simulator did not report task success: {manifest.get('status')}")
    if manifest.get("controller_source") != "GR00T_N1.7_model_inference":
        raise ValueError("Controller source is not GR00T N1.7 inference")
    if manifest.get("replay_used") is not False or manifest.get("recorded_actions_used_after_reset") is not False:
        raise ValueError("A recorded-action replay cannot be accepted as online inference")
    schema = manifest.get("recording_schema_version", 1)
    if type(schema) is not int or schema not in (1, 2):
        raise ValueError(f"Unsupported recording schema: {schema}")
    if require_success and schema < 2:
        raise ValueError("Formal acceptance requires recording_schema_version=2 with raw actions and video evidence")

    trace_path = manifest_path.with_suffix(".npz")
    video_path = manifest_path.with_suffix(".mp4")
    if not trace_path.is_file() or not video_path.is_file():
        raise FileNotFoundError("The action trace and MP4 must accompany the manifest")
    with np.load(trace_path) as trace:
        actions = trace["actions"]
        chunks = trace["policy_chunks"]
        initial = trace["initial_action"]
        raw_chunks = trace["raw_policy_chunks"] if "raw_policy_chunks" in trace else None
        clip_lower = trace["clip_lower"] if "clip_lower" in trace else None
        clip_upper = trace["clip_upper"] if "clip_upper" in trace else None
        video_control_steps = trace["video_control_steps"] if "video_control_steps" in trace else None
        positions = trace["block_positions"] if "block_positions" in trace else None
        block_names = trace["block_names"] if "block_names" in trace else None
        evidence_keys = (
            "initial_block_positions",
            "block_positions",
            "block_linear_velocities",
            "block_angular_velocities",
            "eef_positions",
            "measured_states",
            "sorting_predicate",
        )
        evidence = {key: trace[key] for key in evidence_keys if key in trace}
    recorded_acceptance = manifest.get("acceptance")
    if require_success and (recorded_acceptance is None or len(evidence) != len(evidence_keys)):
        raise ValueError("Formal acceptance requires measured grasp and stable-placement evidence")
    acceptance = None
    acceptance_report_matches = None
    if recorded_acceptance is not None:
        if len(evidence) != len(evidence_keys):
            raise ValueError("Incomplete measured grasp evidence")
        acceptance = evaluate_grasp_sorting(
            **evidence,
            step_dt=float(manifest["step_dt"]),
            config=GraspAcceptanceConfig(**recorded_acceptance["config"]),
        )
        acceptance_report_matches = acceptance == recorded_acceptance
        if not acceptance_report_matches and (schema >= 2 or require_success):
            raise ValueError("Acceptance report disagrees with measured evidence")
        expected_success = acceptance["success"] and manifest.get("stop_reason") not in (
            "unexpected_termination",
            "unexpected_truncation",
        )
        if (manifest["status"] == "success") != expected_success and (schema >= 2 or require_success):
            raise ValueError("Success status disagrees with grasp/placement evidence or episode termination")
        if require_success and not expected_success:
            raise ValueError("The episode did not meet the kinematic grasp/sort acceptance gate")
    video_review = manifest.get("video_review", "pending")
    if video_review not in ("pending", "passed", "failed"):
        raise ValueError(f"Unknown video review status: {video_review}")
    verified_status = manifest["status"]
    if verified_status == "success" and video_review != "passed":
        if require_success and (not allow_unreviewed or video_review == "failed"):
            raise ValueError("Formal acceptance requires video_review=passed after reviewing the matching recording")
        verified_status = (
            "kinematic_pass_pending_review" if video_review == "pending" else "kinematic_pass_review_failed"
        )
    if actions.ndim != 2 or actions.shape[1] != 16 or chunks.ndim != 3 or chunks.shape[2] != 16:
        raise ValueError("Expected 16-D executed actions and predicted chunks")
    if initial.shape != (16,) or len(actions) != manifest["steps_executed"]:
        raise ValueError("Initial or executed action count does not match the manifest")
    if len(actions) < 1 or any(not np.isfinite(value).all() for value in (actions, chunks, initial)):
        raise ValueError("Executed actions and model chunks must be nonempty and finite")
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
    raw_clip_verified = False
    if schema >= 2:
        if raw_chunks is None or clip_lower is None or clip_upper is None:
            raise ValueError("Raw model chunks and actual clip bounds are required")
        if raw_chunks.shape != chunks.shape or not np.isfinite(raw_chunks).all():
            raise ValueError("Raw model chunks must match clipped chunks and be finite")
        lower, upper = expand_clip_bounds(clip_lower, "clip_lower"), expand_clip_bounds(clip_upper, "clip_upper")
        if np.any(lower > upper):
            raise ValueError("Clip lower bounds exceed upper bounds")
        for call, raw_chunk in zip(calls, raw_chunks, strict=True):
            if hashlib.sha256(np.ascontiguousarray(raw_chunk).tobytes()).hexdigest() != call.get("raw_action_sha256"):
                raise ValueError("Raw model chunk does not match its recorded SHA-256")
        if not np.array_equal(np.clip(raw_chunks, lower, upper), chunks):
            raise ValueError("Clipped chunks are not the saved raw model outputs projected into the recorded bounds")
        raw_clip_verified = True

    settings = manifest["action_safety"]
    alpha = float(settings["low_pass_alpha"])
    limits = np.asarray(
        [settings["max_joint_step"]] * 6
        + [settings["max_gripper_step"]] * 2
        + [settings["max_joint_step"]] * 6
        + [settings["max_gripper_step"]] * 2,
        dtype=np.float32,
    )
    if not math.isfinite(alpha) or not 0 < alpha <= 1 or not np.isfinite(limits).all() or np.any(limits <= 0):
        raise ValueError("Action filtering parameters must be finite and positive, with alpha <= 1")
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
    if require_success and min(arm_motion.values()) < 0.1:
        raise ValueError(f"Both arms did not move meaningfully: {arm_motion}")
    video = verify_video(video_path, manifest, len(actions), video_control_steps)
    if schema < 2 and verified_status == "success":
        verified_status = "legacy_diagnostic_unverified"
    return {
        "status": verified_status,
        "recording_schema_version": schema,
        "controller_source": manifest["controller_source"],
        "steps_executed": len(actions),
        "policy_calls": len(calls),
        "both_arms_motion_rad": arm_motion,
        "action_trace_reconstructed": True,
        "raw_to_clipped_actions_verified": raw_clip_verified,
        "grasp_acceptance": acceptance,
        "acceptance_report_matches": acceptance_report_matches,
        "video_review": video_review,
        "direct_contact_sensor_verified": False,
        "video_duration_s": video["duration_s"],
        "video_sha256": video["sha256"],
        "video_frame_count": video["frame_count"],
        "video_identity_and_completeness_verified": video["complete"],
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
    parser.add_argument(
        "--allow-unreviewed",
        action="store_true",
        help="Diagnose a kinematic pass pending video review, not formal success",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            verify_recording(
                args.manifest, require_success=not args.allow_failed, allow_unreviewed=args.allow_unreviewed
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
