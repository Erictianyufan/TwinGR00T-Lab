"""Record an ALOHA episode driven by live GR00T N1.7 model predictions.

The source demonstration supplies only the initial scene/robot state. Its saved
actions are never used after reset. The policy server must be launched separately
with ``--model-path`` pointing at a post-trained N1.7 checkpoint.
"""

from __future__ import annotations

import json
import subprocess
import time
import traceback
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from isaaclab.app import AppLauncher
from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.examples.example_environments.cli import (
    add_example_environments_cli_args,
    get_arena_builder_from_cli,
)
from replay_live_multiview import (
    CAMERA_KEYS,
    compose_frame,
    ffmpeg_command,
    instruction_for,
    read_demo,
    task_key_for,
    teleport,
)

from twingr00t_lab.dataset import collapse_isaac_action
from twingr00t_lab.grasp_acceptance import evaluate_grasp_sorting
from twingr00t_lab.model_identity import checkpoint_identity, validate_model_identity
from twingr00t_lab.phase_router import ThreeBlockPhaseRouter
from twingr00t_lab.policy_bridge import build_groot_observation, groot_action_to_isaac, to_numpy
from twingr00t_lab.policy_client import PolicyTransportClient
from twingr00t_lab.policy_input_evidence import (
    SCENE_STYLES,
    save_policy_input_evidence,
    scene_evidence,
    scene_video_title,
)
from twingr00t_lab.recording_evidence import action_sha256, video_evidence
from twingr00t_lab.seeding import SEED_NOTE, validate_seed


def spawn_factory_background(scene_style: str = "factory") -> None:
    """Keep shared HDR lighting; optionally add the factory's visual dressing."""
    import isaaclab.sim as sim_utils

    scene_evidence(scene_style)
    hdr = Path(__file__).resolve().parents[1] / "vendor/RoboTwin/assets/texture/base0.exr"
    if not hdr.is_file():
        raise FileNotFoundError(f"Training-scene HDR lighting asset is missing: {hdr}")
    light = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75), texture_file=str(hdr))
    light.func("/World/Light", light)
    if scene_style == "no-factory-dressing":
        return

    def panel(
        path: str, size: tuple[float, float, float], pos: tuple[float, float, float], color: tuple[float, float, float]
    ) -> None:
        cfg = sim_utils.CuboidCfg(
            size=size,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, metallic=0.05, roughness=0.75),
        )
        cfg.func(path, cfg, translation=pos)

    panel("/World/Factory/Floor", (6.0, 6.0, 0.08), (0.0, 0.0, -0.09), (0.28, 0.31, 0.33))
    # The policy's front/head cameras look toward +Y. Put the industrial wall
    # behind the workcell from their viewpoint rather than behind the cameras.
    panel("/World/Factory/BackWall", (6.0, 0.10, 3.0), (0.0, 2.3, 1.4), (0.60, 0.63, 0.65))
    panel("/World/Factory/CameraWall", (6.0, 0.10, 3.0), (0.0, -2.3, 1.4), (0.56, 0.60, 0.63))
    panel("/World/Factory/LeftWall", (0.10, 4.6, 3.0), (-2.8, 0.0, 1.4), (0.51, 0.55, 0.57))
    panel("/World/Factory/RightWall", (0.10, 4.6, 3.0), (2.8, 0.0, 1.4), (0.51, 0.55, 0.57))
    for index, x in enumerate((-1.9, 1.9)):
        panel(f"/World/Factory/SafetyStripe{index}", (0.055, 3.2, 0.005), (x, 0.0, -0.045), (0.95, 0.64, 0.10))
    for index, x in enumerate((-1.6, 1.6)):
        panel(f"/World/Factory/StorageRack{index}", (0.75, 0.35, 1.35), (x, 1.8, 0.65), (0.18, 0.27, 0.35))
        panel(f"/World/Factory/RearRack{index}", (0.75, 0.35, 1.35), (x, -1.8, 0.65), (0.18, 0.27, 0.35))


def success_from_term(env: gym.Env, success_term: object) -> bool:
    result = success_term.func(env, **success_term.params)
    if isinstance(result, torch.Tensor):
        return bool(result.any().item())
    return bool(result)


def measured_state(policy: dict) -> np.ndarray:
    return np.concatenate(
        [
            to_numpy(policy[key]).reshape(-1)
            for key in ("left_arm_pos", "left_gripper_pos", "right_arm_pos", "right_gripper_pos")
        ]
    ).astype(np.float32)


def action_bounds_from_stats(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Expand the training split's 14-D empirical limits to 16 Isaac joints."""
    statistics = json.loads(path.read_text(encoding="utf-8"))["action"]
    lower = np.asarray(statistics["min"], dtype=np.float32)
    upper = np.asarray(statistics["max"], dtype=np.float32)
    if lower.shape != (14,) or upper.shape != (14,) or not np.all(lower <= upper):
        raise ValueError("Expected valid 14-D action limits from the training split")

    def expand(values: np.ndarray) -> np.ndarray:
        return np.concatenate((values[:6], np.repeat(values[6], 2), values[7:13], np.repeat(values[13], 2)))

    margin = np.asarray([0.10] * 6 + [0.0] * 2 + [0.10] * 6 + [0.0] * 2, dtype=np.float32)
    return expand(lower) - margin, expand(upper) + margin


def main() -> None:
    parser = get_isaaclab_arena_cli_parser()
    parser.add_argument("demo", type=Path, help="Held-out re-recording used only for initial state")
    parser.add_argument("--output", required=True, type=Path, help="New MP4 path")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Checkpoint served by the policy process")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--execute-steps", type=int, default=8)
    parser.add_argument("--record-every", type=int, default=2)
    parser.add_argument(
        "--expected-policy-fps", type=float, help="Expected action rate; defaults to dataset info.json FPS"
    )
    parser.add_argument("--action-alpha", type=float, default=0.2, help="Causal low-pass coefficient for joint targets")
    parser.add_argument("--max-joint-step", type=float, default=0.06, help="Maximum arm-joint change per control step")
    parser.add_argument("--max-gripper-step", type=float, default=0.008, help="Maximum gripper change per control step")
    parser.add_argument("--prompt", help="Override the demonstration's task instruction")
    parser.add_argument(
        "--phase-router",
        action="store_true",
        help="Route six language subtasks from observed object/gripper milestones; never substitute actions",
    )
    parser.add_argument(
        "--phase-context", action="store_true", help="Prefix phase prompts with the original task instruction"
    )
    parser.add_argument(
        "--phase-arms", nargs=3, choices=("left", "right"), default=("right", "left", "right")
    )
    parser.add_argument("--initial-state-split", choices=("train", "heldout"), default="heldout")
    parser.add_argument("--flow-steps", type=int, default=4, help="N1.7 server flow integration steps for audit")
    parser.add_argument("--sample-average", type=int, default=1, help="N1.7 server sample count for audit")
    parser.add_argument("--policy-seed", type=int, default=42, help="RNG seed configured in the policy server")
    parser.add_argument(
        "--policy-camera-bgr-swap",
        action="store_true",
        help="Match legacy RoboTwin JPEG training data that had RGB/BGR reversed",
    )
    parser.add_argument("--snapshot-only", action="store_true", help="Save initial policy images without inference")
    parser.add_argument(
        "--scene-style",
        choices=SCENE_STYLES,
        default="factory",
        help="no-factory-dressing is a visual-domain diagnostic, not a formal factory result",
    )
    parser.add_argument(
        "--action-stats", required=True, type=Path, help="Train-split statistics for action safety limits"
    )
    add_example_environments_cli_args(parser)
    # Some Arena revisions expose --seed with None as the default. Make the
    # paired diagnostic default explicit without registering the flag twice.
    parser.set_defaults(seed=42)
    args = parser.parse_args()
    try:
        validate_seed(args.seed)
        validate_seed(args.policy_seed)
    except ValueError as error:
        parser.error(str(error))

    if args.max_steps < 1 or args.execute_steps < 1 or args.record_every < 1:
        parser.error("max-steps, execute-steps and record-every must be positive")
    if args.flow_steps < 1 or args.sample_average < 1:
        parser.error("flow-steps and sample-average must be positive")
    if not 0 < args.action_alpha <= 1 or min(args.max_joint_step, args.max_gripper_step) <= 0:
        parser.error("Action smoothing and step limits must be positive")
    if not args.demo.is_file() or not args.checkpoint.is_dir():
        parser.error("The held-out demo and N1.7 checkpoint must exist")
    if not args.action_stats.is_file():
        parser.error("Training action statistics are required")
    if args.output.exists() or args.output.with_suffix(".json").exists():
        parser.error("Refusing to overwrite existing recording or manifest")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    action_lower, action_upper = action_bounds_from_stats(args.action_stats)
    # Persist the effective clipping limits in the canonical logical order.
    # These include the existing arm margin; this does not alter the controller.
    logical_clip_limits = collapse_isaac_action(np.stack((action_lower, action_upper)))

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    env = None
    process = None
    actions_executed: list[np.ndarray] = []
    block_positions: list[np.ndarray] = []
    block_linear_velocities: list[np.ndarray] = []
    block_angular_velocities: list[np.ndarray] = []
    eef_positions: list[np.ndarray] = []
    measured_states: list[np.ndarray] = []
    sorting_flags: list[bool] = []
    policy_chunks: list[np.ndarray] = []
    raw_policy_chunks: list[np.ndarray] = []
    video_control_steps: list[int] = []
    policy_calls: list[dict] = []
    policy_input_metadata = None
    start = time.monotonic()
    try:
        arena_builder = get_arena_builder_from_cli(args)
        env_name, env_cfg = arena_builder.build_registered()
        env_cfg.seed = args.seed
        env_cfg.sim.dt = 1 / 250
        env_cfg.sim.decimation = 5
        # Keep the policy-facing workcell materials and lighting identical to
        # the re-recorded training episodes. Factory dressing is outside it.
        success_term = env_cfg.terminations.success
        if success_term is None:
            raise RuntimeError("Online recording requires an explicit task success predicate")
        # Evaluate success ourselves after a stable-placement window. An auto
        # reset would replace the final physical evidence with a fresh scene.
        env_cfg.terminations.success = None
        if hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
        env = gym.make(env_name, cfg=env_cfg).unwrapped
        spawn_factory_background(args.scene_style)
        env.reset()
        env.sim.reset()

        _, initial_joint, rigid_objects = read_demo(args.demo, str(env.device))
        observation = teleport(env, initial_joint, rigid_objects)
        block_names = sorted(name for name in env.scene.rigid_objects if name.startswith("block"))
        if len(block_names) != 3:
            raise RuntimeError(f"Factory sorting requires three block objects, found {block_names}")
        initial_block_positions = np.stack(
            [to_numpy(env.scene.rigid_objects[name].data.root_pos_w[0]) for name in block_names]
        ).astype(np.float32)
        frame_names = list(env.scene["ee_frame"].data.target_frame_names)
        eef_indices = [frame_names.index(name) for name in ("left_end_effector", "right_end_effector")]
        missing = set(CAMERA_KEYS) - set(observation.get("camera_obs", {}))
        if missing:
            raise RuntimeError(f"Five-camera recording is unavailable: {sorted(missing)}")
        task_key = task_key_for(args.environment)
        prompt = args.prompt or instruction_for(args.demo, task_key)
        if args.phase_context and not args.phase_router:
            raise ValueError("--phase-context requires --phase-router")
        router = (
            ThreeBlockPhaseRouter(
                initial_block_positions,
                tuple(args.phase_arms),
                context_prompt=prompt if args.phase_context else "",
            )
            if args.phase_router
            else None
        )
        active_prompt = router.prompt if router is not None else prompt
        if args.snapshot_only:
            snapshot = build_groot_observation(
                observation["policy"],
                observation["camera_obs"],
                active_prompt,
                swap_rgb_bgr=args.policy_camera_bgr_swap,
            )
            save_policy_input_evidence(
                snapshot,
                observation["camera_obs"],
                args.output,
                scene_style=args.scene_style,
                policy_call_pending=False,
            )
            state = {key: values[0, 0].tolist() for key, values in snapshot["state"].items()}
            args.output.with_suffix(".snapshot.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
            print("[policy] initial camera and proprioception snapshots saved", flush=True)
            return
        initial_policy_state = observation["policy"]
        last_action = np.concatenate(
            (
                to_numpy(initial_policy_state["left_arm_pos"]).reshape(-1),
                np.repeat(to_numpy(initial_policy_state["left_gripper_pos"]).reshape(-1), 2),
                to_numpy(initial_policy_state["right_arm_pos"]).reshape(-1),
                np.repeat(to_numpy(initial_policy_state["right_gripper_pos"]).reshape(-1), 2),
            )
        ).astype(np.float32)
        if last_action.shape != (16,):
            raise RuntimeError(f"Unexpected initial robot state shape: {last_action.shape}")
        initial_action = last_action.copy()
        step_limits = np.asarray(
            [args.max_joint_step] * 6
            + [args.max_gripper_step] * 2
            + [args.max_joint_step] * 6
            + [args.max_gripper_step] * 2,
            dtype=np.float32,
        )

        if not prompt.strip():
            raise RuntimeError("The policy task instruction is empty")
        control_fps = 1.0 / float(env.step_dt)
        info_path = args.action_stats.with_name("info.json")
        dataset_fps = json.loads(info_path.read_text(encoding="utf-8"))["fps"] if info_path.is_file() else None
        expected_fps = args.expected_policy_fps if args.expected_policy_fps is not None else dataset_fps
        if expected_fps is None or not np.isclose(float(expected_fps), control_fps, rtol=0.01):
            raise RuntimeError(
                f"Action cadence mismatch: simulator={control_fps:g} Hz, dataset/expected={expected_fps}. "
                "Verify actual recording interval and set --expected-policy-fps explicitly if metadata was wrong."
            )
        record_fps = control_fps / args.record_every
        expected_checkpoint_identity = checkpoint_identity(args.checkpoint)
        ffmpeg_args = ffmpeg_command(args.output, record_fps)
        ffmpeg_args[ffmpeg_args.index("+faststart")] = "+frag_keyframe+empty_moov+default_base_moof"
        process = subprocess.Popen(
            ffmpeg_args,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert process.stdin is not None

        success = False
        stop_reason = "max_steps"
        chunk = np.empty((0, 16), dtype=np.float32)
        chunk_index = 0
        with PolicyTransportClient(args.host, args.port) as client:
            if not client.ping():
                raise RuntimeError("GR00T policy server is not responding")
            model_identity = client.call("get_model_identity")
            validate_model_identity(
                model_identity,
                expected_checkpoint_identity=expected_checkpoint_identity,
                expected_flow_steps=args.flow_steps,
                expected_sample_average=args.sample_average,
                expected_seed=args.policy_seed,
            )
            client.reset({"episode_index": 0, "step_index": 0})
            for step in range(args.max_steps):
                if chunk_index >= len(chunk):
                    policy_input = build_groot_observation(
                        observation["policy"],
                        observation["camera_obs"],
                        active_prompt,
                        swap_rgb_bgr=args.policy_camera_bgr_swap,
                    )
                    if policy_input_metadata is None:
                        policy_input_metadata = save_policy_input_evidence(
                            policy_input,
                            observation["camera_obs"],
                            args.output,
                            scene_style=args.scene_style,
                            policy_call_pending=True,
                        )
                    inference_start = time.monotonic()
                    prediction, prediction_info = client.get_action(policy_input)
                    if (
                        not isinstance(prediction_info, dict)
                        or prediction_info.get("model_identity_id") != model_identity["identity_id"]
                    ):
                        raise RuntimeError("Policy response is not bound to the verified N1.7 checkpoint identity")
                    all_actions = groot_action_to_isaac(prediction)
                    if all_actions.shape[0] != 1:
                        raise RuntimeError(f"Expected one environment, got {all_actions.shape[0]}")
                    raw_chunk = all_actions[0, : args.execute_steps]
                    if not len(raw_chunk):
                        raise RuntimeError("Model returned an empty action chunk")
                    chunk = np.clip(raw_chunk, action_lower, action_upper)
                    raw_policy_chunks.append(np.ascontiguousarray(raw_chunk.copy(), dtype=np.float32))
                    policy_chunks.append(np.ascontiguousarray(chunk.copy()))
                    clipped_values = int(np.count_nonzero(chunk != raw_chunk))
                    if clipped_values:
                        print(
                            f"[policy] safety projection at step {step}: "
                            f"{clipped_values}/{raw_chunk.size} values; raw_abs_max={np.max(np.abs(raw_chunk)):.3f}",
                            flush=True,
                        )
                    chunk_index = 0
                    policy_calls.append(
                        {
                            "step": step,
                            "returned_horizon": int(all_actions.shape[1]),
                            "executed_horizon": int(len(chunk)),
                            "latency_s": round(time.monotonic() - inference_start, 3),
                            "raw_abs_max": round(float(np.max(np.abs(raw_chunk))), 4),
                            "clipped_values": clipped_values,
                            "action_sha256": action_sha256(chunk),
                            "raw_action_sha256": action_sha256(raw_chunk),
                            "model_identity_id": prediction_info["model_identity_id"],
                            "language_prompt": active_prompt,
                        }
                    )

                target = np.asarray(chunk[chunk_index], dtype=np.float32)
                desired_step = args.action_alpha * (target - last_action)
                action = np.ascontiguousarray(last_action + np.clip(desired_step, -step_limits, step_limits))
                last_action = action
                chunk_index += 1
                observation, _, terminated, truncated, _ = env.step(
                    torch.as_tensor(action, device=env.device).unsqueeze(0)
                )
                actions_executed.append(action)
                block_positions.append(
                    np.stack(
                        [to_numpy(env.scene.rigid_objects[name].data.root_pos_w[0]) for name in block_names]
                    ).astype(np.float32)
                )
                block_linear_velocities.append(
                    np.stack(
                        [to_numpy(env.scene.rigid_objects[name].data.root_lin_vel_w[0]) for name in block_names]
                    ).astype(np.float32)
                )
                block_angular_velocities.append(
                    np.stack(
                        [to_numpy(env.scene.rigid_objects[name].data.root_ang_vel_w[0]) for name in block_names]
                    ).astype(np.float32)
                )
                eef_positions.append(to_numpy(env.scene["ee_frame"].data.target_pos_w[0])[eef_indices].copy())
                measured_states.append(measured_state(observation["policy"]))
                sorting_flags.append(success_from_term(env, success_term))
                if router is not None and router.observe(
                    block_positions=block_positions[-1],
                    block_linear_velocities=block_linear_velocities[-1],
                    eef_positions=eef_positions[-1],
                    gripper_positions=measured_states[-1][[6, 13]],
                    control_step=step + 1,
                ):
                    active_prompt = router.prompt
                    # Finish this genuine model chunk before replanning, so the
                    # saved action trace remains exactly reconstructible.
                    print(f"[policy] phase router -> {router.phase_index}: {active_prompt}", flush=True)
                if step % args.record_every == 0:
                    video_prompt = (
                        "Current subtask: " + active_prompt.rsplit(" Current subtask: ", 1)[-1]
                        if args.phase_context
                        else active_prompt
                    )
                    frame = compose_frame(
                        observation["camera_obs"],
                        step // args.record_every,
                        (args.max_steps + args.record_every - 1) // args.record_every,
                        record_fps,
                        task_key,
                        video_prompt,
                        title_override=scene_video_title(args.scene_style, args.initial_state_split),
                    )
                    process.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
                    video_control_steps.append(step + 1)
                if bool(torch.as_tensor(truncated).any()):
                    stop_reason = "unexpected_truncation"
                    print(f"[policy] simulator truncated at step {step + 1}", flush=True)
                    break
                if bool(torch.as_tensor(terminated).any()):
                    stop_reason = "unexpected_termination"
                    print(f"[policy] simulator terminated at step {step + 1}", flush=True)
                    break
                if (step + 1) % 25 == 0:
                    acceptance = evaluate_grasp_sorting(
                        initial_block_positions=initial_block_positions,
                        block_positions=np.stack(block_positions),
                        block_linear_velocities=np.stack(block_linear_velocities),
                        block_angular_velocities=np.stack(block_angular_velocities),
                        eef_positions=np.stack(eef_positions),
                        measured_states=np.stack(measured_states),
                        sorting_predicate=np.asarray(sorting_flags),
                        step_dt=float(env.step_dt),
                    )
                    if acceptance["success"]:
                        success = True
                        stop_reason = "grasp_sort_and_stable_placement"
                        print(f"[policy] kinematic grasp/sort gate passed at step {step + 1}", flush=True)
                        break
                if (step + 1) % 100 == 0:
                    print(f"[policy] step={step + 1} calls={len(policy_calls)}", flush=True)

        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        if process.wait():
            raise RuntimeError(f"ffmpeg recording failed: {stderr}")
        video_metadata = video_evidence(
            args.output,
            steps=len(actions_executed),
            record_every=args.record_every,
            record_fps=record_fps,
            frames_written=len(video_control_steps),
        )
        acceptance = evaluate_grasp_sorting(
            initial_block_positions=initial_block_positions,
            block_positions=np.stack(block_positions),
            block_linear_velocities=np.stack(block_linear_velocities),
            block_angular_velocities=np.stack(block_angular_velocities),
            eef_positions=np.stack(eef_positions),
            measured_states=np.stack(measured_states),
            sorting_predicate=np.asarray(sorting_flags),
            step_dt=float(env.step_dt),
        )
        success = acceptance["success"] and stop_reason not in ("unexpected_truncation", "unexpected_termination")
        np.savez_compressed(
            args.output.with_suffix(".npz"),
            actions=np.stack(actions_executed),
            policy_chunks=np.stack(policy_chunks),
            raw_policy_chunks=np.stack(raw_policy_chunks),
            clip_lower=logical_clip_limits[0],
            clip_upper=logical_clip_limits[1],
            video_control_steps=np.asarray(video_control_steps, dtype=np.int64),
            initial_action=initial_action,
            block_positions=np.stack(block_positions),
            block_names=np.asarray(block_names),
            initial_block_positions=initial_block_positions,
            block_linear_velocities=np.stack(block_linear_velocities),
            block_angular_velocities=np.stack(block_angular_velocities),
            eef_positions=np.stack(eef_positions),
            measured_states=np.stack(measured_states),
            sorting_predicate=np.asarray(sorting_flags, dtype=bool),
        )
        manifest = {
            "recording_schema_version": 2,
            "action_evidence": {
                "raw_policy_chunks": "float32 (calls, executed_horizon, 16), decoded and expanded before clipping",
                "policy_chunks": "float32 same shape, clipped targets before smoothing",
                "clip_lower_upper": "float32 (14,), effective bounds including arm margin",
                "logical_order": "left_arm[6], left_gripper[1], right_arm[6], right_gripper[1]",
                "physical_order": "left_arm[6], left_gripper[2], right_arm[6], right_gripper[2]",
                "hash_format": "SHA256 C-order float32 physical-joint bytes",
                "last_chunk": "may contain unused trailing targets after episode stops",
                "video_control_steps": "int64 1-based executed control steps captured after env.step",
            },
            **video_metadata,
            **scene_evidence(args.scene_style),
            "first_policy_input": policy_input_metadata,
            "status": "success" if success else "task_not_completed",
            "stop_reason": stop_reason,
            "acceptance": acceptance,
            "video_review": "pending",
            "controller_source": "GR00T_N1.7_model_inference",
            "replay_used": False,
            "recorded_actions_used_after_reset": False,
            "initial_state_source": str(args.demo),
            "initial_state_split": args.initial_state_split,
            "checkpoint": str(args.checkpoint),
            "model_identity": model_identity,
            "expected_checkpoint_identity": expected_checkpoint_identity,
            "model_sampling": {"flow_steps": args.flow_steps, "sample_average": args.sample_average},
            "policy_seed": args.policy_seed,
            "simulation_seed": args.seed,
            "seed_determinism_note": SEED_NOTE,
            "environment": env_name,
            "task_instruction": prompt,
            "phase_router": {
                "enabled": router is not None,
                "arm_order": list(args.phase_arms) if router is not None else None,
                "context_prompt": prompt if router is not None and args.phase_context else None,
                "prompts": list(router.prompts) if router is not None else None,
                "events": router.events if router is not None else [],
                "action_source": "GR00T_N1.7_model_inference_only",
            },
            "policy_camera_bgr_swap": args.policy_camera_bgr_swap,
            "steps_executed": len(actions_executed),
            "policy_calls": policy_calls,
            "video": str(args.output),
            "trace": str(args.output.with_suffix(".npz")),
            "final_block_positions_m": {
                name: block_positions[-1][index].round(5).tolist() for index, name in enumerate(block_names)
            },
            "record_fps": record_fps,
            "control_fps": control_fps,
            "step_dt": float(env.step_dt),
            "dataset_fps": dataset_fps,
            "expected_policy_fps": expected_fps,
            "action_safety": {
                "bounds_from": str(args.action_stats),
                "low_pass_alpha": args.action_alpha,
                "max_joint_step": args.max_joint_step,
                "max_gripper_step": args.max_gripper_step,
                "total_clipped_values": sum(call["clipped_values"] for call in policy_calls),
            },
            "cameras": list(CAMERA_KEYS),
            "elapsed_wall_s": round(time.monotonic() - start, 2),
        }
        args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in manifest.items() if k != "policy_calls"}, indent=2), flush=True)
        if not success:
            raise RuntimeError("The recorded online N1.7 episode did not pass the task predicate")
    except BaseException:
        traceback.print_exc()
        if process is not None and process.poll() is None:
            process.kill()
        raise
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
