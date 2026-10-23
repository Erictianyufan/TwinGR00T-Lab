"""Re-execute source teachers in the factory to collect physically checked data.

This is explicitly teacher replay for training, NOT GR00T inference. Images and
states are measured after every real env.step; conversion must use offset=1.
No source image, post-step measurement, or artificial frozen tail is reused.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
import traceback
from pathlib import Path

import numpy as np

from twingr00t_lab.conversion import episode_number, select_demo_paths
from twingr00t_lab.grasp_acceptance import evaluate_grasp_sorting
from twingr00t_lab.policy_bridge import to_numpy
from twingr00t_lab.seeding import validate_seed
from twingr00t_lab.teacher_collection import (
    POLICY_CAMERAS,
    TeacherHdf5Writer,
    collection_metadata,
    named_eef_indices,
    replay_plan,
    sha256_file,
)


def scene_snapshot(env: object) -> dict[str, np.ndarray]:
    """Read physical state in world coordinates, not stale policy EEF fields."""
    values = {}

    def flatten(tree: dict, prefix: str = "") -> None:
        for key, value in tree.items():
            path = f"{prefix}/{key}" if prefix else key
            if isinstance(value, dict):
                flatten(value, path)
            else:
                array = to_numpy(value)
                if array.ndim < 1 or array.shape[0] != 1:
                    raise ValueError(f"Expected one-environment physical state for {path}, got {array.shape}")
                values[path] = np.asarray(array[0], dtype=np.float32).copy()

    flatten(env.scene.get_state(is_relative=False))
    for required in ("articulation/robot/joint_position", "articulation/robot/root_pose"):
        if required not in values:
            raise RuntimeError(f"Scene state lacks read_demo-compatible field: {required}")
    return values


def collect_episode(env: object, source: Path, args: object, success_term: object, cameras: tuple[str, ...]) -> dict:
    import torch
    from replay_live_multiview import read_demo, teleport
    from run_aloha_n17_closedloop import measured_state, success_from_term

    episode_id = episode_number(source)
    instruction = source.parent.parent / "instructions" / f"{source.stem}.json"
    actions, initial_joint, rigid_objects = read_demo(source, str(env.device))
    source_actions = to_numpy(actions)
    targets, source_indices = replay_plan(
        source_actions, step_dt=float(env.step_dt), stable_tail_s=args.stable_tail_seconds
    )
    report = collection_metadata(
        source_demo=source,
        source_instruction=instruction,
        episode_id=episode_id,
        source_steps=len(actions),
        total_steps=len(targets),
        step_dt=float(env.step_dt),
        cameras=cameras,
    )
    report.update(environment=args.environment, simulation_seed=args.seed, jpeg_quality=args.jpeg_quality)
    report["source_action_sha256"] = hashlib.sha256(source_actions.tobytes(order="C")).hexdigest()
    diagnostic = args.output_dir / "diagnostics" / f"{source.stem}.hdf5"
    writer = None
    accepted = False
    start = time.monotonic()
    try:
        env.reset()
        missing_objects = set(rigid_objects) - set(env.scene.rigid_objects)
        if missing_objects or not {"block1", "block2", "block3"}.issubset(rigid_objects):
            raise RuntimeError(f"Source rigid objects do not match the scene: missing={sorted(missing_objects)}")
        observation = teleport(env, initial_joint, rigid_objects)
        if env.num_envs != 1:
            raise RuntimeError("Teacher collection requires exactly one environment")
        missing = set(cameras) - set(observation.get("camera_obs", {}))
        if missing:
            raise RuntimeError(f"Factory cameras unavailable: {sorted(missing)}")
        initial = scene_snapshot(env)
        block_names = sorted(name for name in env.scene.rigid_objects if name.startswith("block"))
        if block_names != ["block1", "block2", "block3"]:
            raise RuntimeError(f"Expected the three RGB ranking blocks, got {block_names}")
        initial_positions = np.stack([initial[f"rigid_object/{name}/root_pose"][:3] for name in block_names])
        eef_ids = named_eef_indices(list(env.scene["ee_frame"].data.target_frame_names))
        report["eef_indices"] = {"left_end_effector": eef_ids[0], "right_end_effector": eef_ids[1]}
        report["initial_state_source"] = "actual_scene_after_source_initial_state_teleport_and_physics_settle"
        writer = TeacherHdf5Writer(diagnostic, source, report, initial)
        positions, linear, angular, eef, states, sorted_flags = [], [], [], [], [], []
        for index, target in enumerate(targets):
            observation, _, terminated, truncated, _ = env.step(torch.as_tensor(target, device=env.device).unsqueeze(0))
            if np.asarray(to_numpy(terminated)).any() or np.asarray(to_numpy(truncated)).any():
                report["unexpected_termination_step"] = index + 1
                raise RuntimeError("Unexpected termination/reset; this episode cannot be promoted to training")
            snapshot = scene_snapshot(env)
            measured = measured_state(observation["policy"])
            frame = env.scene["ee_frame"].data
            wrist_pos = np.stack([to_numpy(frame.target_pos_w[0, item]).copy() for item in eef_ids])
            wrist_quat = np.stack([to_numpy(frame.target_quat_w[0, item]).copy() for item in eef_ids])
            block_pose = np.stack([snapshot[f"rigid_object/{name}/root_pose"] for name in block_names])
            block_velocity = np.stack([snapshot[f"rigid_object/{name}/root_velocity"] for name in block_names])
            sorting = success_from_term(env, success_term)
            policy = {
                "actions": target.copy(),
                "left_arm_actions": target[:6],
                "left_gripper_actions": target[6:7],
                "right_arm_actions": target[8:14],
                "right_gripper_actions": target[14:15],
                "left_arm_pos": measured[:6],
                "left_gripper_pos": measured[6:7],
                "right_arm_pos": measured[7:13],
                "right_gripper_pos": measured[13:14],
                "left_eef_pos": wrist_pos[0],
                "right_eef_pos": wrist_pos[1],
                "left_eef_quat": wrist_quat[0],
                "right_eef_quat": wrist_quat[1],
            }
            values = {f"observations/policy/{key}": value for key, value in policy.items()}
            values.update({f"states/{key}": value for key, value in snapshot.items()})
            values.update(
                {
                    "processed_actions": target.copy(),
                    "evidence/source_action_index": np.int64(source_indices[index]),
                    "evidence/is_stable_tail": np.bool_(index >= len(actions)),
                    "evidence/sorting_predicate": np.bool_(sorting),
                    "evidence/control_step": np.int64(index + 1),
                    "evidence/timestamp_s": np.float64((index + 1) * env.step_dt),
                }
            )
            writer.append(
                values,
                {name: to_numpy(observation["camera_obs"][name]) for name in cameras},
                jpeg_quality=args.jpeg_quality,
            )
            positions.append(block_pose[:, :3])
            linear.append(block_velocity[:, :3])
            angular.append(block_velocity[:, 3:])
            eef.append(wrist_pos)
            states.append(measured)
            sorted_flags.append(sorting)
            if (index + 1) % 100 == 0 or index + 1 == len(targets):
                print(f"[teacher replay, NOT model inference] {source.stem} {index + 1}/{len(targets)}", flush=True)
        report["acceptance"] = evaluate_grasp_sorting(
            initial_block_positions=initial_positions,
            block_positions=np.stack(positions),
            block_linear_velocities=np.stack(linear),
            block_angular_velocities=np.stack(angular),
            eef_positions=np.stack(eef),
            measured_states=np.stack(states),
            sorting_predicate=np.asarray(sorted_flags),
            step_dt=float(env.step_dt),
        )
        if sha256_file(source) != report["source_demo_sha256"]:
            raise RuntimeError("Source demonstration changed during re-collection")
        if sha256_file(instruction) != report["source_instruction_sha256"]:
            raise RuntimeError("Source instructions changed during re-collection")
        accepted = bool(report["acceptance"]["success"])
        report["status"] = "accepted_for_training" if accepted else "rejected_physics"
    except Exception as error:
        report["status"] = "error"
        report["error"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
    finally:
        if writer is not None:
            report["total_steps"] = writer.rows
            writer.close(accepted=accepted)
        report["elapsed_wall_s"] = round(time.monotonic() - start, 3)
    if diagnostic.is_file():
        report["hdf5_sha256"] = sha256_file(diagnostic)
        report["hdf5"] = diagnostic.relative_to(args.output_dir).as_posix()
    # Preserve the original instructions verbatim; never relabel an unsuccessful
    # action sequence as a shorter or easier task.
    copied_instruction = args.output_dir / "diagnostics" / f"{source.stem}.instructions.json"
    shutil.copyfile(instruction, copied_instruction)
    report["instruction_sha256"] = sha256_file(copied_instruction)
    if accepted:
        destination = args.output_dir / "data" / diagnostic.name
        copied_instruction.rename(args.output_dir / "instructions" / instruction.name)
        diagnostic.rename(destination)
        report["hdf5"] = destination.relative_to(args.output_dir).as_posix()
        report["training_eligible"] = True
    (args.output_dir / "reports" / f"{source.stem}.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    import gymnasium as gym
    from isaaclab.app import AppLauncher
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.examples.example_environments.cli import (
        add_example_environments_cli_args,
        get_arena_builder_from_cli,
    )
    from replay_live_multiview import CAMERA_KEYS
    from run_aloha_n17_closedloop import spawn_factory_background

    parser = get_isaaclab_arena_cli_parser()
    parser.add_argument("--input-dir", required=True, type=Path, help="Source dataset with data/ and instructions/")
    parser.add_argument("--output-dir", required=True, type=Path, help="Must be a NEW dataset directory")
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1, help="Start with one teacher; use 9 only after inspection")
    parser.add_argument("--stable-tail-seconds", type=float, default=1.0)
    parser.add_argument("--source-fps", type=float, default=62.5, help="Verified cadence of the source command stream")
    parser.add_argument("--camera-set", choices=("policy", "all"), default="policy")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.set_defaults(seed=42)
    add_example_environments_cli_args(parser)
    args = parser.parse_args()
    try:
        validate_seed(args.seed)
        if args.num_envs != 1 or args.example_environment != "blocks_ranking_rgb":
            raise ValueError("Only one-environment blocks_ranking_rgb collection is supported")
        if getattr(args, "embodiment", None) != "aloha":
            raise ValueError("Factory teacher collection requires --embodiment aloha")
        if not 1 <= args.jpeg_quality <= 100:
            raise ValueError("jpeg-quality must lie in [1, 100]")
        # Validate the tail and FPS even before the simulator is started.
        replay_plan(
            np.zeros((1, 16), dtype=np.float32), step_dt=1 / args.source_fps, stable_tail_s=args.stable_tail_seconds
        )
        sources = select_demo_paths(args.input_dir, start_episode=args.start_episode, limit=args.limit)
        if len(sources) != args.limit:
            raise ValueError(f"Requested {args.limit} episodes but found {len(sources)}")
        for source in sources:
            instruction = source.parent.parent / "instructions" / f"{source.stem}.json"
            payload = json.loads(instruction.read_text(encoding="utf-8"))
            if not isinstance(payload.get("seen"), list) or not any(
                isinstance(text, str) and text.strip() for text in payload["seen"]
            ):
                raise ValueError(f"Missing usable source seen instruction: {instruction}")
        args.output_dir.mkdir(parents=True, exist_ok=False)
    except (ValueError, OSError, ZeroDivisionError) as error:
        parser.error(str(error))
    for name in ("data", "instructions", "reports", "diagnostics"):
        (args.output_dir / name).mkdir()
    launcher = AppLauncher(args)
    env = None
    summaries = []
    try:
        env_name, cfg = get_arena_builder_from_cli(args).build_registered()
        cfg.seed = args.seed
        cfg.sim.dt = 1 / 250
        cfg.recorders = None
        # Use the same top-level control decimation as the live pipeline. Never
        # interpolate or silently repeat source commands to mask a cadence gap.
        success_term = cfg.terminations.success
        if success_term is None:
            raise RuntimeError("The environment must provide an explicit sorting predicate")
        cfg.terminations.success = None
        if hasattr(cfg.terminations, "time_out"):
            cfg.terminations.time_out = None
        env = gym.make(env_name, cfg=cfg).unwrapped
        spawn_factory_background("factory")
        env.reset()
        env.sim.reset()
        if not np.isclose(1 / float(env.step_dt), args.source_fps, rtol=0.001):
            raise RuntimeError(f"Cadence mismatch: source={args.source_fps}, simulator={1 / float(env.step_dt)} Hz")
        cameras = CAMERA_KEYS if args.camera_set == "all" else POLICY_CAMERAS
        for source in sources:
            report = collect_episode(env, source, args, success_term, cameras)
            summaries.append(
                {
                    "source_episode_id": report["source_episode_id"],
                    "status": report["status"],
                    "training_eligible": report["training_eligible"],
                    "report": f"reports/{source.stem}.json",
                }
            )
            (args.output_dir / "collection_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "controller_source": "teacher_replay_for_training",
                        "replay_used": True,
                        "model_inference_result": False,
                        "scene_style": "factory",
                        "episodes": summaries,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            if report["status"] == "error":
                raise RuntimeError(f"Collection error in {source.stem}; diagnostics preserved, stopping batch")
        if not all(item["training_eligible"] for item in summaries):
            raise RuntimeError("Some teachers failed unchanged physical acceptance; only accepted data were promoted")
    finally:
        if env is not None:
            env.close()
        launcher.app.close()


if __name__ == "__main__":
    main()
