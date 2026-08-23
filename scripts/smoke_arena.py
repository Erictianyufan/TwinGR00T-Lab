"""Launch the RoboTwin Arena task headlessly and execute zero-action steps."""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import gymnasium as gym
import torch
from isaaclab.app import AppLauncher
from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.examples.example_environments.cli import (
    add_example_environments_cli_args,
    get_arena_builder_from_cli,
)


def main() -> None:
    parser = get_isaaclab_arena_cli_parser()
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--output", type=Path)
    add_example_environments_cli_args(parser)
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    env = None
    try:
        print("[smoke] simulator ready; building Arena environment", flush=True)
        arena_builder = get_arena_builder_from_cli(args)
        env_name, env_cfg = arena_builder.build_registered()
        env_cfg.sim.dt = 1 / 250
        env_cfg.sim.decimation = 5
        env = gym.make(env_name, cfg=env_cfg).unwrapped
        print(f"[smoke] environment ready: {env_name}", flush=True)
        observation, _ = env.reset()

        action_dim = sum(env.action_manager.action_term_dim)
        actions = torch.zeros((env.num_envs, action_dim), dtype=torch.float32, device=env.device)
        for _ in range(args.steps):
            observation, _, _, _, _ = env.step(actions)

        policy = observation.get("policy", {})
        camera_obs = observation.get("camera_obs", {})
        result = {
            "status": "pass",
            "environment": env_name,
            "device": str(env.device),
            "num_envs": env.num_envs,
            "steps": args.steps,
            "action_dim": action_dim,
            "policy_keys": sorted(policy),
            "camera_shapes": {name: list(value.shape) for name, value in camera_obs.items()},
        }
        rendered = json.dumps(result, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered, flush=True)
    except BaseException as exc:
        failure = {
            "status": "fail",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        rendered = json.dumps(failure, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered, flush=True)
        raise
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
