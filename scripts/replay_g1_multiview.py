"""Replay one recorded G1 physics trajectory and render an auditable multiview MP4."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.examples.example_environments.cli import (
    get_arena_builder_from_cli,
    get_isaaclab_arena_example_environment_cli_parser,
)
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext
from PIL import Image, ImageDraw, ImageFont
from run_g1_groot_closedloop import (
    _configure_cinematic_cameras,
    _frame_image,
    _set_camera_view,
    _spawn_cinematic_lighting,
    _start_video,
    _use_local_homie_models,
    _use_simple_home_scene,
)


def _compose(obs: dict, step: int, cargo_z: float) -> Image.Image:
    keys = ("grasp_camera_rgb", "hand_camera_rgb", "follow_camera_rgb", "robot_head_cam_rgb")
    labels = ("FULL BODY + TRAY", "HANDS + CONTACT", "FOLLOW", "G1 HEAD POV")
    canvas = Image.new("RGB", (1920, 1080), (7, 10, 15))
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    font = ImageFont.truetype(str(font_path), 22) if font_path.is_file() else ImageFont.load_default()
    small = ImageFont.truetype(str(font_path), 17) if font_path.is_file() else ImageFont.load_default()
    draw = ImageDraw.Draw(canvas, "RGBA")
    for index, (key, label) in enumerate(zip(keys, labels, strict=True)):
        x = (index % 2) * 960
        y = (index // 2) * 540
        canvas.paste(_frame_image(obs[key], (960, 540)), (x, y))
        draw.rounded_rectangle((x + 15, y + 14, x + 330, y + 52), 8, fill=(0, 0, 0, 175))
        draw.text((x + 28, y + 22), label, font=font, fill=(255, 255, 255, 255))
    status = f"MimicGen action replay · native PhysX · no state teleport · step {step:04d} · cargo z={cargo_z:.3f} m"
    draw.rounded_rectangle((350, 1023, 1570, 1070), 10, fill=(0, 0, 0, 205))
    draw.text((390, 1036), status, font=small, fill=(224, 238, 250, 255))
    return canvas


def main() -> None:
    parser = get_isaaclab_arena_cli_parser()
    parser.add_argument("--dataset_file", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--cargo_usd", type=Path, required=True)
    parser.add_argument("--cargo_scale", type=float, nargs=3, default=(1.0, 1.0, 1.0))
    parser.add_argument("--video_fps", type=float, default=25.0)
    parser.add_argument("--video_stride", type=int, default=2)
    args_cli, _ = parser.parse_known_args()

    with SimulationAppContext(args_cli):
        from isaaclab.utils.datasets import HDF5DatasetFileHandler

        _use_local_homie_models()
        _use_simple_home_scene(args_cli.cargo_usd.resolve(), tuple(args_cli.cargo_scale))
        parser = get_isaaclab_arena_example_environment_cli_parser(parser)
        args_cli = parser.parse_args()
        args_cli.enable_cameras = True

        arena_builder = get_arena_builder_from_cli(args_cli)
        env_name, env_cfg = arena_builder.build_registered()
        _configure_cinematic_cameras(env_cfg)
        env_cfg.recorders = {}
        env_cfg.terminations = {}
        env = gym.make(env_name, cfg=env_cfg).unwrapped
        _spawn_cinematic_lighting()

        dataset = HDF5DatasetFileHandler()
        dataset.open(str(args_cli.dataset_file))
        episode_name = list(dataset.get_episode_names())[0]
        episode = dataset.load_episode(episode_name, env.device)

        env.reset()
        env.reset_to(episode.get_initial_state(), torch.tensor([0], device=env.device), is_relative=True)
        _set_camera_view(env.scene["grasp_camera"], (-0.55, 1.05, 0.78), (0.38, 0.0, 0.16))
        _set_camera_view(env.scene["hand_camera"], (0.55, 0.0, 0.92), (0.55, 0.0, 0.08))
        _set_camera_view(env.scene["handoff_camera"], (1.15, -1.25, 0.85), (0.0, -0.65, 0.1))
        _set_camera_view(env.scene["follow_camera"], (-0.9, 1.5, 1.05), (0.1, 0.0, 0.3))
        env.sim.render()

        process = _start_video(args_cli.video, args_cli.video_fps, 1920, 1080)
        frames = 0
        step = 0
        initial_cargo_z = float(env.scene["brown_box"].data.root_pos_w[0, 2])
        max_cargo_z = initial_cargo_z
        try:
            while True:
                action = episode.get_next_action()
                if action is None:
                    break
                result = env.step(action.unsqueeze(0))
                obs = result[0]
                if step % args_cli.video_stride == 0:
                    robot = env.scene["robot"].data.root_pos_w[0]
                    eye = (float(robot[0]) - 0.75, float(robot[1]) + 1.25, 1.05)
                    target = (float(robot[0]) + 0.2, float(robot[1]), 0.30)
                    _set_camera_view(env.scene["follow_camera"], eye, target)
                    cargo_z = float(env.scene["brown_box"].data.root_pos_w[0, 2])
                    max_cargo_z = max(max_cargo_z, cargo_z)
                    frame = _compose(obs["camera_obs"], step, cargo_z)
                    assert process.stdin is not None
                    process.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
                    frames += 1
                step += 1
        finally:
            assert process.stdin is not None
            process.stdin.close()
            stderr = process.stderr.read().decode("utf-8", errors="replace")
            return_code = process.wait()
            env.close()
            if return_code:
                raise RuntimeError(f"ffmpeg failed with exit code {return_code}: {stderr}")
        print(f"[TwinGR00T] Recorded {frames} multiview replay frames to {args_cli.video}")
        print(f"[TwinGR00T] Replay maximum cargo lift: {max_cargo_z - initial_cargo_z:.6f} m")


if __name__ == "__main__":
    main()
