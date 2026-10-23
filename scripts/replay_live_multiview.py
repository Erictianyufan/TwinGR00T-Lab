"""Replay one IsaacLab-Arena demonstration and record all five live cameras."""

from __future__ import annotations

import json
import shutil
import subprocess
import traceback
from pathlib import Path
from typing import Any

import gymnasium as gym
import h5py
import numpy as np
import torch
from isaaclab.app import AppLauncher
from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.examples.example_environments.cli import (
    add_example_environments_cli_args,
    get_arena_builder_from_cli,
)
from PIL import Image, ImageDraw, ImageFont

CANVAS_SIZE = (1920, 1080)
HEADER_HEIGHT = 60
MAIN_SIZE = (1280, 960)
SIDE_SIZE = (640, 320)
CAMERA_KEYS = (
    "third_camera_rgb",
    "front_camera_rgb",
    "head_camera_rgb",
    "left_camera_rgb",
    "right_camera_rgb",
)

TASK_LABELS = {
    "stack_bowls_two": "Two-bowl stacking",
    "stack_bowls_three": "Three-bowl stacking",
    "stack_blocks_two": "Two-block stacking",
    "stack_blocks_three": "Three-block stacking",
    "handover_block": "Bimanual block handover",
    "blocks_ranking_rgb": "RGB block ranking",
}

TASK_INSTRUCTIONS = {
    "stack_bowls_two": "Stack the two bowls",
    "stack_bowls_three": "Stack the three bowls",
    "stack_blocks_two": "Stack the two blocks",
    "stack_blocks_three": "Stack the three blocks",
    "handover_block": "Hand the block from one gripper to the other",
    "blocks_ranking_rgb": "Arrange the red, green, and blue blocks in order",
}


def configure_showcase_materials(env_cfg: Any) -> None:
    """Apply presentation-friendly materials before the environment is spawned."""
    import isaaclab.sim as sim_utils

    env_cfg.scene.robot.spawn.visual_material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.22, 0.42, 0.62),
        metallic=0.08,
        roughness=0.3,
    )
    env_cfg.scene.robotwin_table.spawn.visual_material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.34, 0.18, 0.07),
        metallic=0.0,
        roughness=0.5,
    )


def spawn_indoor_showcase(asset_root: Path) -> None:
    """Create a softly lit indoor room around the RoboTwin workcell."""
    import isaaclab.sim as sim_utils

    def panel(
        path: str,
        size: tuple[float, float, float],
        position: tuple[float, float, float],
        color: tuple[float, float, float],
        roughness: float = 0.72,
    ) -> None:
        cfg = sim_utils.CuboidCfg(
            size=size,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color,
                metallic=0.0,
                roughness=roughness,
            ),
        )
        cfg.func(path, cfg, translation=position)

    # A complete room keeps every fixed and wrist camera from looking into a black void.
    panel("/World/ShowcaseRoom/Floor", (5.0, 5.0, 0.1), (0.0, 0.0, -0.08), (0.20, 0.17, 0.14))
    panel("/World/ShowcaseRoom/BackWall", (5.0, 0.10, 3.0), (0.0, -2.0, 1.45), (0.62, 0.59, 0.53))
    panel("/World/ShowcaseRoom/FrontWall", (5.0, 0.10, 3.0), (0.0, 2.0, 1.45), (0.58, 0.61, 0.64))
    panel("/World/ShowcaseRoom/LeftWall", (0.10, 4.0, 3.0), (-2.45, 0.0, 1.45), (0.48, 0.53, 0.58))
    panel("/World/ShowcaseRoom/RightWall", (0.10, 4.0, 3.0), (2.45, 0.0, 1.45), (0.48, 0.53, 0.58))

    # Muted blue wall panels give the main camera depth without competing with the bowls.
    panel("/World/ShowcaseRoom/AccentLeft", (1.0, 0.035, 1.25), (-1.1, -1.93, 1.48), (0.16, 0.28, 0.40), 0.55)
    panel("/World/ShowcaseRoom/AccentRight", (1.0, 0.035, 1.25), (1.1, -1.93, 1.48), (0.16, 0.28, 0.40), 0.55)

    hdr = asset_root / "assets" / "texture" / "base0.exr"
    dome = sim_utils.DomeLightCfg(
        intensity=2200.0,
        color=(0.96, 0.92, 0.86),
        texture_file=str(hdr) if hdr.exists() else None,
        visible_in_primary_ray=False,
    )
    dome.func("/World/ShowcaseRoom/DomeLight", dome)

    key = sim_utils.SphereLightCfg(
        intensity=14000.0,
        radius=0.45,
        color=(1.0, 0.82, 0.67),
        normalize=True,
    )
    key.func("/World/ShowcaseRoom/KeyLight", key, translation=(1.25, 0.75, 2.55))

    fill = sim_utils.SphereLightCfg(
        intensity=9000.0,
        radius=0.5,
        color=(0.68, 0.82, 1.0),
        normalize=True,
    )
    fill.func("/World/ShowcaseRoom/FillLight", fill, translation=(-1.35, 0.3, 2.25))

    rim = sim_utils.SphereLightCfg(
        intensity=7500.0,
        radius=0.35,
        color=(1.0, 0.95, 0.84),
        normalize=True,
    )
    rim.func("/World/ShowcaseRoom/RimLight", rim, translation=(0.0, -1.45, 2.35))


def load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        return ImageFont.load_default()


def to_rgb(value: Any) -> Image.Image:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim == 4:
        array = array[0]
    if array.dtype != np.uint8:
        maximum = float(array.max()) if array.size else 0.0
        if maximum <= 1.5:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.shape[-1] == 4:
        array = array[..., :3]
    return Image.fromarray(np.ascontiguousarray(array))


def letterbox(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    panel = Image.new("RGB", size, (12, 15, 20))
    scale = min(size[0] / image.width, size[1] / image.height)
    scaled_size = (round(image.width * scale), round(image.height * scale))
    image = image.resize(scaled_size, Image.Resampling.LANCZOS)
    panel.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return panel


def draw_label(canvas: Image.Image, xy: tuple[int, int], text: str, font: ImageFont.ImageFont) -> None:
    draw = ImageDraw.Draw(canvas, "RGBA")
    x, y = xy
    bounds = draw.textbbox((x + 14, y + 12), text, font=font)
    draw.rounded_rectangle((x + 8, y + 7, bounds[2] + 9, bounds[3] + 8), 7, fill=(0, 0, 0, 180))
    draw.text((x + 14, y + 12), text, font=font, fill=(255, 255, 255, 255))


def fit_text_line(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, width: int) -> str:
    """Fit display-only text in one line without changing the policy instruction."""
    text = " ".join(text.split())

    def fits(value: str) -> bool:
        left, _, right, _ = draw.textbbox((0, 0), value, font=font)
        return right - left <= width

    if fits(text):
        return text
    suffix = "..."
    if not fits(suffix):
        return ""
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if fits(text[:middle] + suffix):
            low = middle
        else:
            high = middle - 1
    return text[:low] + suffix


def task_key_for(environment: str) -> str:
    module = environment.split(":", 1)[0]
    name = module.rsplit(".", 1)[-1]
    return name.removesuffix("_environment")


def instruction_for(demo: Path, task_key: str) -> str:
    path = demo.parent.parent / "instructions" / f"{demo.stem}.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        for key in ("seen", "unseen"):
            values = payload.get(key, [])
            if values:
                return str(values[0])
    return TASK_INSTRUCTIONS.get(task_key, task_key.replace("_", " ").capitalize())


def read_demo(path: Path, device: str) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    with h5py.File(path, "r") as handle:
        actions = torch.as_tensor(
            handle["observations/policy/actions"][:], dtype=torch.float32, device=device
        )
        initial_joint = torch.as_tensor(
            handle["initial_state/articulation/robot/joint_position"][:],
            dtype=torch.float32,
            device=device,
        )
        rigid_objects = {
            name: torch.as_tensor(group["root_pose"][:], dtype=torch.float32, device=device)
            for name, group in handle["initial_state/rigid_object"].items()
        }
    return actions, initial_joint, rigid_objects


def interpolate_actions(actions: torch.Tensor, factor: int) -> torch.Tensor:
    """Restore smooth control-rate targets after converter step skipping."""
    if factor <= 1 or len(actions) <= 1:
        return actions
    segments = []
    weights = torch.arange(factor, device=actions.device, dtype=actions.dtype) / factor
    for start, end in zip(actions[:-1], actions[1:], strict=True):
        segments.append(start.unsqueeze(0) + weights.unsqueeze(1) * (end - start).unsqueeze(0))
    segments.append(actions[-1].unsqueeze(0))
    return torch.cat(segments, dim=0)


def block_stability_metrics(env: gym.Env) -> dict[str, float | bool]:
    block_assets = [
        asset for name, asset in env.scene.rigid_objects.items() if name.startswith("block")
    ]
    block_assets.sort(key=lambda asset: float(asset.data.root_pos_w[0, 2]))
    if len(block_assets) < 2:
        raise ValueError("Block stability validation requires at least two blocks")
    positions = torch.stack([asset.data.root_pos_w[0] for asset in block_assets])
    quaternions = torch.stack([asset.data.root_quat_w[0] for asset in block_assets])
    centered_xy = float(torch.max(torch.abs(positions[1:, :2] - positions[:-1, :2])))
    upright_cos = 1.0 - 2.0 * (quaternions[:, 1] ** 2 + quaternions[:, 2] ** 2)
    min_upright_cos = float(torch.min(upright_cos))
    max_linear_speed = float(
        torch.max(
            torch.stack(
                [torch.linalg.vector_norm(asset.data.root_lin_vel_w[0]) for asset in block_assets]
            )
        )
    )
    max_angular_speed = float(
        torch.max(
            torch.stack(
                [torch.linalg.vector_norm(asset.data.root_ang_vel_w[0]) for asset in block_assets]
            )
        )
    )
    stable = (
        centered_xy <= 0.015
        and min_upright_cos >= 0.994
        and max_linear_speed <= 0.02
        and max_angular_speed <= 0.20
    )
    return {
        "centered_xy_m": centered_xy,
        "min_upright_cos": min_upright_cos,
        "max_linear_speed_m_s": max_linear_speed,
        "max_angular_speed_rad_s": max_angular_speed,
        "pass": stable,
    }


def teleport(env: gym.Env, initial_joint: torch.Tensor, rigid_objects: dict[str, torch.Tensor]) -> dict:
    robot = env.scene["robot"]
    robot.write_joint_position_to_sim(initial_joint)
    robot.write_joint_velocity_to_sim(torch.zeros_like(initial_joint))
    robot.set_joint_position_target(initial_joint)

    for name, root_pose in rigid_objects.items():
        if name not in env.scene.rigid_objects:
            continue
        obj = env.scene.rigid_objects[name]
        obj.write_root_pose_to_sim(root_pose)
        obj.write_root_velocity_to_sim(torch.zeros((env.num_envs, 6), device=env.device))

    for _ in range(50):
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
    for _ in range(3):
        env.sim.render()
    return env.observation_manager.compute(update_history=False)


def ffmpeg_command(output: Path, fps: float) -> list[str]:
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{CANVAS_SIZE[0]}x{CANVAS_SIZE[1]}",
        "-framerate",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]


def compose_frame(
    camera_obs: dict,
    index: int,
    frames: int,
    fps: float,
    task_key: str,
    instruction: str,
    title_override: str | None = None,
) -> Image.Image:
    canvas = Image.new("RGB", CANVAS_SIZE, (6, 9, 14))
    title_font = load_font(28, bold=True)
    label_font = load_font(18, bold=True)
    info_font = load_font(18)

    main = letterbox(to_rgb(camera_obs["third_camera_rgb"]), MAIN_SIZE)
    canvas.paste(main, (0, HEADER_HEIGHT))
    draw_label(canvas, (0, HEADER_HEIGHT), "THIRD · OVERALL ROBOT + OBJECTS", label_font)

    for panel_index, (key, label) in enumerate(
        (("front_camera_rgb", "FRONT"), ("head_camera_rgb", "HEAD"))
    ):
        y = HEADER_HEIGHT + panel_index * SIDE_SIZE[1]
        canvas.paste(letterbox(to_rgb(camera_obs[key]), SIDE_SIZE), (MAIN_SIZE[0], y))
        draw_label(canvas, (MAIN_SIZE[0], y), label, label_font)

    wrist_y = HEADER_HEIGHT + 2 * SIDE_SIZE[1]
    wrist_size = (SIDE_SIZE[0] // 2, SIDE_SIZE[1])
    for wrist_index, (key, label) in enumerate(
        (("left_camera_rgb", "LEFT WRIST"), ("right_camera_rgb", "RIGHT WRIST"))
    ):
        x = MAIN_SIZE[0] + wrist_index * wrist_size[0]
        canvas.paste(letterbox(to_rgb(camera_obs[key]), wrist_size), (x, wrist_y))
        draw_label(canvas, (x, wrist_y), label, label_font)

    task_label = TASK_LABELS.get(task_key, task_key.replace("_", " ").title())
    title = title_override or f"TwinGR00T Lab  |  Isaac Sim indoor showcase  |  {task_label}"
    # Render text into separate bands so even unusual fonts cannot cover cameras.
    header = Image.new("RGB", (CANVAS_SIZE[0], HEADER_HEIGHT), (6, 9, 14))
    draw = ImageDraw.Draw(header)
    draw.text(
        (22, 13), fit_text_line(draw, title, title_font, CANVAS_SIZE[0] - 44),
        font=title_font, fill=(239, 244, 255),
    )
    canvas.paste(header, (0, 0))
    footer_y = HEADER_HEIGHT + MAIN_SIZE[1]
    footer = Image.new("RGB", (CANVAS_SIZE[0], CANVAS_SIZE[1] - footer_y), (6, 9, 14))
    draw = ImageDraw.Draw(footer)
    status = f"frame {index + 1:03d}/{frames}  |  t = {index / fps:05.2f} s"
    for y, text in ((6, status), (31, f"Instruction: {instruction}")):
        draw.text(
            (22, y), fit_text_line(draw, text, info_font, CANVAS_SIZE[0] - 44),
            font=info_font, fill=(190, 205, 225),
        )
    canvas.paste(footer, (0, footer_y))
    return canvas


def main() -> None:
    parser = get_isaaclab_arena_cli_parser()
    parser.add_argument("demo", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=31.25)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--interpolation-factor", type=int, default=1)
    parser.add_argument("--pre-roll-frames", type=int, default=0)
    parser.add_argument("--post-roll-frames", type=int, default=0)
    parser.add_argument(
        "--validate-block-stability",
        action="store_true",
        help="Fail the render manifest unless the final block stack is upright and stationary.",
    )
    add_example_environments_cli_args(parser)
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing video: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    env = None
    process = None
    try:
        arena_builder = get_arena_builder_from_cli(args)
        env_name, env_cfg = arena_builder.build_registered()
        env_cfg.sim.dt = 1 / 250
        env_cfg.sim.decimation = 5
        configure_showcase_materials(env_cfg)
        env = gym.make(env_name, cfg=env_cfg).unwrapped
        spawn_indoor_showcase(Path(__file__).resolve().parents[1] / "vendor" / "RoboTwin")
        env.reset()
        env.sim.reset()

        actions, initial_joint, rigid_objects = read_demo(args.demo, str(env.device))
        if args.max_frames > 0:
            actions = actions[: args.max_frames]
        source_frames = len(actions)
        actions = interpolate_actions(actions, args.interpolation_factor)
        observation = teleport(env, initial_joint, rigid_objects)
        available = observation.get("camera_obs", {})
        missing = [key for key in CAMERA_KEYS if key not in available]
        if missing:
            raise KeyError(f"Live environment is missing cameras: {missing}")

        task_key = task_key_for(args.environment)
        instruction = instruction_for(args.demo, task_key)
        process = subprocess.Popen(
            ffmpeg_command(args.output, args.fps), stdin=subprocess.PIPE, stderr=subprocess.PIPE
        )
        assert process.stdin is not None
        replay_actions = torch.cat(
            [
                actions[:1].repeat(max(args.pre_roll_frames, 0), 1),
                actions,
                actions[-1:].repeat(max(args.post_roll_frames, 0), 1),
            ],
            dim=0,
        )
        for index, action in enumerate(replay_actions):
            observation, _, _, _, _ = env.step(action.unsqueeze(0))
            frame = compose_frame(
                observation["camera_obs"], index, len(replay_actions), args.fps, task_key, instruction
            )
            process.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
            if index == 0 or (index + 1) % 100 == 0 or index + 1 == len(replay_actions):
                print(f"[replay] {index + 1}/{len(replay_actions)}", flush=True)

        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        return_code = process.wait()
        if return_code:
            raise RuntimeError(f"ffmpeg failed with exit code {return_code}: {stderr}")

        stability = block_stability_metrics(env) if args.validate_block_stability else None
        status = "pass" if stability is None or stability["pass"] else "fail"
        if stability is not None:
            print(f"[replay] final block stability: {json.dumps(stability)}", flush=True)

        manifest = {
            "status": status,
            "mode": "live_isaac_sim_replay",
            "environment": env_name,
            "demo": str(args.demo),
            "output": str(args.output),
            "frames": len(replay_actions),
            "source_frames": source_frames,
            "interpolation_factor": args.interpolation_factor,
            "pre_roll_frames": args.pre_roll_frames,
            "post_roll_frames": args.post_roll_frames,
            "fps": args.fps,
            "duration_seconds": len(replay_actions) / args.fps,
            "resolution": list(CANVAS_SIZE),
            "cameras": list(CAMERA_KEYS),
            "main_camera": "third_camera_rgb",
            "scene_style": "indoor_showcase",
            "task_key": task_key,
            "task": instruction,
        }
        if stability is not None:
            manifest["block_stability"] = stability
        args.output.with_suffix(".json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(manifest, indent=2), flush=True)
        if status != "pass":
            raise RuntimeError(f"Final block stack failed stability validation: {stability}")
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
