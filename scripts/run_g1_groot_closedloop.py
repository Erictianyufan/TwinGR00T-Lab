"""Run the IsaacLab-Arena G1 GR00T policy with local Homie WBC weights.

IsaacLab-Arena's upstream Homie configuration retrieves both ONNX files from
Nucleus with ``force_download=True`` every time an environment starts.  That can
stall cloud machines on the synchronous download during PhysX initialization.
This launcher preserves the upstream environment and policy implementation, but
points Homie at verified local copies before the environment is constructed.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import tqdm
from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.examples.example_environments.cli import get_arena_builder_from_cli
from isaaclab_arena.examples.policy_runner_cli import create_policy, setup_policy_argument_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext
from PIL import Image, ImageDraw, ImageFont

from twingr00t_lab.experiment_metrics import G1EpisodeMetrics


def _use_local_homie_models() -> tuple[Path, Path]:
    """Patch the Homie config to avoid an unconditional network download."""

    from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.config.configs import HomieV2Config
    from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.policy import g1_homie_policy

    model_dir = Path(
        os.environ.get(
            "TWINGR00T_HOMIE_MODEL_DIR",
            Path.cwd() / "checkpoints" / "homie_v2",
        )
    ).resolve()
    stand_path = model_dir / "stand.onnx"
    walk_path = model_dir / "walk.onnx"
    missing = [path for path in (stand_path, walk_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing local Homie WBC model(s): {missing}")

    upstream_init = HomieV2Config.__init__

    def local_init(self, *args, **kwargs):
        upstream_init(self, *args, **kwargs)
        self.wbc_model_path = f"{stand_path},{walk_path}"

    upstream_retrieve = g1_homie_policy.retrieve_file_path

    def local_retrieve(path, *args, **kwargs):
        local_path = Path(path)
        if local_path.is_file():
            print(f"[TwinGR00T] Reusing local WBC model: {local_path}")
            return str(local_path)
        return upstream_retrieve(path, *args, **kwargs)

    HomieV2Config.__init__ = local_init
    g1_homie_policy.retrieve_file_path = local_retrieve
    print(f"[TwinGR00T] Homie WBC models: {stand_path}, {walk_path}")
    return stand_path, walk_path


def _use_simple_home_scene(
    cargo_usd: Path | None = None,
    cargo_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Path | None:
    """Optionally replace the collision-heavy Galileo background."""

    scene_value = os.environ.get("TWINGR00T_SCENE_USD") or os.environ.get("TWINGR00T_SIMPLE_SCENE_USD")
    if not scene_value:
        return None
    scene_path = Path(scene_value).resolve()
    if not scene_path.is_file():
        raise FileNotFoundError(f"Simple scene USD does not exist: {scene_path}")
    scene_pose = tuple(float(value) for value in os.environ.get("TWINGR00T_SCENE_POSE_XYZ", "0,0,0").split(","))
    if len(scene_pose) != 3:
        raise ValueError("TWINGR00T_SCENE_POSE_XYZ must contain exactly three comma-separated values")
    cargo_pose_xyz = tuple(
        float(value)
        for value in os.environ.get("TWINGR00T_CARGO_POSE_XYZ", "0.58,0.18,0.071").split(",")
    )
    if len(cargo_pose_xyz) != 3:
        raise ValueError("TWINGR00T_CARGO_POSE_XYZ must contain exactly three comma-separated values")
    support_value = os.environ.get("TWINGR00T_CARGO_SUPPORT_USD")
    support_path = Path(support_value).resolve() if support_value else None
    if support_path is not None and not support_path.is_file():
        raise FileNotFoundError(f"Cargo support USD does not exist: {support_path}")
    robot_pose_xyz = tuple(
        float(value)
        for value in os.environ.get("TWINGR00T_ROBOT_POSE_XYZ", "0,0.18,0").split(",")
    )
    if len(robot_pose_xyz) != 3:
        raise ValueError("TWINGR00T_ROBOT_POSE_XYZ must contain exactly three comma-separated values")
    cargo_rotation_wxyz = tuple(
        float(value)
        for value in os.environ.get("TWINGR00T_CARGO_ROTATION_WXYZ", "1,0,0,0").split(",")
    )
    if len(cargo_rotation_wxyz) != 4:
        raise ValueError("TWINGR00T_CARGO_ROTATION_WXYZ must contain exactly four comma-separated values")

    from isaaclab_arena.assets.background import Background
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_base import ObjectType
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.examples.example_environments.galileo_g1_locomanip_pick_and_place_environment import (
        GalileoG1LocomanipPickAndPlaceEnvironment,
    )
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.g1_locomanip_pick_and_place_task import G1LocomanipPickAndPlaceTask
    from isaaclab_arena.utils.pose import Pose

    def get_env(self, args_cli):
        background = Background(
            name=scene_path.stem,
            usd_path=str(scene_path),
            object_min_z=-1.0,
            initial_pose=Pose(position_xyz=scene_pose, rotation_wxyz=(1.0, 0.0, 0.0, 0.0)),
        )
        if cargo_usd is None:
            pick_up_object = self.asset_registry.get_asset_by_name(args_cli.object)()
        else:
            pick_up_object = Object(
                name="brown_box",
                prim_path="{ENV_REGEX_NS}/brown_box",
                object_type=ObjectType.RIGID,
                usd_path=str(cargo_usd),
                scale=cargo_scale,
            )
        destination_bin = self.asset_registry.get_asset_by_name("blue_sorting_bin")()
        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(enable_cameras=args_cli.enable_cameras)
        teleop_device = (
            self.device_registry.get_device_by_name(args_cli.teleop_device)()
            if args_cli.teleop_device is not None
            else None
        )

        cargo_pose = (
            Pose(position_xyz=cargo_pose_xyz, rotation_wxyz=cargo_rotation_wxyz)
            if cargo_usd is not None
            else Pose(position_xyz=(0.5785, 0.18, 0.0707), rotation_wxyz=(0.0, 0.0, 1.0, 0.0))
        )
        pick_up_object.set_initial_pose(cargo_pose)
        destination_bin.set_initial_pose(
            Pose(position_xyz=(-0.245, -1.6272, -0.2641), rotation_wxyz=(0.0, 0.0, 0.0, 1.0))
        )
        embodiment.set_initial_pose(Pose(position_xyz=robot_pose_xyz, rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))

        if (
            args_cli.embodiment == "g1_wbc_pink"
            and (
                getattr(args_cli, "mimic", False)
                or os.environ.get("TWINGR00T_FORCE_MIMIC_PATCH", "0") == "1"
            )
            and not hasattr(args_cli, "auto")
        ):
            # Keep Arena's native locomotion Mimic patch and navigation
            # subgoals when this environment override is used for data
            # generation.  Without this block the same scene would silently
            # execute a different controller than the official workflow.
            from isaaclab_arena.utils.locomanip_mimic_patch import patch_g1_locomanip_mimic

            patch_g1_locomanip_mimic()
            action_cfg = embodiment.get_action_cfg()
            action_cfg.g1_action.use_p_control = True
            table_y = robot_pose_xyz[1]
            action_cfg.g1_action.navigation_subgoals = [
                ([0.18, table_y, 0.0], False),
                ([0.18, table_y, -1.78], True),
                ([-0.0955, -1.1070, -1.78], False),
                ([-0.0955, -1.1070, -1.78], False),
            ]

        assets = [background, pick_up_object, destination_bin]
        if support_path is not None:
            assets.append(
                Background(
                    name="cargo_support",
                    usd_path=str(support_path),
                    object_min_z=-1.0,
                    initial_pose=Pose.identity(),
                )
            )
        scene = Scene(assets=assets)
        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=G1LocomanipPickAndPlaceTask(
                pick_up_object,
                destination_bin,
                background,
                episode_length_s=30.0,
            ),
            teleop_device=teleop_device,
        )

    GalileoG1LocomanipPickAndPlaceEnvironment.get_env = get_env
    print(f"[TwinGR00T] Local scene: {scene_path}, pose={scene_pose}")
    if cargo_usd is not None:
        print(f"[TwinGR00T] Serving cargo: {cargo_usd}, scale={cargo_scale}, pose={cargo_pose_xyz}")
    if support_path is not None:
        print(f"[TwinGR00T] Cargo support: {support_path}")
    return scene_path


def _add_twingroot_args(args_parser) -> None:
    args_parser.add_argument(
        "--cargo_usd",
        type=Path,
        default=None,
        help="Replace the training box with a real rigid serving-object USD.",
    )
    args_parser.add_argument(
        "--cargo_scale",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 1.0),
        metavar=("X", "Y", "Z"),
    )
    args_parser.add_argument(
        "--action_trace",
        type=Path,
        default=None,
        help="Optional JSONL trace containing GR00T actions and simulated poses.",
    )
    args_parser.add_argument(
        "--head_video",
        type=Path,
        default=None,
        help="Optional MP4 recording of the exact head-camera images sent to GR00T.",
    )
    args_parser.add_argument(
        "--multiview_video",
        type=Path,
        default=None,
        help="Optional 2x2 cinematic MP4 with grasp, follow, handoff, and GR00T POV views.",
    )
    args_parser.add_argument(
        "--video_fps",
        type=float,
        default=25.0,
        help="Output FPS for --head_video.",
    )
    args_parser.add_argument(
        "--video_stride",
        type=int,
        default=2,
        help="Record every Nth 50 Hz control step.",
    )
    args_parser.add_argument(
        "--demo_assist",
        action="store_true",
        help="Keep real GR00T/Homie control while assisting plate grasp, carry, and release for a reliable demo.",
    )
    args_parser.add_argument(
        "--result_manifest",
        type=Path,
        default=None,
        help="Write one auditable episode summary as JSON.",
    )
    args_parser.add_argument(
        "--stop_on_terminal",
        action="store_true",
        help="Stop at the first terminal transition instead of auto-resetting.",
    )
    args_parser.add_argument(
        "--recorder_dir",
        type=Path,
        default=None,
        help="Use an isolated Isaac Lab HDF5 recorder directory for concurrent runs.",
    )


def _start_video(path: Path, fps: float, width: int, height: int) -> subprocess.Popen:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for --head_video")
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing video: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
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
            str(path),
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _tensor_list(value: torch.Tensor) -> list[float]:
    return [round(float(item), 6) for item in value.detach().cpu().flatten()]


def _json_safe(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


def _smoothstep(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value * value * (3.0 - 2.0 * value)


def _assist_navigation(actions: torch.Tensor, step: int) -> str:
    """Blend a gentle forward command into the real GR00T action chunk."""

    if step < 320:
        speed = 0.0
        phase = "gr00t_reach"
    elif step < 950:
        ramp_up = _smoothstep((step - 320) / 80.0)
        ramp_down = _smoothstep((950 - step) / 100.0)
        speed = 0.4 * min(ramp_up, ramp_down)
        phase = "wbc_walk"
    else:
        speed = 0.0
        phase = "handoff"

    actions[:, -7:-4] = torch.tensor((speed, 0.0, 0.0), dtype=actions.dtype, device=actions.device)
    return phase


def _assist_robot_translation(env, step: int, walk_start_pose: torch.Tensor) -> None:
    """Move the base smoothly while Homie continues to animate the walking gait."""

    if step < 320:
        return
    progress = _smoothstep((step - 320) / 630.0)
    robot = env.scene["robot"]
    pose = robot.data.root_pose_w[0].clone()
    pose[0] = walk_start_pose[0]
    pose[1] = walk_start_pose[1] + (-1.1 - float(walk_start_pose[1])) * progress
    pose[2] = walk_start_pose[2]
    robot.write_root_pose_to_sim(pose.unsqueeze(0))


def _assist_plate_pose(env, step: int, initial_pose: torch.Tensor) -> str:
    """Kinematically hold the plate only after GR00T has completed its reach."""

    robot = env.scene["robot"]
    plate = env.scene["brown_box"]
    left_id = robot.body_names.index("left_wrist_yaw_link")
    right_id = robot.body_names.index("right_wrist_yaw_link")
    wrist_midpoint = 0.5 * (robot.data.body_pos_w[0, left_id] + robot.data.body_pos_w[0, right_id])
    grasp_pose = initial_pose.clone()
    grasp_pose[:3] = wrist_midpoint
    grasp_pose[2] -= 0.055

    if step < 300:
        pose = initial_pose
        phase = "approach"
    elif step < 340:
        blend = _smoothstep((step - 300) / 40.0)
        pose = initial_pose.clone()
        pose[:3] = torch.lerp(initial_pose[:3], grasp_pose[:3], blend)
        phase = "grasp_constraint"
    elif step < 950:
        pose = grasp_pose
        phase = "carry_constraint"
    elif step < 1080:
        target_pose = initial_pose.clone()
        target_pose[:3] = torch.tensor((-0.25, -1.63, -0.245), dtype=target_pose.dtype, device=target_pose.device)
        blend = _smoothstep((step - 950) / 130.0)
        pose = grasp_pose.clone()
        pose[:3] = torch.lerp(grasp_pose[:3], target_pose[:3], blend)
        phase = "release_constraint"
    else:
        pose = initial_pose.clone()
        pose[:3] = torch.tensor((-0.25, -1.63, -0.245), dtype=pose.dtype, device=pose.device)
        phase = "released"

    plate.write_root_pose_to_sim(pose.unsqueeze(0))
    plate.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=pose.dtype, device=pose.device))
    return phase


def _configure_cinematic_cameras(env_cfg) -> None:
    import isaaclab.sim as sim_utils
    from isaaclab.envs import mdp
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.sensors import TiledCameraCfg

    for name in ("grasp_camera", "hand_camera", "follow_camera", "handoff_camera"):
        camera_cfg = TiledCameraCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{name}",
            update_period=0.0,
            height=540,
            width=960,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=3.0,
                clipping_range=(0.05, 20.0),
            ),
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0),
                convention="world",
            ),
        )
        setattr(env_cfg.scene, name, camera_cfg)
        setattr(
            env_cfg.observations.camera_obs,
            f"{name}_rgb",
            ObsTerm(
                func=mdp.image,
                params={"sensor_cfg": SceneEntityCfg(name), "data_type": "rgb", "normalize": False},
            ),
        )


def _set_camera_view(camera, eye: tuple[float, float, float], target: tuple[float, float, float]) -> None:
    device = camera.device
    camera.set_world_poses_from_view(
        eyes=torch.tensor([eye], dtype=torch.float32, device=device),
        targets=torch.tensor([target], dtype=torch.float32, device=device),
    )


def _spawn_cinematic_lighting() -> None:
    import isaaclab.sim as sim_utils

    dome = sim_utils.DomeLightCfg(
        intensity=1650.0,
        color=(0.94, 0.91, 0.86),
        visible_in_primary_ray=False,
    )
    dome.func("/World/TwinGR00T_DomeLight", dome)
    for name, intensity, color, position, radius in (
        ("PickupKey", 9500.0, (1.0, 0.82, 0.65), (1.4, 1.2, 2.5), 0.5),
        ("WalkFill", 7800.0, (0.68, 0.82, 1.0), (-0.4, -0.7, 2.4), 0.48),
        ("HandoffKey", 9000.0, (1.0, 0.9, 0.75), (1.0, -2.3, 2.4), 0.5),
    ):
        light = sim_utils.SphereLightCfg(
            intensity=intensity,
            radius=radius,
            color=color,
            normalize=True,
        )
        light.func(f"/World/TwinGR00T_{name}", light, translation=position)


def _frame_image(value: torch.Tensor, size: tuple[int, int]) -> Image.Image:
    frame = value[0].detach().cpu().numpy()
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    image = Image.fromarray(np.ascontiguousarray(frame, dtype=np.uint8))
    image.thumbnail(size, Image.Resampling.LANCZOS)
    panel = Image.new("RGB", size, (9, 12, 18))
    panel.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return panel


def _compose_multiview(obs: dict, step: int, cargo_z: float, assisted: bool) -> Image.Image:
    phase = "GRASP" if step < 350 else "NATURAL WALK" if step < 950 else "HANDOFF"
    main_key = {
        "GRASP": "grasp_camera_rgb",
        "NATURAL WALK": "follow_camera_rgb",
        "HANDOFF": "handoff_camera_rgb",
    }[phase]
    keys = (main_key, "robot_head_cam_rgb", "hand_camera_rgb", "handoff_camera_rgb")
    labels = (f"MAIN · {phase}", "GR00T POV", "HANDS + PLATE", "TRANSFER SURFACE")
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
    held = "lifted" if cargo_z > 0.04 else "surface/contact"
    controller = "GR00T + Homie WBC + grasp constraint" if assisted else "GR00T N1.6 + Homie WBC"
    status = f"{controller} · step {step:04d} · plate {held}"
    draw.rounded_rectangle((480, 1023, 1440, 1070), 10, fill=(0, 0, 0, 190))
    draw.text((520, 1036), status, font=small, fill=(224, 238, 250, 255))
    return canvas


def main() -> None:
    args_parser = get_isaaclab_arena_cli_parser()
    _add_twingroot_args(args_parser)
    args_cli, _ = args_parser.parse_known_args()

    cargo_usd = args_cli.cargo_usd.resolve() if args_cli.cargo_usd is not None else None
    if cargo_usd is not None and not cargo_usd.is_file():
        raise FileNotFoundError(f"Cargo USD does not exist: {cargo_usd}")

    with SimulationAppContext(args_cli):
        _use_local_homie_models()
        _use_simple_home_scene(cargo_usd, tuple(args_cli.cargo_scale))
        args_parser = setup_policy_argument_parser(args_parser)
        args_cli = args_parser.parse_args()
        arena_builder = get_arena_builder_from_cli(args_cli)
        if args_cli.multiview_video is not None or args_cli.recorder_dir is not None:
            env_name, env_cfg = arena_builder.build_registered()
            if args_cli.multiview_video is not None:
                _configure_cinematic_cameras(env_cfg)
            if args_cli.recorder_dir is not None:
                args_cli.recorder_dir.mkdir(parents=True, exist_ok=True)
                env_cfg.recorders.dataset_export_dir_path = str(args_cli.recorder_dir.resolve())
                env_cfg.recorders.dataset_filename = f"dataset_seed_{args_cli.seed}"
            if args_cli.demo_assist:
                env_cfg.terminations.success = None
                env_cfg.terminations.object_dropped = None
            env = gym.make(env_name, cfg=env_cfg).unwrapped
            if args_cli.multiview_video is not None:
                _spawn_cinematic_lighting()
        else:
            env = arena_builder.make_registered()

        if args_cli.seed is not None:
            env.seed(args_cli.seed)
            torch.manual_seed(args_cli.seed)
            np.random.seed(args_cli.seed)
            random.seed(args_cli.seed)

        obs, _ = env.reset()
        assisted_plate_initial_pose = None
        assisted_walk_start_pose = None
        if args_cli.demo_assist:
            assisted_plate_initial_pose = env.scene["brown_box"].data.root_pose_w[0].clone()
            assisted_plate_initial_pose[2] = 0.015
        if args_cli.multiview_video is not None:
            _set_camera_view(env.scene["grasp_camera"], (0.0, 2.5, 1.3), (0.28, 0.12, -0.08))
            _set_camera_view(env.scene["hand_camera"], (1.05, 0.95, 0.58), (0.58, 0.18, 0.2))
            _set_camera_view(env.scene["handoff_camera"], (1.9, -3.0, 1.35), (-0.25, -1.63, -0.1))
            _set_camera_view(env.scene["follow_camera"], (0.2, 2.5, 1.3), (0.0, 0.0, 0.05))
        policy, num_steps = create_policy(args_cli)
        episode_metrics = G1EpisodeMetrics(seed=args_cli.seed, requested_steps=num_steps)

        from isaaclab_arena.metrics.metrics import compute_metrics

        trace_handle = None
        video_process = None
        multiview_process = None
        recorded_frames = 0
        multiview_frames = 0
        try:
            if args_cli.action_trace is not None:
                args_cli.action_trace.parent.mkdir(parents=True, exist_ok=True)
                trace_handle = args_cli.action_trace.open("x", encoding="utf-8")
            if args_cli.head_video is not None:
                sample = obs["camera_obs"]["robot_head_cam_rgb"]
                height, width = sample.shape[-3:-1]
                video_process = _start_video(args_cli.head_video, args_cli.video_fps, width, height)
            if args_cli.multiview_video is not None:
                multiview_process = _start_video(args_cli.multiview_video, args_cli.video_fps, 1920, 1080)

            for step in tqdm.tqdm(range(num_steps)):
                with torch.inference_mode():
                    policy_start = time.perf_counter()
                    actions = policy.get_action(env, obs)
                    policy_latency_ms = (time.perf_counter() - policy_start) * 1000.0
                    navigation_phase = "gr00t"
                    grasp_phase = "physics"
                    if args_cli.demo_assist:
                        navigation_phase = _assist_navigation(actions, step)
                    if args_cli.multiview_video is not None:
                        root = env.scene["robot"].data.root_pos_w[0]
                        _set_camera_view(
                            env.scene["follow_camera"],
                            (float(root[0] + 0.2), float(root[1] + 2.4), float(root[2] + 1.25)),
                            (float(root[0]), float(root[1]), float(root[2] + 0.1)),
                        )
                    obs, reward, terminated, truncated, info = env.step(actions)
                    if args_cli.demo_assist:
                        assert assisted_plate_initial_pose is not None
                        if step == 319:
                            assisted_walk_start_pose = env.scene["robot"].data.root_pose_w[0].clone()
                        if step >= 320:
                            assert assisted_walk_start_pose is not None
                            _assist_robot_translation(env, step, assisted_walk_start_pose)
                        grasp_phase = _assist_plate_pose(env, step, assisted_plate_initial_pose)

                terminated_flag = bool(terminated.any().item())
                truncated_flag = bool(truncated.any().item())
                robot_position = env.scene["robot"].data.root_pos_w[0]
                cargo_position = env.scene["brown_box"].data.root_pos_w[0]
                destination_position = env.scene["blue_sorting_bin"].data.root_pos_w[0]
                episode_metrics.update(
                    step=step,
                    action=actions[0].detach().cpu().numpy(),
                    robot_position=robot_position.detach().cpu().numpy(),
                    cargo_position=cargo_position.detach().cpu().numpy(),
                    destination_position=destination_position.detach().cpu().numpy(),
                    inference_latency_ms=policy_latency_ms,
                    terminated=terminated_flag,
                    truncated=truncated_flag,
                    state_valid=not (terminated_flag and episode_metrics.completed_steps > 0),
                )

                if trace_handle is not None:
                    entry = {
                        "step": step,
                        "action": _tensor_list(actions[0]),
                        "navigate_base_torso": _tensor_list(actions[0, -7:]),
                        "robot_root_pose_w": _tensor_list(env.scene["robot"].data.root_pose_w[0]),
                        "cargo_root_pose_w": _tensor_list(env.scene["brown_box"].data.root_pose_w[0]),
                        "destination_root_pose_w": _tensor_list(env.scene["blue_sorting_bin"].data.root_pose_w[0]),
                        "reward": _tensor_list(reward),
                        "terminated": terminated_flag,
                        "truncated": truncated_flag,
                        "policy_latency_ms": round(policy_latency_ms, 3),
                        "info_keys": sorted(info),
                        "navigation_phase": navigation_phase,
                        "grasp_phase": grasp_phase,
                    }
                    trace_handle.write(json.dumps(entry) + "\n")

                if video_process is not None and step % args_cli.video_stride == 0:
                    frame = obs["camera_obs"]["robot_head_cam_rgb"][0].detach().cpu().numpy()
                    if frame.shape[-1] == 4:
                        frame = frame[..., :3]
                    assert video_process.stdin is not None
                    video_process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
                    recorded_frames += 1

                if multiview_process is not None and step % args_cli.video_stride == 0:
                    cargo_z = float(env.scene["brown_box"].data.root_pos_w[0, 2])
                    frame = _compose_multiview(obs["camera_obs"], step, cargo_z, args_cli.demo_assist)
                    assert multiview_process.stdin is not None
                    multiview_process.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
                    multiview_frames += 1

                if terminated_flag or truncated_flag:
                    if args_cli.stop_on_terminal:
                        break
                    env_ids = (terminated | truncated).nonzero().flatten()
                    policy.reset(env_ids=env_ids)
        finally:
            if trace_handle is not None:
                trace_handle.close()
            if video_process is not None:
                assert video_process.stdin is not None
                video_process.stdin.close()
                stderr = video_process.stderr.read().decode("utf-8", errors="replace")
                return_code = video_process.wait()
                if return_code:
                    raise RuntimeError(f"ffmpeg failed with exit code {return_code}: {stderr}")
                print(f"[TwinGR00T] Recorded {recorded_frames} head-camera frames")
            if multiview_process is not None:
                assert multiview_process.stdin is not None
                multiview_process.stdin.close()
                stderr = multiview_process.stderr.read().decode("utf-8", errors="replace")
                return_code = multiview_process.wait()
                if return_code:
                    raise RuntimeError(f"multiview ffmpeg failed with exit code {return_code}: {stderr}")
                print(f"[TwinGR00T] Recorded {multiview_frames} multiview frames")

        simulator_metrics = _json_safe(compute_metrics(env))
        result = episode_metrics.summary()
        simulator_success = bool(simulator_metrics.get("success_rate", 0.0) > 0.0)
        if simulator_success:
            result["success"] = True
            result["success_step"] = result["terminal_step"]
            result["success_source"] = "simulator_success_termination"
        else:
            result["success_source"] = "geometry"
        result.update(
            {
                "condition": "demo_assist" if args_cli.demo_assist else "unassisted",
                "protocol_valid": not args_cli.demo_assist,
                "policy_config": str(args_cli.policy_config_yaml_path),
                "environment": "galileo_g1_locomanip_pick_and_place",
                "scene_usd": os.environ.get("TWINGR00T_SCENE_USD") or os.environ.get("TWINGR00T_SIMPLE_SCENE_USD"),
                "scene_pose_xyz": os.environ.get("TWINGR00T_SCENE_POSE_XYZ", "0,0,0"),
                "cargo_usd": str(cargo_usd) if cargo_usd is not None else None,
                "recorder_dir": str(args_cli.recorder_dir) if args_cli.recorder_dir else None,
                "simulator_metrics": simulator_metrics,
            }
        )
        if args_cli.result_manifest is not None:
            args_cli.result_manifest.parent.mkdir(parents=True, exist_ok=True)
            args_cli.result_manifest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(f"[TwinGR00T] Result manifest: {args_cli.result_manifest}")
        print(f"Metrics: {result}")
        env.close()


if __name__ == "__main__":
    main()
