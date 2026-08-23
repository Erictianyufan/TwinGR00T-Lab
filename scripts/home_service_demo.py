"""Render a continuous home-service collaboration demo in Isaac Sim."""

from __future__ import annotations

import json
import math
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

from twingr00t_lab.g1_motion import G1MotionPlan, apply_named_offsets
from twingr00t_lab.home_demo import DemoPhase, HomeDemoTimeline
from twingr00t_lab.multi_robot import MultiRobotCoordinator

CANVAS_SIZE = (1920, 1080)
HEADER_HEIGHT = 60
MAIN_SIZE = (1280, 960)
SIDE_SIZE = (640, 360)
CAMERA_KEYS = (
    "third_camera_rgb",
    "front_camera_rgb",
    "head_camera_rgb",
    "left_camera_rgb",
    "right_camera_rgb",
)
ISAAC_PEOPLE_ROOT = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/People/Characters"
)
ANIMGRAPH_ROOT = "https://omniverse-content-production.s3.us-west-2.amazonaws.com/Assets/AnimGraph/Characters"
PEOPLE_ASSETS = (
    f"{ISAAC_PEOPLE_ROOT}/Biped_Setup.usd",
    f"{ANIMGRAPH_ROOT}/Reallusion/Debra/Debra.usd",
    f"{ANIMGRAPH_ROOT}/NVIDIA/biped_demo.usd",
)


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


def compose_frame(camera_obs: dict, frame: int, timeline: HomeDemoTimeline, fps: float) -> Image.Image:
    canvas = Image.new("RGB", CANVAS_SIZE, (6, 9, 14))
    title_font = load_font(27, bold=True)
    label_font = load_font(18, bold=True)
    info_font = load_font(17)

    main = letterbox(to_rgb(camera_obs["third_camera_rgb"]), MAIN_SIZE)
    canvas.paste(main, (0, HEADER_HEIGHT))
    phase = timeline.phase(frame)
    main_label = {
        DemoPhase.ESTABLISH: "KITCHEN · HUMANOID PICKS UP THE BOWL TRAY",
        DemoPhase.TRANSPORT: "HUMANOID · FULL-BODY TRAY TRANSPORT",
        DemoPhase.HANDOFF: "HANDOFF · HUMANOID + ALOHA IN THE SAME FRAME",
        DemoPhase.STACK: "ALOHA · BIMANUAL BOWL STACKING",
        DemoPhase.COMPLETE: "HOME OVERVIEW · COOPERATIVE TASK COMPLETE",
    }[phase]
    if timeline.pickup_camera_active(frame):
        main_label = "G1 HAND CLOSE-UP · ARTICULATED GRASP WITH VISIBLE BOWLS"
    draw_label(canvas, (0, HEADER_HEIGHT), main_label, label_font)

    for panel_index, (key, label) in enumerate(
        (("front_camera_rgb", "ALOHA FRONT"), ("head_camera_rgb", "ALOHA HEAD"))
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

    phase_number = list(DemoPhase).index(phase) + 1
    draw = ImageDraw.Draw(canvas)
    title = "TwinGR00T HomeRelay  |  Continuous multi-robot home service"
    draw.text((22, 13), title, font=title_font, fill=(239, 244, 255))
    status = (
        f"stage {phase_number}/5 · {phase.value}  |  "
        f"t = {frame / fps:05.2f} s  |  {timeline.instruction(frame)}"
    )
    draw.text((1290, 1040), status, font=info_font, fill=(205, 218, 235))

    progress = (frame + 1) / timeline.total_frames
    draw.rectangle((0, 1072, CANVAS_SIZE[0], 1079), fill=(28, 35, 46))
    draw.rectangle((0, 1072, round(CANVAS_SIZE[0] * progress), 1079), fill=(72, 176, 212))
    return canvas


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


def configure_environment_materials(env_cfg: Any) -> None:
    import isaaclab.sim as sim_utils

    env_cfg.scene.robot.spawn.visual_material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.18, 0.38, 0.58), metallic=0.08, roughness=0.32
    )
    env_cfg.scene.robotwin_table.spawn.visual_material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.34, 0.18, 0.07), metallic=0.0, roughness=0.5
    )
    env_cfg.scene.third_camera.spawn.focal_length = 18.0


def configure_g1_asset(env_cfg: Any) -> str:
    """Attach Isaac Lab's official Unitree G1 articulation to the shared scene."""

    from isaaclab_assets.robots.unitree import G1_CFG

    g1_cfg = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/G1Carrier")
    g1_cfg.spawn.rigid_props.disable_gravity = True
    g1_cfg.spawn.rigid_props.linear_damping = 1.0
    g1_cfg.spawn.rigid_props.angular_damping = 1.0
    g1_cfg.init_state.pos = (-3.22, -1.88, 0.74)
    # InteractiveSceneCfg discovers public config attributes at construction.
    env_cfg.scene.g1 = g1_cfg
    return str(g1_cfg.spawn.usd_path)


def spawn_home_scene(
    asset_root: Path, bowl_poses: dict[str, torch.Tensor]
) -> tuple[Any, list[str], list[str]]:
    import isaaclab.sim as sim_utils
    import isaacsim.core.utils.prims as prim_utils
    import omni.usd
    from pxr import Usd, UsdGeom, UsdPhysics

    def material(color: tuple[float, float, float], roughness: float = 0.65):
        return sim_utils.PreviewSurfaceCfg(diffuse_color=color, metallic=0.0, roughness=roughness)

    def box(
        path: str,
        size: tuple[float, float, float],
        position: tuple[float, float, float],
        color: tuple[float, float, float],
        roughness: float = 0.65,
    ) -> None:
        cfg = sim_utils.CuboidCfg(size=size, visual_material=material(color, roughness))
        cfg.func(path, cfg, translation=position)

    def sphere(
        path: str,
        radius: float,
        position: tuple[float, float, float],
        color: tuple[float, float, float],
    ) -> None:
        cfg = sim_utils.SphereCfg(radius=radius, visual_material=material(color, 0.48))
        cfg.func(path, cfg, translation=position)

    def cylinder(
        path: str,
        radius: float,
        height: float,
        position: tuple[float, float, float],
        color: tuple[float, float, float],
    ) -> None:
        cfg = sim_utils.CylinderCfg(
            radius=radius,
            height=height,
            axis="Z",
            visual_material=material(color, 0.32),
        )
        cfg.func(path, cfg, translation=position)

    root = "/World/HomeRelay"
    prim_utils.create_prim(root, "Xform")

    # Architectural shell: warm open-plan kitchen, dining area, and living room.
    box(f"{root}/Floor", (9.0, 7.0, 0.10), (0.0, -0.25, -0.08), (0.24, 0.20, 0.16), 0.72)
    box(f"{root}/BackWall", (9.0, 0.10, 3.2), (0.0, -3.6, 1.5), (0.70, 0.67, 0.60), 0.78)
    box(f"{root}/LeftWall", (0.10, 7.0, 3.2), (-4.45, -0.25, 1.5), (0.55, 0.61, 0.66), 0.78)
    box(f"{root}/RightWall", (0.10, 7.0, 3.2), (4.45, -0.25, 1.5), (0.55, 0.61, 0.66), 0.78)

    # Kitchen cabinetry, worktop, refrigerator, and warm backsplash.
    box(f"{root}/Kitchen/Counter", (4.5, 0.65, 0.85), (-1.7, -3.15, 0.42), (0.20, 0.23, 0.26), 0.4)
    box(f"{root}/Kitchen/Worktop", (4.65, 0.78, 0.08), (-1.7, -3.12, 0.88), (0.76, 0.73, 0.67), 0.34)
    for index, x in enumerate((-3.25, -2.2, -1.15, -0.1)):
        box(f"{root}/Kitchen/UpperCabinet{index}", (0.88, 0.25, 0.72), (x, -3.42, 2.0), (0.84, 0.80, 0.72))
    box(f"{root}/Kitchen/Fridge", (0.9, 0.72, 1.95), (1.05, -3.15, 0.96), (0.52, 0.57, 0.61), 0.25)
    box(f"{root}/Kitchen/Backsplash", (4.7, 0.025, 0.7), (-1.7, -3.48, 1.32), (0.32, 0.46, 0.52), 0.38)

    # Dining table and a small living-room sofa establish the family environment.
    table_center = (2.35, -1.35)
    box(f"{root}/Dining/TableTop", (2.15, 1.25, 0.10), (*table_center, 0.76), (0.42, 0.23, 0.10), 0.42)
    for index, (dx, dy) in enumerate(((-0.85, -0.4), (-0.85, 0.4), (0.85, -0.4), (0.85, 0.4))):
        position = (table_center[0] + dx, table_center[1] + dy, 0.36)
        box(f"{root}/Dining/Leg{index}", (0.10, 0.10, 0.72), position, (0.18, 0.10, 0.05))
    box(f"{root}/Living/SofaBase", (2.15, 0.8, 0.42), (-2.95, 1.55, 0.24), (0.18, 0.38, 0.45), 0.7)
    box(f"{root}/Living/SofaBack", (2.15, 0.22, 0.72), (-2.95, 1.88, 0.68), (0.18, 0.38, 0.45), 0.7)
    box(f"{root}/Living/CoffeeTable", (1.15, 0.7, 0.08), (-1.7, 1.05, 0.48), (0.46, 0.27, 0.12), 0.45)

    # Production Isaac character assets replace the former geometric family proxies.
    # AnimGraph legacy assets use mixed units/axes.  The final two entries are
    # authored in centimeters; Biped_Setup and biped_demo also need +90 deg X.
    character_specs = (
        ("ParentA", PEOPLE_ASSETS[0], (1.42, -2.70, 0.0), 0.78, 0.15, math.pi / 2.0),
        ("ParentB", PEOPLE_ASSETS[1], (3.25, -2.65, 0.0), 0.0092, 2.85, 0.0),
        ("Child", PEOPLE_ASSETS[2], (3.40, -0.40, 0.0), 0.0052, -2.35, math.pi / 2.0),
    )
    loaded_people: list[str] = []
    for name, usd_path, position, scale, yaw, x_rotation in character_specs:
        yaw_cos = math.cos(yaw / 2.0)
        yaw_sin = math.sin(yaw / 2.0)
        x_cos = math.cos(x_rotation / 2.0)
        x_sin = math.sin(x_rotation / 2.0)
        prim_utils.create_prim(
            f"{root}/Family/{name}",
            "Xform",
            translation=position,
            orientation=(
                yaw_cos * x_cos,
                yaw_cos * x_sin,
                yaw_sin * x_sin,
                yaw_sin * x_cos,
            ),
            scale=(scale, scale, scale),
            usd_path=usd_path,
        )
        loaded_people.append(usd_path)

    # Cargo ends at world origin, so its proxies coincide with the recorded physical bowl poses.
    cargo = f"{root}/TransitCargo"
    prim_utils.create_prim(cargo, "Xform")
    xy = np.asarray(
        [pose[0, :2].detach().cpu().numpy() for name, pose in bowl_poses.items() if name.startswith("bowl")]
    )
    center = xy.mean(axis=0)
    extent = np.maximum(xy.max(axis=0) - xy.min(axis=0) + 0.20, np.asarray((0.58, 0.40)))
    tray_size = (float(extent[0]), float(extent[1]), 0.055)
    tray_position = (float(center[0]), float(center[1]), 0.735)
    box(f"{cargo}/Tray", tray_size, tray_position, (0.20, 0.22, 0.24), 0.25)
    grip_y = float(center[1] + extent[1] / 2.0 + 0.018)
    for name, y in (("LeftGrip", grip_y), ("RightGrip", float(center[1] * 2.0 - grip_y))):
        box(
            f"{cargo}/{name}",
            (0.18, 0.055, 0.045),
            (float(center[0]), y, 0.775),
            (0.08, 0.28, 0.38),
            0.28,
        )
    transit_bowls: list[str] = []
    bowl_usd = asset_root / "assets" / "Objects" / "bowl" / "002" / "base2.usd"
    if not bowl_usd.exists():
        raise FileNotFoundError(f"RoboTwin bowl asset is missing: {bowl_usd}")
    for name, pose in bowl_poses.items():
        if not name.startswith("bowl"):
            continue
        xyz = pose[0, :3].detach().cpu().numpy()
        bowl_path = f"{cargo}/{name}"
        prim_utils.create_prim(
            bowl_path,
            "Xform",
            translation=(float(xyz[0]), float(xyz[1]), float(xyz[2])),
            scale=(0.75, 0.75, 0.70),
            usd_path=str(bowl_usd),
        )
        # The source YCB USD contains rigid/collision schemas.  These transit
        # copies are visual payload models under the moving tray Xform, so turn
        # those schemas off; the physical RoboTwin bowls take over at handoff.
        stage = omni.usd.get_context().get_stage()
        for child in Usd.PrimRange(stage.GetPrimAtPath(bowl_path)):
            if child.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(child).CreateRigidBodyEnabledAttr(False)
            if child.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(child).CreateCollisionEnabledAttr(False)
        transit_bowls.append(bowl_path)

    hdr = asset_root / "assets" / "texture" / "base0.exr"
    dome = sim_utils.DomeLightCfg(
        intensity=1900.0,
        color=(0.96, 0.92, 0.86),
        texture_file=str(hdr) if hdr.exists() else None,
        visible_in_primary_ray=False,
    )
    dome.func(f"{root}/DomeLight", dome)
    for name, intensity, color, position, radius in (
        ("KitchenLight", 10000.0, (1.0, 0.82, 0.66), (-1.7, -1.7, 2.7), 0.48),
        ("DiningLight", 9000.0, (1.0, 0.88, 0.72), (2.3, -0.8, 2.6), 0.42),
        ("RobotFill", 7500.0, (0.68, 0.82, 1.0), (-0.7, 0.7, 2.35), 0.42),
    ):
        light = sim_utils.SphereLightCfg(
            intensity=intensity,
            radius=radius,
            color=color,
            normalize=True,
        )
        light.func(f"{root}/{name}", light, translation=position)

    stage = omni.usd.get_context().get_stage()

    def translate_op(path: str):
        xformable = UsdGeom.Xformable(stage.GetPrimAtPath(path))
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                return op
        return xformable.AddTranslateOp()

    return translate_op(cargo), loaded_people, transit_bowls


def command_g1(env: gym.Env, frame: int, timeline: HomeDemoTimeline, motion: G1MotionPlan) -> None:
    """Write one stable full-body pose to the real G1 articulation."""

    g1 = env.scene["g1"]
    sample = motion.sample(frame, timeline)
    default = g1.data.default_joint_pos[0].detach().cpu().numpy()
    target_np = apply_named_offsets(default, tuple(g1.joint_names), sample.joint_offsets)
    targets = torch.as_tensor(target_np, dtype=torch.float32, device=env.device).unsqueeze(0)
    root_pose = torch.as_tensor(
        np.concatenate((sample.root_position, sample.root_quaternion_wxyz)),
        dtype=torch.float32,
        device=env.device,
    ).unsqueeze(0)
    g1.write_root_pose_to_sim(root_pose)
    g1.write_root_velocity_to_sim(torch.zeros((env.num_envs, 6), device=env.device))
    g1.write_joint_position_to_sim(targets)
    g1.write_joint_velocity_to_sim(torch.zeros_like(targets))
    g1.set_joint_position_target(targets)


def prim_world_bounds(path: str) -> dict[str, list[float]]:
    """Return a compact runtime proof that a referenced USD resolved to geometry."""

    import omni.usd
    from pxr import Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"Expected rendered prim is missing: {path}")
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(prim)
    extent = bounds.ComputeAlignedRange()
    minimum = extent.GetMin()
    maximum = extent.GetMax()
    return {
        "min": [float(minimum[index]) for index in range(3)],
        "max": [float(maximum[index]) for index in range(3)],
    }


def initialize_task_state(
    env: gym.Env,
    initial_joint: torch.Tensor,
    rigid_objects: dict[str, torch.Tensor],
) -> None:
    robot = env.scene["robot"]
    robot.write_joint_position_to_sim(initial_joint)
    robot.write_joint_velocity_to_sim(torch.zeros_like(initial_joint))
    robot.set_joint_position_target(initial_joint)

    for name, root_pose in rigid_objects.items():
        if name not in env.scene.rigid_objects:
            continue
        pose = root_pose.clone()
        if name.startswith("bowl"):
            pose[:, 2] = -4.0
        obj = env.scene.rigid_objects[name]
        obj.write_root_pose_to_sim(pose)
        obj.write_root_velocity_to_sim(torch.zeros((env.num_envs, 6), device=env.device))

    for _ in range(35):
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
    for _ in range(3):
        env.sim.render()


def reveal_physical_bowls(env: gym.Env, rigid_objects: dict[str, torch.Tensor]) -> None:
    for name, root_pose in rigid_objects.items():
        if not name.startswith("bowl") or name not in env.scene.rigid_objects:
            continue
        obj = env.scene.rigid_objects[name]
        obj.write_root_pose_to_sim(root_pose)
        obj.write_root_velocity_to_sim(torch.zeros((env.num_envs, 6), device=env.device))
    for _ in range(18):
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
    for _ in range(3):
        env.sim.render()


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


def main() -> None:
    parser = get_isaaclab_arena_cli_parser()
    parser.add_argument("demo", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=31.25)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--groot-checkpoint", type=Path, default=Path("checkpoints/GR00T-N1.7-3B"))
    add_example_environments_cli_args(parser)
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing video: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    timeline = HomeDemoTimeline()
    total_frames = timeline.total_frames if args.max_frames <= 0 else min(args.max_frames, timeline.total_frames)
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    env = None
    process = None
    try:
        from pxr import Gf

        arena_builder = get_arena_builder_from_cli(args)
        env_name, env_cfg = arena_builder.build_registered()
        env_cfg.sim.dt = 1 / 250
        env_cfg.sim.decimation = 5
        env_cfg.wait_for_textures = True
        configure_environment_materials(env_cfg)
        g1_usd_path = configure_g1_asset(env_cfg)
        env = gym.make(env_name, cfg=env_cfg).unwrapped

        actions, initial_joint, rigid_objects = read_demo(args.demo, str(env.device))
        if len(actions) != timeline.stack_frames:
            raise ValueError(f"Timeline expects {timeline.stack_frames} actions, found {len(actions)}")

        asset_root = Path(__file__).resolve().parents[1] / "vendor" / "RoboTwin"
        cargo_translate, people_assets, transit_bowls = spawn_home_scene(asset_root, rigid_objects)
        coordinator = MultiRobotCoordinator()
        g1_motion = G1MotionPlan()
        env.reset()
        env.sim.reset()
        initialize_task_state(env, initial_joint, rigid_objects)
        command_g1(env, 0, timeline, g1_motion)

        overview = env.scene["third_camera"]
        initial_eye, initial_target = timeline.overview_camera_pose(0)
        eyes = torch.as_tensor(initial_eye, device=env.device).unsqueeze(0)
        targets = torch.as_tensor(initial_target, device=env.device).unsqueeze(0)
        overview.set_world_poses_from_view(eyes, targets)
        for _ in range(3):
            env.sim.render()
        g1_bounds = prim_world_bounds("/World/envs/env_0/G1Carrier")
        people_bounds = {
            name: prim_world_bounds(f"/World/HomeRelay/Family/{name}")
            for name in ("ParentA", "ParentB", "Child")
        }
        transit_bowl_bounds = {path: prim_world_bounds(path) for path in transit_bowls}
        print(f"[home-demo] resolved G1 bounds: {g1_bounds}", flush=True)
        print(f"[home-demo] resolved people bounds: {people_bounds}", flush=True)
        print(f"[home-demo] resolved transit bowl bounds: {transit_bowl_bounds}", flush=True)

        process = subprocess.Popen(
            ffmpeg_command(args.output, args.fps), stdin=subprocess.PIPE, stderr=subprocess.PIPE
        )
        assert process.stdin is not None
        bowls_revealed = False
        held_action = actions[0].unsqueeze(0)

        for frame in range(total_frames):
            cargo_position = timeline.cargo_position(frame)
            cargo_translate.Set(Gf.Vec3d(*(float(value) for value in cargo_position)))
            command_g1(env, frame, timeline, g1_motion)

            eye, target = timeline.overview_camera_pose(frame)
            eyes = torch.as_tensor(eye, device=env.device).unsqueeze(0)
            targets = torch.as_tensor(target, device=env.device).unsqueeze(0)
            overview.set_world_poses_from_view(eyes, targets)

            if frame == timeline.stack_start and not bowls_revealed:
                reveal_physical_bowls(env, rigid_objects)
                bowls_revealed = True

            action_index = timeline.aloha_action_index(frame)
            action = held_action if action_index is None else actions[action_index].unsqueeze(0)
            observation, _, _, _, _ = env.step(action)
            missing = [key for key in CAMERA_KEYS if key not in observation["camera_obs"]]
            if missing:
                raise KeyError(f"Live environment is missing cameras: {missing}")
            rendered = compose_frame(observation["camera_obs"], frame, timeline, args.fps)
            process.stdin.write(np.asarray(rendered, dtype=np.uint8).tobytes())
            if frame == 0 or (frame + 1) % 100 == 0 or frame + 1 == total_frames:
                print(f"[home-demo] {frame + 1}/{total_frames} · {timeline.phase(frame).value}", flush=True)

        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        return_code = process.wait()
        if return_code:
            raise RuntimeError(f"ffmpeg failed with exit code {return_code}: {stderr}")

        manifest = {
            "status": "pass",
            "mode": "continuous_home_service_demo",
            "environment": env_name,
            "demo": str(args.demo),
            "output": str(args.output),
            "frames": total_frames,
            "fps": args.fps,
            "duration_seconds": total_frames / args.fps,
            "resolution": list(CANVAS_SIZE),
            "phases": [phase.value for phase in DemoPhase],
            "cameras": list(CAMERA_KEYS),
            "humanoid_asset": g1_usd_path,
            "humanoid_world_bounds": g1_bounds,
            "humanoid_controller": "real_g1_articulation_named_joint_gait_v1",
            "g1_policy_route": coordinator.g1.__dict__,
            "aloha_policy_route": coordinator.aloha.__dict__,
            "groot_checkpoint": str(args.groot_checkpoint),
            "groot_checkpoint_complete": all(
                (args.groot_checkpoint / name).exists()
                for name in ("config.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
            ),
            "groot_runtime": "checkpoint_ready; gated Cosmos-Reason2-2B authentication required for live inference",
            "g1_execution_mode": "REAL_G1 policy contract with deterministic gait adapter",
            "aloha_controller": "recorded_isaac_actions",
            "family_members": len(people_assets),
            "people_assets": people_assets,
            "people_world_bounds": people_bounds,
            "transit_bowl_asset": str(asset_root / "assets" / "Objects" / "bowl" / "002" / "base2.usd"),
            "transit_bowl_count": len(transit_bowls),
            "transit_bowl_world_bounds": transit_bowl_bounds,
            "g1_hand_motion": "articulated zero-to-six finger joints with mirrored bilateral closure",
        }
        args.output.with_suffix(".json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(manifest, indent=2), flush=True)
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
