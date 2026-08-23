"""Render a multi-camera ALOHA parcel-sorting showcase in an Isaac warehouse.

The robot follows a successful RoboTwin bimanual action recording.  Parcel
transfers are deterministic visual choreography for the presentation demo; the
manifest deliberately does not claim policy-controlled grasp success or a
surface-velocity conveyor simulation.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import traceback
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from isaaclab.app import AppLauncher
from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.examples.example_environments.cli import (
    add_example_environments_cli_args,
    get_arena_builder_from_cli,
)
from PIL import ImageDraw
from replay_live_multiview import (
    CAMERA_KEYS,
    TASK_LABELS,
    compose_frame,
    configure_showcase_materials,
    ffmpeg_command,
    load_font,
    read_demo,
    teleport,
)

from twingr00t_lab.parcel_sorting import ParcelPhase, ParcelSortingTimeline, interpolate

ISAAC_ASSET_ROOT = Path("/home/ubuntu/readonly/Assets/Isaac/5.1/Isaac")
WAREHOUSE_ASSET = ISAAC_ASSET_ROOT / "Environments/Digital_Twin_Warehouse/small_warehouse_digital_twin.usd"
CONVEYOR_ASSET = ISAAC_ASSET_ROOT / "Props/Conveyors/ConveyorBelt_A01.usd"


def _as_xyz(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=float)
    if result.ndim > 1:
        result = result[0]
    return result[:3]


def _eef_positions(observation: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    policy = observation.get("policy", {})
    left = policy.get("left_eef_pos")
    right = policy.get("right_eef_pos")
    if left is None or right is None:
        raise KeyError("The selected environment must expose left_eef_pos and right_eef_pos")
    return _as_xyz(left), _as_xyz(right)


def spawn_warehouse_workcell() -> tuple[list[Any], dict[str, str]]:
    """Reference production Isaac assets and add bins plus readable parcel props."""
    import isaaclab.sim as sim_utils
    import isaacsim.core.utils.prims as prim_utils
    import omni.usd
    from pxr import UsdGeom

    for path in (WAREHOUSE_ASSET, CONVEYOR_ASSET):
        if not path.exists():
            raise FileNotFoundError(f"Required Isaac asset is missing: {path}")

    root = "/World/ParcelSorting"
    prim_utils.create_prim(root, "Xform")
    prim_utils.create_prim(
        f"{root}/Warehouse",
        "Xform",
        translation=(-3.5, -2.0, 0.0),
        scale=(0.72, 0.72, 0.72),
        usd_path=str(WAREHOUSE_ASSET),
    )
    yaw = math.pi / 2.0
    prim_utils.create_prim(
        f"{root}/Conveyor/OfficialAsset",
        "Xform",
        translation=(-0.82, 0.34, 0.25),
        orientation=(math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)),
        scale=(0.85, 0.85, 0.85),
        usd_path=str(CONVEYOR_ASSET),
    )

    def material(color: tuple[float, float, float], roughness: float = 0.55):
        return sim_utils.PreviewSurfaceCfg(
            diffuse_color=color, metallic=0.0, roughness=roughness
        )

    def box(
        path: str,
        size: tuple[float, float, float],
        position: tuple[float, float, float],
        color: tuple[float, float, float],
        roughness: float = 0.55,
    ) -> None:
        cfg = sim_utils.CuboidCfg(size=size, visual_material=material(color, roughness))
        cfg.func(path, cfg, translation=position)

    # A dark belt and safety rails remain visible if the referenced asset's
    # authored origin differs between Isaac asset releases.
    box(f"{root}/Conveyor/Belt", (1.75, 0.42, 0.07), (-0.65, 0.38, 0.77), (0.055, 0.065, 0.075), 0.32)
    box(f"{root}/Conveyor/LeftRail", (1.75, 0.035, 0.16), (-0.65, 0.17, 0.84), (0.50, 0.54, 0.58), 0.25)
    box(f"{root}/Conveyor/RightRail", (1.75, 0.035, 0.16), (-0.65, 0.59, 0.84), (0.50, 0.54, 0.58), 0.25)
    for index, x in enumerate(np.linspace(-1.38, 0.08, 7)):
        box(f"{root}/Conveyor/Leg{index}", (0.055, 0.34, 0.72), (float(x), 0.38, 0.37), (0.26, 0.28, 0.30), 0.4)

    # Destination bins sit inside both arm workspaces.
    for name, x, color in (
        ("Priority", -0.56, (0.10, 0.32, 0.62)),
        ("Standard", 0.56, (0.14, 0.52, 0.34)),
    ):
        box(f"{root}/Bins/{name}/Base", (0.42, 0.35, 0.055), (x, -0.03, 0.795), color)
        box(f"{root}/Bins/{name}/Back", (0.42, 0.035, 0.20), (x, 0.13, 0.88), color)
        box(f"{root}/Bins/{name}/SideL", (0.035, 0.35, 0.20), (x - 0.195, -0.03, 0.88), color)
        box(f"{root}/Bins/{name}/SideR", (0.035, 0.35, 0.20), (x + 0.195, -0.03, 0.88), color)

    package_specs = (
        ("Express", (0.18, 0.14, 0.12), (0.92, 0.46, 0.10)),
        ("Standard", (0.20, 0.13, 0.10), (0.20, 0.55, 0.88)),
        ("Transfer", (0.16, 0.16, 0.13), (0.78, 0.68, 0.12)),
    )
    package_ops: list[Any] = []
    stage = omni.usd.get_context().get_stage()
    for index, (name, size, color) in enumerate(package_specs):
        path = f"{root}/Packages/{name}"
        position = (-1.45 - index * 0.24, 0.38, 0.88)
        box(path, size, position, color, 0.62)
        box(f"{path}/Tape", (size[0] * 1.02, 0.024, size[2] * 1.02), position, (0.93, 0.91, 0.82), 0.5)
        xformable = UsdGeom.Xformable(stage.GetPrimAtPath(path))
        translate = next(
            (op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate),
            None,
        )
        if translate is None:
            translate = xformable.AddTranslateOp()
        package_ops.append(translate)

    for name, intensity, color, position in (
        ("HighBayKey", 13000.0, (0.88, 0.94, 1.0), (1.2, 1.4, 3.2)),
        ("WorkcellFill", 9000.0, (1.0, 0.80, 0.62), (-1.4, -0.1, 2.5)),
    ):
        light = sim_utils.SphereLightCfg(
            intensity=intensity, radius=0.55, color=color, normalize=True
        )
        light.func(f"{root}/Lighting/{name}", light, translation=position)

    return package_ops, {
        "warehouse": str(WAREHOUSE_ASSET),
        "conveyor": str(CONVEYOR_ASSET),
    }


def hide_source_task_objects(env: gym.Env) -> None:
    """Keep source physics but remove colored task props from the presentation."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    for name in env.scene.rigid_objects:
        if name.startswith(("block", "bowl")):
            prim = stage.GetPrimAtPath(f"/World/envs/env_0/{name}")
            if prim.IsValid():
                UsdGeom.Imageable(prim).MakeInvisible()


def package_positions(
    frame: int,
    timeline: ParcelSortingTimeline,
    left_eef: np.ndarray,
    right_eef: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Continuous conveyor, pickup, handoff, and bin-placement paths."""
    belt_start = np.asarray((-1.48, 0.38, 0.88))
    pickup = np.asarray((-0.18, 0.38, 0.88))
    left_bin = np.asarray((-0.56, -0.03, 0.93))
    right_bin = np.asarray((0.56, -0.03, 0.93))
    left_grip = left_eef + np.asarray((0.0, 0.0, -0.055))
    right_grip = right_eef + np.asarray((0.0, 0.0, -0.055))

    def belt(offset: int) -> np.ndarray:
        fraction = (frame - timeline.feed_start - offset) / 92.0
        return interpolate(belt_start, pickup, fraction)

    first = belt(0)
    if frame >= 150:
        first = interpolate(pickup, left_grip, (frame - 150) / 18.0)
    if frame >= 168:
        first = left_grip
    if frame >= 226:
        first = interpolate(left_grip, left_bin, (frame - 226) / 28.0)
    if frame >= 254:
        first = left_bin

    second = belt(34)
    if frame >= 180:
        second = interpolate(pickup, right_grip, (frame - 180) / 18.0)
    if frame >= 198:
        second = right_grip
    if frame >= 258:
        second = interpolate(right_grip, right_bin, (frame - 258) / 28.0)
    if frame >= 286:
        second = right_bin

    third = belt(68)
    if frame >= 252:
        third = interpolate(pickup, left_grip, (frame - 252) / 18.0)
    if frame >= 270:
        third = left_grip
    if frame >= 305:
        third = interpolate(left_grip, right_grip, (frame - 305) / 24.0)
    if frame >= 329:
        third = right_grip
    transfer_bin = right_bin + np.asarray((0.0, 0.0, 0.14))
    if frame >= 340:
        third = interpolate(right_grip, transfer_bin, (frame - 340) / 20.0)
    if frame >= 360:
        third = transfer_bin
    return first, second, third


def decorate_frame(image: Any, phase: ParcelPhase, index: int) -> Any:
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle((0, 0, 1280, 60), fill=(5, 12, 20, 235))
    draw.text(
        (22, 13),
        "TwinGR00T Lab  |  Isaac digital-twin warehouse  |  Bimanual parcel sorting",
        font=load_font(28, bold=True),
        fill=(239, 244, 255, 255),
    )
    draw.rounded_rectangle((24, 928, 690, 1004), 10, fill=(4, 10, 18, 205))
    draw.text(
        (42, 944),
        f"PHASE {list(ParcelPhase).index(phase) + 1}/5 · {phase.value.upper()}",
        font=load_font(22, bold=True),
        fill=(116, 216, 255, 255),
    )
    draw.text(
        (42, 974),
        f"frame {index + 1:03d} · scripted showcase choreography",
        font=load_font(17),
        fill=(220, 228, 238, 255),
    )
    return image


def main() -> None:
    parser = get_isaaclab_arena_cli_parser()
    parser.add_argument("demo", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-frames", type=int, default=0)
    add_example_environments_cli_args(parser)
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing video: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    timeline = ParcelSortingTimeline()
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
        configure_showcase_materials(env_cfg)
        env = gym.make(env_name, cfg=env_cfg).unwrapped
        actions, initial_joint, rigid_objects = read_demo(args.demo, str(env.device))
        package_ops, assets = spawn_warehouse_workcell()
        env.reset()
        env.sim.reset()
        observation = teleport(env, initial_joint, rigid_objects)
        hide_source_task_objects(env)

        overview = env.scene["third_camera"]
        eyes = torch.as_tensor((2.55, 2.15, 2.15), device=env.device).unsqueeze(0)
        targets = torch.as_tensor((-0.25, 0.0, 0.82), device=env.device).unsqueeze(0)
        overview.set_world_poses_from_view(eyes, targets)
        for _ in range(3):
            env.sim.render()

        TASK_LABELS["parcel_sorting"] = "Bimanual parcel sorting"
        process = subprocess.Popen(
            ffmpeg_command(args.output, args.fps), stdin=subprocess.PIPE, stderr=subprocess.PIPE
        )
        assert process.stdin is not None

        for frame in range(total_frames):
            action = actions[timeline.action_index(frame, len(actions))].unsqueeze(0)
            observation, _, _, _, _ = env.step(action)
            left_eef, right_eef = _eef_positions(observation)
            positions = package_positions(frame, timeline, left_eef, right_eef)
            for op, position in zip(package_ops, positions, strict=True):
                op.Set(Gf.Vec3d(*(float(value) for value in position)))
            env.sim.render()
            camera_obs = env.observation_manager.compute(update_history=False)["camera_obs"]
            missing = [key for key in CAMERA_KEYS if key not in camera_obs]
            if missing:
                raise KeyError(f"Live environment is missing cameras: {missing}")
            phase = timeline.phase(frame)
            rendered = compose_frame(
                camera_obs,
                frame,
                total_frames,
                args.fps,
                "parcel_sorting",
                phase.value,
            )
            rendered = decorate_frame(rendered, phase, frame)
            process.stdin.write(np.asarray(rendered, dtype=np.uint8).tobytes())
            if frame == 0 or (frame + 1) % 60 == 0 or frame + 1 == total_frames:
                print(f"[parcel-demo] {frame + 1}/{total_frames} · {phase.value}", flush=True)

        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        return_code = process.wait()
        if return_code:
            raise RuntimeError(f"ffmpeg failed with exit code {return_code}: {stderr}")

        manifest = {
            "status": "pass",
            "mode": "isaac_warehouse_bimanual_parcel_sorting_showcase",
            "environment": env_name,
            "output": str(args.output),
            "frames": total_frames,
            "fps": args.fps,
            "duration_seconds": total_frames / args.fps,
            "phases": [phase.value for phase in ParcelPhase],
            "cameras": list(CAMERA_KEYS),
            "warehouse_asset": assets["warehouse"],
            "conveyor_asset": assets["conveyor"],
            "robot_controller": "recorded_successful_RoboTwin_bimanual_actions",
            "parcel_motion": "deterministic_kinematic_showcase_choreography",
            "claims_policy_inference": False,
            "claims_surface_velocity_conveyor_physics": False,
            "parcel_count": 3,
            "destination_bins": 2,
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
