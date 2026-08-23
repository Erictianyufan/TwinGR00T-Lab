"""Record a warehouse parcel demo using only native PhysX rigid-body grasps.

Unlike ``parcel_sorting_demo.py``, this renderer never animates parcel world
poses.  The three source task rigid bodies remain dynamic after initialization;
ALOHA transports them through finger collision, friction, gravity, and the
recorded gripper commands.
"""

from __future__ import annotations

import json
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
    interpolate_actions,
    load_font,
    read_demo,
    teleport,
)

from twingr00t_lab.physics_grasp import evaluate_physics_grasps

ISAAC_ASSET_ROOT = Path("/home/ubuntu/readonly/Assets/Isaac/5.1/Isaac")
WAREHOUSE_ASSET = (
    ISAAC_ASSET_ROOT / "Environments/Simple_Warehouse/warehouse_multiple_shelves.usd"
)
CONVEYOR_ASSET = ISAAC_ASSET_ROOT / "Props/Conveyors/ConveyorBelt_A01.usd"
PALLET_ASSET = ISAAC_ASSET_ROOT / "Props/Pallet/pallet.usd"
BOX_ASSET = ISAAC_ASSET_ROOT / "Environments/Simple_Warehouse/Props/SM_CardBoxA_01.usd"
PARCEL_NAMES = ("block1", "block2", "block3")
CONTACT_SENSOR_NAMES = tuple(f"{name}_gripper_contacts" for name in PARCEL_NAMES)


def configure_physical_parcels(env_cfg: Any) -> None:
    """Keep RoboTwin dynamics unchanged and only restyle/observe the parcels."""
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import ContactSensorCfg

    colors = (
        (0.82, 0.47, 0.12),
        (0.18, 0.52, 0.78),
        (0.78, 0.68, 0.12),
    )
    for name, color, sensor_name in zip(
        PARCEL_NAMES, colors, CONTACT_SENSOR_NAMES, strict=True
    ):
        cfg = getattr(env_cfg.scene, name)
        cfg.spawn.visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=color,
            metallic=0.0,
            roughness=0.68,
        )
        cfg.spawn.activate_contact_sensors = True
        setattr(
            env_cfg.scene,
            sensor_name,
            ContactSensorCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{name}",
                update_period=0.0,
                history_length=3,
                track_pose=True,
                track_contact_points=True,
                max_contact_data_count_per_prim=8,
                filter_prim_paths_expr=[
                    "{ENV_REGEX_NS}/Robot/fl_link7",
                    "{ENV_REGEX_NS}/Robot/fl_link8",
                    "{ENV_REGEX_NS}/Robot/fr_link7",
                    "{ENV_REGEX_NS}/Robot/fr_link8",
                ],
            ),
        )
    env_cfg.scene.robot.spawn.activate_contact_sensors = True


def spawn_warehouse_context() -> dict[str, str]:
    """Build a visible, collision-free logistics set around the workcell."""
    import isaaclab.sim as sim_utils
    import isaacsim.core.utils.prims as prim_utils
    import omni.usd
    from pxr import Usd, UsdPhysics

    for path in (WAREHOUSE_ASSET, CONVEYOR_ASSET, PALLET_ASSET, BOX_ASSET):
        if not path.exists():
            raise FileNotFoundError(f"Required Isaac asset is missing: {path}")

    stage = omni.usd.get_context().get_stage()

    def disable_physics(prim_path: str) -> None:
        """Keep set dressing visible without letting it disturb task physics."""
        root_prim = stage.GetPrimAtPath(prim_path)
        for prim in Usd.PrimRange(root_prim):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)

    root = "/World/PhysicalParcelSorting"
    prim_utils.create_prim(root, "Xform")
    prim_utils.create_prim(
        f"{root}/Warehouse",
        "Xform",
        # The authored warehouse spans roughly y=[-23.5, 26.5].  Move its open
        # loading-bay edge to the workcell instead of enclosing the cameras in
        # a wall at the asset origin.
        translation=(0.0, -26.0, 0.0),
        scale=(1.0, 1.0, 1.0),
        usd_path=str(WAREHOUSE_ASSET),
    )
    disable_physics(f"{root}/Warehouse")
    prim_utils.create_prim(
        f"{root}/Conveyor",
        "Xform",
        translation=(-1.05, -0.95, 0.0),
        scale=(0.22, 0.22, 0.22),
        usd_path=str(CONVEYOR_ASSET),
    )
    disable_physics(f"{root}/Conveyor")

    prim_utils.create_prim(
        f"{root}/Pallet",
        "Xform",
        translation=(-1.05, -0.92, 0.0),
        scale=(0.65, 0.65, 0.65),
        usd_path=str(PALLET_ASSET),
    )
    disable_physics(f"{root}/Pallet")
    for index, (position, scale) in enumerate(
        (
            ((-1.18, -0.94, 0.10), 0.35),
            ((-0.91, -0.94, 0.10), 0.35),
            ((-1.05, -0.94, 0.29), 0.35),
            ((-0.82, -2.02, 0.50), 0.42),
            ((-0.28, -2.02, 1.08), 0.42),
            ((0.30, -2.02, 1.08), 0.42),
            ((0.80, -2.02, 1.66), 0.42),
            ((0.72, -0.62, 0.67), 0.28),
            ((1.28, -0.62, 0.67), 0.28),
        )
    ):
        path = f"{root}/BackgroundBoxes/Box{index + 1}"
        prim_utils.create_prim(
            path,
            "Xform",
            translation=position,
            scale=(scale, scale, scale),
            usd_path=str(BOX_ASSET),
        )
        disable_physics(path)

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

    # A camera-safe industrial wall and storage rack make the logistics context
    # legible even in the tight RoboTwin camera frusta.  The cartons and pallet
    # remain official Isaac assets; these beams are non-colliding set dressing.
    panel(f"{root}/LogisticsWall", (4.8, 0.06, 2.8), (0.0, -2.25, 1.38), (0.10, 0.14, 0.18))
    for index, x in enumerate((-1.20, -0.42, 0.42, 1.20)):
        panel(
            f"{root}/Rack/Upright{index + 1}",
            (0.055, 0.20, 2.15),
            (x, -2.12, 1.08),
            (0.94, 0.57, 0.05),
            0.38,
        )
    for index, z in enumerate((0.38, 0.96, 1.54, 2.12)):
        panel(
            f"{root}/Rack/Shelf{index + 1}",
            (2.48, 0.42, 0.055),
            (0.0, -2.08, z),
            (0.22, 0.27, 0.31),
            0.44,
        )
        panel(
            f"{root}/Rack/Beam{index + 1}",
            (2.52, 0.045, 0.10),
            (0.0, -1.85, z),
            (0.95, 0.55, 0.04),
            0.36,
        )

    # Compact camera-facing feed conveyor.  It complements the official Isaac
    # conveyor USD with a silhouette that stays legible in RoboTwin's tight
    # overall camera while remaining entirely outside the manipulation physics.
    panel(f"{root}/FeedConveyor/Belt", (1.34, 0.42, 0.075), (1.00, -0.62, 0.62), (0.035, 0.045, 0.055), 0.30)
    for index, y in enumerate((-0.39, -0.85)):
        panel(
            f"{root}/FeedConveyor/Rail{index + 1}",
            (1.42, 0.045, 0.13),
            (1.00, y, 0.70),
            (0.95, 0.55, 0.04),
            0.34,
        )
    for index, (x, y) in enumerate(
        ((0.45, -0.43), (0.45, -0.81), (1.55, -0.43), (1.55, -0.81))
    ):
        panel(
            f"{root}/FeedConveyor/Leg{index + 1}",
            (0.07, 0.07, 0.58),
            (x, y, 0.29),
            (0.23, 0.27, 0.31),
            0.42,
        )
    for index, x in enumerate((0.52, 0.76, 1.00, 1.24, 1.48)):
        panel(
            f"{root}/FeedConveyor/Marker{index + 1}",
            (0.10, 0.30, 0.004),
            (x, -0.62, 0.660),
            (0.48, 0.52, 0.56),
            0.24,
        )

    # Thin non-colliding destination markers sit on the existing RoboTwin
    # table.  They do not alter parcel physics or support the objects.
    for index, (x, color) in enumerate(
        ((-0.085, (0.12, 0.38, 0.68)), (0.0, (0.12, 0.55, 0.32)), (0.085, (0.86, 0.58, 0.08)))
    ):
        panel(
            f"{root}/SortZones/Zone{index + 1}",
            (0.07, 0.09, 0.002),
            (x, -0.19, 0.746),
            color,
            0.46,
        )

    for name, intensity, color, position in (
        ("HighBay", 18000.0, (0.86, 0.93, 1.0), (1.2, 1.3, 3.1)),
        ("Workcell", 14000.0, (1.0, 0.82, 0.65), (-1.1, 0.1, 2.4)),
        ("Rim", 9000.0, (0.66, 0.82, 1.0), (0.2, -2.0, 2.6)),
    ):
        light = sim_utils.SphereLightCfg(
            intensity=intensity,
            radius=0.5,
            color=color,
            normalize=True,
        )
        light.func(f"{root}/Lights/{name}", light, translation=position)

    return {
        "warehouse": str(WAREHOUSE_ASSET),
        "conveyor": str(CONVEYOR_ASSET),
        "pallet": str(PALLET_ASSET),
        "background_boxes": str(BOX_ASSET),
    }


def _xyz(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    value = np.asarray(value, dtype=float)
    return value[0, :3] if value.ndim > 1 else value[:3]


def end_effector_positions(env: gym.Env, body_ids: list[int]) -> np.ndarray:
    """Read true world poses from articulation bodies, bypassing task observations."""
    robot = env.scene["robot"]
    return robot.data.body_pos_w[0, body_ids].detach().cpu().numpy()


def end_effector_orientations(env: gym.Env, body_ids: list[int]) -> np.ndarray:
    """Read scalar-first world quaternions from the true wrist bodies."""
    robot = env.scene["robot"]
    return robot.data.body_quat_w[0, body_ids].detach().cpu().numpy()


def filtered_hand_forces(env: gym.Env) -> np.ndarray:
    """Return filtered normal force magnitudes as (parcel, left/right hand)."""
    result = np.zeros((len(PARCEL_NAMES), 2), dtype=float)
    for parcel_index, sensor_name in enumerate(CONTACT_SENSOR_NAMES):
        matrix = env.scene[sensor_name].data.force_matrix_w
        if matrix is None:
            continue
        values = matrix[0].detach().cpu().numpy()
        # One parcel body, four filters: left fingers then right fingers.
        magnitudes = np.linalg.norm(values[0], axis=-1)
        result[parcel_index, 0] = float(np.sum(magnitudes[:2]))
        result[parcel_index, 1] = float(np.sum(magnitudes[2:4]))
    return result


def physics_scene_contract(env: gym.Env) -> dict[str, Any]:
    import omni.usd
    from pxr import Usd, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    rigid = []
    collision = []
    for name in PARCEL_NAMES:
        prim = stage.GetPrimAtPath(f"/World/envs/env_0/{name}")
        rigid.append(bool(prim and prim.HasAPI(UsdPhysics.RigidBodyAPI)))
        collision.append(
            bool(
                prim
                and any(
                    descendant.HasAPI(UsdPhysics.CollisionAPI)
                    for descendant in Usd.PrimRange(prim)
                )
            )
        )
    return {
        "parcel_rigid_body_api": rigid,
        "parcel_collision_api": collision,
        "post_initialization_world_pose_writes": 0,
        "pass": all(rigid) and all(collision),
    }


def decorate_frame(image: Any, forces: np.ndarray, index: int) -> Any:
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle((0, 0, image.width, 58), fill=(3, 10, 17, 255))
    draw.text(
        (22, 13),
        "TwinGR00T Lab  |  Isaac Sim warehouse logistics  |  Contact-physics parcel routing",
        font=load_font(21, bold=True),
        fill=(239, 244, 255, 255),
    )
    draw.rounded_rectangle((945, 72, 1254, 117), 8, fill=(7, 14, 21, 210))
    draw.text(
        (968, 82),
        "CONVEYOR FEED CELL",
        font=load_font(17, bold=True),
        fill=(247, 177, 57, 255),
    )
    maximum = float(np.max(forces))
    active = int(np.count_nonzero(forces >= 0.05))
    draw.rounded_rectangle((24, 925, 750, 1011), 10, fill=(3, 10, 17, 215))
    draw.text(
        (42, 940),
        "PHYSX RIGID-BODY GRASP · NO POSE ANIMATION",
        font=load_font(21, bold=True),
        fill=(111, 226, 165, 255),
    )
    draw.text(
        (42, 975),
        f"frame {index + 1:03d} · filtered finger contacts {active} · peak {maximum:5.2f} N",
        font=load_font(17),
        fill=(224, 232, 241, 255),
    )
    return image


def main() -> None:
    parser = get_isaaclab_arena_cli_parser()
    parser.add_argument("demo", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--interpolation-factor", type=int, default=2)
    parser.add_argument("--pre-roll-frames", type=int, default=25)
    parser.add_argument("--post-roll-frames", type=int, default=50)
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
        env_cfg.wait_for_textures = True
        # The source ranking task terminates as soon as the third parcel enters
        # the ordered region.  ManagerBasedRLEnv would then auto-reset the arm
        # while its fingers are still touching the parcel, creating a one-frame
        # wrist jump and physically launching the object.  A presentation replay
        # must run through the recorded release and settling tail instead.
        success_termination_disabled = False
        if hasattr(env_cfg.terminations, "success"):
            env_cfg.terminations.success = None
            success_termination_disabled = True
        configure_showcase_materials(env_cfg)
        configure_physical_parcels(env_cfg)
        env = gym.make(env_name, cfg=env_cfg).unwrapped
        assets = spawn_warehouse_context()
        env.reset()
        env.sim.reset()

        actions, initial_joint, rigid_objects = read_demo(args.demo, str(env.device))
        if args.max_frames > 0:
            actions = actions[: args.max_frames]
        source_frames = len(actions)
        actions = interpolate_actions(actions, args.interpolation_factor)
        observation = teleport(env, initial_joint, rigid_objects)
        replay_actions = torch.cat(
            (
                actions[:1].repeat(max(args.pre_roll_frames, 0), 1),
                actions,
                actions[-1:].repeat(max(args.post_roll_frames, 0), 1),
            ),
            dim=0,
        )

        TASK_LABELS["parcel_sorting_physics"] = "Contact-physics parcel routing"
        process = subprocess.Popen(
            ffmpeg_command(args.output, args.fps), stdin=subprocess.PIPE, stderr=subprocess.PIPE
        )
        assert process.stdin is not None

        parcel_assets = [env.scene.rigid_objects[name] for name in PARCEL_NAMES]
        eef_body_ids, eef_body_names = env.scene["robot"].find_bodies(
            ["fl_link6", "fr_link6"], preserve_order=True
        )
        if eef_body_names != ["fl_link6", "fr_link6"]:
            raise RuntimeError(f"Unexpected ALOHA end-effector bodies: {eef_body_names}")
        positions: list[np.ndarray] = []
        linear_velocities: list[np.ndarray] = []
        angular_velocities: list[np.ndarray] = []
        eef_positions: list[np.ndarray] = []
        eef_orientations: list[np.ndarray] = []
        contact_forces: list[np.ndarray] = []

        for index, action in enumerate(replay_actions):
            observation, _, _, _, _ = env.step(action.unsqueeze(0))
            frame_positions = np.stack(
                [asset.data.root_pos_w[0].detach().cpu().numpy() for asset in parcel_assets]
            )
            frame_linear = np.stack(
                [asset.data.root_lin_vel_w[0].detach().cpu().numpy() for asset in parcel_assets]
            )
            frame_angular = np.stack(
                [asset.data.root_ang_vel_w[0].detach().cpu().numpy() for asset in parcel_assets]
            )
            forces = filtered_hand_forces(env)
            positions.append(frame_positions)
            linear_velocities.append(frame_linear)
            angular_velocities.append(frame_angular)
            eef_positions.append(end_effector_positions(env, eef_body_ids))
            eef_orientations.append(end_effector_orientations(env, eef_body_ids))
            contact_forces.append(forces)

            camera_obs = observation.get("camera_obs")
            if camera_obs is None:
                camera_obs = env.observation_manager.compute(update_history=False)["camera_obs"]
            rendered = compose_frame(
                camera_obs,
                index,
                len(replay_actions),
                args.fps,
                "parcel_sorting_physics",
                "Contact-driven pickup, transport, and stable release",
            )
            rendered = decorate_frame(rendered, forces, index)
            process.stdin.write(np.asarray(rendered, dtype=np.uint8).tobytes())
            if index == 0 or (index + 1) % 100 == 0 or index + 1 == len(replay_actions):
                print(
                    f"[physics-parcel] {index + 1}/{len(replay_actions)} · peak contact {np.max(forces):.3f} N",
                    flush=True,
                )

        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        return_code = process.wait()
        if return_code:
            raise RuntimeError(f"ffmpeg failed with exit code {return_code}: {stderr}")

        trace_path = args.output.with_suffix(".trace.npz")
        np.savez_compressed(
            trace_path,
            object_positions=np.asarray(positions),
            object_linear_velocities=np.asarray(linear_velocities),
            object_angular_velocities=np.asarray(angular_velocities),
            eef_positions=np.asarray(eef_positions),
            eef_orientations=np.asarray(eef_orientations),
            hand_contact_forces=np.asarray(contact_forces),
        )
        scene_contract = physics_scene_contract(env)
        grasp_metrics = evaluate_physics_grasps(
            np.asarray(positions),
            np.asarray(linear_velocities),
            np.asarray(angular_velocities),
            np.asarray(eef_positions),
            np.asarray(contact_forces),
            eef_orientations=np.asarray(eef_orientations),
            settle_frames=min(max(args.post_roll_frames // 2, 10), len(replay_actions)),
        )
        status = "pass" if scene_contract["pass"] and grasp_metrics["pass"] else "fail"
        manifest = {
            "status": status,
            "mode": "native_physx_contact_grasp_parcel_routing",
            "environment": env_name,
            "demo": str(args.demo),
            "output": str(args.output),
            "physics_trace": str(trace_path),
            "frames": len(replay_actions),
            "source_frames": source_frames,
            "fps": args.fps,
            "duration_seconds": len(replay_actions) / args.fps,
            "cameras": list(CAMERA_KEYS),
            "warehouse_asset": assets["warehouse"],
            "conveyor_asset": assets["conveyor"],
            "logistics_set_assets": assets,
            "set_dressing_physics_disabled": True,
            "robot": "ALOHA AgileX",
            "controller": "recorded successful RoboTwin joint and gripper targets",
            "parcel_dynamics": "PhysX dynamic rigid bodies with collision, friction, and gravity",
            "grasp_mechanism": "native finger contact and friction; no fixed joint",
            "uses_scripted_parcel_world_poses": False,
            "claims_policy_inference": False,
            "success_auto_reset_disabled": success_termination_disabled,
            "physics_scene_contract": scene_contract,
            "physics_grasp_validation": grasp_metrics,
        }
        args.output.with_suffix(".json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(manifest, indent=2), flush=True)
        if status != "pass":
            raise RuntimeError("Physical grasp acceptance failed; inspect the adjacent manifest")
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
