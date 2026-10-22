# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to Replay and Record RoboTwin Demonstrations in Isaac Lab.

This script loads pre-recorded human teleoperation data (from RoboTwin), replays it 
within the Isaac Lab simulation environment, and re-records the data into HDF5 datasets
and video visualizations. 

This version includes checkpointing: it tracks processed episodes in 'processed_episodes.txt'
to allow resuming from where the previous session left off.

Usage:
    python record_demos_robotwin.py --task <TASK_NAME> --robotwin_data_root <PATH> --output <OUTPUT_DIR>
"""

import argparse
import contextlib
import datetime
import json
import os
import time
from collections.abc import Callable
import numpy as np
import cv2
from tqdm import tqdm

# Third-party imports
import gymnasium as gym
import h5py
import imageio
import torch

# Isaac Sim / Omniverse imports
from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.examples.example_environments.cli import add_example_environments_cli_args, get_arena_builder_from_cli

# Initialize CLI parser
parser = get_isaaclab_arena_cli_parser()

# Custom arguments
parser.add_argument(
    "--num_demos", 
    type=int, 
    default=-1, 
    help="Number of successful demonstrations to record in this session. Set to -1 for infinite."
)
parser.add_argument(
    "--start_episode",
    type=int,
    default=0,
    help="First source episode to replay; useful for a bounded diagnostic run.",
)
parser.add_argument(
    "--robotwin_data_root", 
    type=str, 
    required=True, 
    help="Path to RoboTwin data directory containing 'data/' folder and 'scene_info.json'."
)
parser.add_argument(
    "--output", 
    type=str, 
    required=True, 
    help="Directory to save the results and the processed log file."
)
parser.add_argument(
    "--step_skip", 
    type=int, 
    default=2, 
    help="Recording interval for HDF5 and Video. (1 = every step, 2 = every other)."
)
parser.add_argument(
    "--estimate_block_poses",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Recover missing colored-block initial poses from the first head-camera frame.",
)
parser.add_argument(
    "--stability_steps",
    type=int,
    default=0,
    help="Hold the final action for this many control steps before success validation.",
)
parser.add_argument(
    "--validate_block_stability",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Require stacked blocks to be centered, upright, and nearly stationary.",
)

# Append environment-specific arguments
add_example_environments_cli_args(parser)

# Parse arguments
args_cli = parser.parse_args()

# Launch the simulator app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.log
import omni.ui as ui
import isaaclab.sim as sim_utils
from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg, Se3SpaceMouse, Se3SpaceMouseCfg
from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode, RecorderTerm, RecorderTermCfg
from isaaclab.managers.recorder_manager import RecorderManager
from isaaclab.utils import configclass

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.append(ROOT_DIR)

def spawn_light_once():
    prim_path = "/World/Light"
    cfg = sim_utils.DomeLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
        texture_file=f"{ROOT_DIR}/assets/texture/base0.exr",
    )
    cfg.func(prim_path, cfg)

def load_processed_log(output_dir: str) -> set[int]:
    """Loads the set of processed episode indices from the log file.

    Reads 'processed_episodes.txt' from the output directory to determine which 
    episodes have already been attempted (success or failure) in previous runs.

    Args:
        output_dir: The directory where the log file is stored.

    Returns:
        set[int]: A set of episode indices that have already been processed.
    """
    log_path = os.path.join(output_dir, "processed_episodes.txt")
    processed = set()
    if not os.path.exists(log_path):
        print(f"[Info] Log file not found. Creating new log: {log_path}")
        with open(log_path, 'w') as f:
            pass  # Create empty file
        return processed

    print(f"[Info] Found checkpoint file: {log_path}")
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.isdigit():
                processed.add(int(line))
                
    print(f"[Info] Loaded {len(processed)} previously processed episodes.")
    return processed

def update_processed_log(output_dir: str, ep_idx: int):
    """Updates the log file with the newly processed episode index.

    Appends the episode index to 'processed_episodes.txt'.

    Args:
        output_dir: The directory where the log file is stored.
        ep_idx: The index of the episode that was just processed.
    """
    log_path = os.path.join(output_dir, "processed_episodes.txt")
    with open(log_path, "a") as f:
        f.write(f"{ep_idx}\n")

def get_next_demo_idx_from_hdf5(output_root: str) -> int:
    data_dir = os.path.join(output_root, "data")
    if not os.path.isdir(data_dir):
        return 0

    mx = -1
    for fn in os.listdir(data_dir):
        if fn.startswith("demo_") and fn.endswith(".hdf5"):
            mid = fn[len("demo_"):-len(".hdf5")]
            if mid.isdigit():
                mx = max(mx, int(mid))
    return mx + 1

def copy_instruction_json(robotwin_root: str, output_root: str, ep_idx: int, demo_idx: int) -> None:
    """Copy RoboTwin instruction json -> output/instructions/demo_<idx>.json

    RoboTwin format:
      <robotwin_root>/instructions/episode<ep_idx>.json
    Output format:
      <output_root>/instructions/demo_<demo_idx>.json
    """
    src = os.path.join(robotwin_root, "instructions", f"episode{ep_idx}.json")
    dst_dir = os.path.join(output_root, "instructions")
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, f"demo_{demo_idx}.json")

    if not os.path.exists(src):
        print(f"[Warning] Instruction file missing: {src} (skip copy)")
        return

    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[Info] Saved instruction: {dst}")

class ObservationRecorderTerm(RecorderTerm):
    """Custom recorder term to capture the full observation buffer."""
    
    def record_post_step(self):
        if not hasattr(self._env, "obs_buf") or not self._env.obs_buf:
            return None, None
            
        recorded_data = {}
        for group_name, group_data in self._env.obs_buf.items():
            if isinstance(group_data, dict):
                recorded_data[group_name] = {
                    k: v.clone() if isinstance(v, torch.Tensor) else v 
                    for k, v in group_data.items()
                }
            elif isinstance(group_data, torch.Tensor):
                recorded_data[group_name] = group_data.clone()
        
        return "observations", recorded_data

@configclass
class ObservationRecorderTermCfg(RecorderTermCfg):
    class_type = ObservationRecorderTerm

def remap_action(env: gym.Env, rt_qpos: torch.Tensor, device: str) -> torch.Tensor:
    """Maps RoboTwin joint positions to Isaac Lab embodiment actions."""
    left_arm_len, _, right_arm_len, _ = env.action_manager.action_term_dim
    l_arm = rt_qpos[:left_arm_len]
    r_arm = rt_qpos[left_arm_len+1:left_arm_len+1+right_arm_len]
    l_grip_val = rt_qpos[left_arm_len]
    r_grip_val = rt_qpos[left_arm_len+1+right_arm_len]
    l_grip_dual = l_grip_val.repeat(2)
    r_grip_dual = r_grip_val.repeat(2)
    if args_cli.embodiment in ["dual_franka"]:
        r_grip_val *= 0.04
        l_grip_val *= 0.04
    elif args_cli.embodiment in ["aloha"]:
        l_grip_dual *= 0.058
        l_grip_dual -= 0.01
        r_grip_dual *= 0.058
        r_grip_dual -= 0.01
    else:
        raise ValueError(f"Unsupported robot_type: {args_cli.embodiment}")
    
    return torch.cat([l_arm, l_grip_dual, r_arm, r_grip_dual], dim=-1).to(device)

BLOCK_COLOR_RANGES = {
    "block1": ((0, 105, 70), (12, 255, 255)),
    "block2": ((38, 70, 45), (92, 255, 255)),
    "block3": ((96, 75, 45), (132, 255, 255)),
}


def _decode_robotwin_rgb(value) -> np.ndarray:
    encoded = np.frombuffer(bytes(value), dtype=np.uint8)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("Could not decode RoboTwin RGB frame")
    # RoboTwin passes source RGB arrays directly to cv2.imencode.  The values
    # returned by cv2.imdecode therefore numerically match the source RGB data.
    return decoded


def _largest_color_centroid(mask: np.ndarray) -> tuple[float, float]:
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        raise ValueError("No colored block component found")
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[component, cv2.CC_STAT_AREA])
    if area < 20:
        raise ValueError(f"Colored block component is too small: {area} pixels")
    return tuple(float(value) for value in centroids[component])


def _pixel_to_world_plane(
    u: float,
    v: float,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    plane_z: float,
) -> np.ndarray:
    rotation = extrinsic[:, :3]
    translation = extrinsic[:, 3]
    camera_origin = -rotation.T @ translation
    camera_ray = np.linalg.inv(intrinsic) @ np.array([u, v, 1.0])
    world_ray = rotation.T @ camera_ray
    distance = (plane_z - camera_origin[2]) / world_ray[2]
    return camera_origin + distance * world_ray


def _estimate_block_objects(handle: h5py.File, task_name: str) -> dict:
    block_count = 2 if task_name == "stack_blocks_two" else 3
    source_plane_z = 0.766 if task_name.startswith("stack_blocks") else 0.765
    spawn_z = 0.770 if task_name.startswith("stack_blocks") else 0.765
    camera = handle["observation/head_camera"]
    image = _decode_robotwin_rgb(camera["rgb"][0])
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    intrinsic = np.asarray(camera["intrinsic_cv"][0], dtype=np.float64)
    extrinsic = np.asarray(camera["extrinsic_cv"][0], dtype=np.float64)

    objects = {}
    for name, (lower, upper) in list(BLOCK_COLOR_RANGES.items())[:block_count]:
        mask = cv2.inRange(hsv, np.asarray(lower, np.uint8), np.asarray(upper, np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        u, v = _largest_color_centroid(mask)
        world = _pixel_to_world_plane(u, v, intrinsic, extrinsic, source_plane_z)
        objects[name] = {
            "pos": [float(world[0]), float(world[1]), spawn_z],
            "quat": [1.0, 0.0, 0.0, 0.0],
            "spawn_clearance": 0.003,
        }
    print(f"[Info] Estimated {task_name} initial block poses: {objects}")
    return objects


def _random_block_pose(
    rng: np.random.RandomState,
    existing: list[np.ndarray],
    target_threshold: float,
) -> tuple[np.ndarray, float]:
    while True:
        x = rng.uniform(-0.28, 0.28)
        y = rng.uniform(-0.08, 0.05)
        while abs(x) < 0.15 and y > 0:
            y = rng.uniform(-0.08, 0.0)
        rng.uniform(0.766, 0.766)
        rng.uniform(0.0, 0.0)
        rng.uniform(0.0, 0.0)
        yaw = rng.uniform(-0.75, 0.75)
        xy = np.array([x, y], dtype=np.float64)
        if abs(x) < 0.05 or np.sum((xy - np.array([0.0, -0.1])) ** 2) < target_threshold:
            continue
        if any(np.sum((xy - other) ** 2) < 0.01 for other in existing):
            continue
        return xy, float(yaw)


def _recover_seeded_block_objects(task_name: str, seed: int) -> dict:
    """Reproduce the clean RoboTwin actor sampler using the archived episode seed."""
    rng = np.random.RandomState(seed)
    # Clean-data setup consumes three draws before load_actors (table/camera setup).
    rng.random_sample(3)
    block_count = 2 if task_name == "stack_blocks_two" else 3

    while True:
        positions: list[np.ndarray] = []
        yaws: list[float] = []
        target_threshold = 0.01 if task_name == "blocks_ranking_rgb" else 0.0225
        for _ in range(block_count):
            xy, yaw = _random_block_pose(rng, positions, target_threshold)
            positions.append(xy)
            yaws.append(yaw)
        if task_name != "blocks_ranking_rgb":
            break
        eps = np.array([0.12, 0.03])
        already_ranked = (
            np.all(np.abs(positions[0] - positions[1]) < eps)
            and np.all(np.abs(positions[1] - positions[2]) < eps)
            and positions[0][0] < positions[1][0] < positions[2][0]
        )
        if not already_ranked:
            break

    spawn_z = 0.770 if task_name.startswith("stack_blocks") else 0.765
    objects = {}
    for index, (xy, yaw) in enumerate(zip(positions, yaws, strict=True), start=1):
        objects[f"block{index}"] = {
            "pos": [float(xy[0]), float(xy[1]), spawn_z],
            "quat": [float(np.cos(yaw / 2.0)), 0.0, 0.0, float(np.sin(yaw / 2.0))],
            "spawn_clearance": 0.003,
        }
    print(f"[Info] Recovered {task_name} seed {seed} initial block poses: {objects}")
    return objects


def _random_bowl_pose(
    rng: np.random.RandomState,
    existing: list[np.ndarray],
) -> np.ndarray:
    """Reproduce RoboTwin's stack_bowls_two rejection sampler."""
    while True:
        x = rng.uniform(-0.3, 0.3)
        y = rng.uniform(-0.15, 0.15)
        while abs(x) < 0.15 and y > 0:
            y = rng.uniform(-0.15, 0.0)
        # rand_pose always consumes the z draw, even for a fixed interval.
        rng.uniform(0.741, 0.741)
        xy = np.array([x, y], dtype=np.float64)
        if abs(x) < 0.09:
            continue
        if np.sum((xy - np.array([0.0, -0.1])) ** 2) < 0.0169:
            continue
        if any(np.sum((xy - other) ** 2) < 0.0169 for other in existing):
            continue
        return xy


def _recover_seeded_bowl_objects(task_name: str, seed: int) -> dict:
    """Restore two-bowl poses omitted from clean-data scene_info.json."""
    if task_name != "stack_bowls_two":
        raise ValueError(f"Unsupported seeded bowl task: {task_name}")
    rng = np.random.RandomState(seed)
    # Base_Task setup consumes three draws before task actors are sampled.
    rng.random_sample(3)
    positions: list[np.ndarray] = []
    for _ in range(2):
        positions.append(_random_bowl_pose(rng, positions))
    positions.sort(key=lambda value: value[1])

    objects = {
        f"bowl{index}": {
            "pos": [float(xy[0]), float(xy[1]), 0.766],
            # The Isaac bowl USD is upright at identity; the SAPIEN source mesh
            # uses a different model-space orientation.
            "quat": [1.0, 0.0, 0.0, 0.0],
            "spawn_clearance": 0.01,
        }
        for index, xy in enumerate(positions, start=1)
    }
    print(f"[Info] Recovered {task_name} seed {seed} initial bowl poses: {objects}")
    return objects


def _estimate_bowl_objects(handle: h5py.File, task_name: str) -> dict:
    """Recover two bowl centers from the archived first head-camera frame."""
    if task_name != "stack_bowls_two":
        raise ValueError(f"Unsupported bowl vision task: {task_name}")
    camera = handle["observation/head_camera"]
    image = _decode_robotwin_rgb(camera["rgb"][0])
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    height, width = image.shape[:2]
    yy, _ = np.indices((height, width))
    # The ceramic asset is the only strongly saturated object on the white
    # tabletop. Exclude the very bottom edge where robot links enter the view.
    mask = ((hsv[..., 1] > 55) & (yy < int(height * 0.9))).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates = []
    for component in range(1, count):
        x, y, w, h, area = (int(value) for value in stats[component])
        if 300 <= area <= 6000 and 20 <= w <= 100 and 20 <= h <= 100:
            candidates.append((area, centroids[component]))
    if len(candidates) < 2:
        raise ValueError(f"Expected two bowl components, found {len(candidates)}")
    candidates.sort(key=lambda item: item[0], reverse=True)

    intrinsic = np.asarray(camera["intrinsic_cv"][0], dtype=np.float64)
    extrinsic = np.asarray(camera["extrinsic_cv"][0], dtype=np.float64)
    positions = []
    for _, centroid in candidates[:2]:
        world = _pixel_to_world_plane(
            float(centroid[0]), float(centroid[1]), intrinsic, extrinsic, 0.741
        )
        positions.append(world[:2])
    # RoboTwin names bowl1/bowl2 after sorting source poses by y.
    positions.sort(key=lambda value: value[1])
    objects = {
        f"bowl{index}": {
            "pos": [float(xy[0]), float(xy[1]), 0.766],
            "quat": [1.0, 0.0, 0.0, 0.0],
            "spawn_clearance": 0.01,
        }
        for index, xy in enumerate(positions, start=1)
    }
    print(f"[Info] Estimated {task_name} initial bowl poses: {objects}")
    return objects


class RoboTwinDataLoader:
    def __init__(self, root_dir: str, device: str = "cpu"):
        self.root_dir = root_dir
        self.device = device
        scene_path = os.path.join(root_dir, "scene_info.json")
        if not os.path.exists(scene_path):
            raise FileNotFoundError(f"scene_info.json not found at {scene_path}")
        with open(scene_path, 'r') as f:
            self.scene_info = json.load(f)
        seed_path = os.path.join(root_dir, "seed.txt")
        self.seeds = []
        if os.path.exists(seed_path):
            with open(seed_path, "r", encoding="utf-8") as seed_file:
                self.seeds = [int(value) for value in seed_file.read().split()]

    def load_episode(self, ep_idx: int) -> dict | None:
        ep_key = f"episode_{ep_idx}"
        if ep_key not in self.scene_info:
            return None
            
        info = self.scene_info[ep_key]
        h5_path = os.path.join(self.root_dir, "data", f"episode{ep_idx}.hdf5")
        
        if not os.path.exists(h5_path):
            print(f"[Warning] HDF5 file missing for episode {ep_idx}")
            return None

        with h5py.File(h5_path, 'r') as f:
            actions = torch.tensor(f['joint_action']['vector'][:], 
                                  dtype=torch.float32, device=self.device)
            objects = info.get("object_info", {})
            if not objects and args_cli.estimate_block_poses:
                normalized_root = self.root_dir.replace("\\", "/")
                task_name = next(
                    (
                        name
                        for name in (
                            "stack_bowls_two",
                            "stack_blocks_two",
                            "stack_blocks_three",
                            "blocks_ranking_rgb",
                        )
                        if f"/{name}/" in normalized_root
                    ),
                    None,
                )
                if task_name is not None:
                    try:
                        if task_name == "stack_bowls_two":
                            try:
                                objects = _estimate_bowl_objects(f, task_name)
                            except (KeyError, ValueError, cv2.error) as vision_error:
                                if ep_idx >= len(self.seeds):
                                    raise vision_error
                                objects = _recover_seeded_bowl_objects(task_name, self.seeds[ep_idx])
                        elif task_name == "blocks_ranking_rgb":
                            # The archived seed does not reliably reproduce every
                            # episode's sampled layout.  Recover the RGB blocks
                            # from that episode's own first camera frame instead.
                            try:
                                objects = _estimate_block_objects(f, task_name)
                            except (KeyError, ValueError, cv2.error) as vision_error:
                                if ep_idx >= len(self.seeds):
                                    raise vision_error
                                print(f"[Warning] Vision pose recovery failed: {vision_error}")
                                objects = _recover_seeded_block_objects(task_name, self.seeds[ep_idx])
                        elif ep_idx < len(self.seeds):
                            objects = _recover_seeded_block_objects(task_name, self.seeds[ep_idx])
                        else:
                            objects = _estimate_block_objects(f, task_name)
                    except (KeyError, ValueError, cv2.error) as error:
                        print(f"[Warning] Could not estimate initial block poses: {error}")
            
        return {
            "objects": objects,
            "actions": actions,
            "length": len(actions)
        }

def teleport_entities(env: gym.Env, scene, episode_data: dict):
    """Teleports objects and robots to the initial state."""
    # 1. Teleport Rigid Objects
    for obj_name, pose_data in episode_data["objects"].items():
        if obj_name in scene.rigid_objects:
            obj_asset = scene.rigid_objects[obj_name]
            position = list(pose_data["pos"])
            position[2] += float(pose_data.get("spawn_clearance", 0.01))
            pos = torch.tensor(position, device=env.device)
            quat = torch.tensor(
                pose_data.get("quat", [1.0, 0.0, 0.0, 0.0]), device=env.device
            )
            combined_pose = torch.cat([pos, quat]).unsqueeze(0)
            obj_asset.write_root_pose_to_sim(combined_pose)
            obj_asset.write_root_velocity_to_sim(torch.zeros_like(combined_pose[:, :6]))

    # 2. Teleport Robot Joints
    if "robot" not in scene.articulations and "left_robot" not in scene.articulations:
        return

    init_q_rt = episode_data["actions"][0]
    action_isaac = remap_action(env, init_q_rt, env.device)

    if args_cli.embodiment in ["aloha"]:
        robot = scene["robot"]
        full_q = robot.data.default_joint_pos.clone()
        idx_l_arm = [6, 14, 18, 22, 26, 30]
        idx_l_grip = [34, 35] 
        idx_r_arm = [8, 15, 19, 23, 27, 31]
        idx_r_grip = [36, 37] 
        full_q[:, idx_l_arm] = action_isaac[0:6]
        full_q[:, idx_l_grip] = action_isaac[6:8]
        full_q[:, idx_r_arm] = action_isaac[8:14]
        full_q[:, idx_r_grip] = action_isaac[14:16]
        robot.write_joint_position_to_sim(full_q)
        robot.write_joint_velocity_to_sim(torch.zeros_like(full_q))
        robot.set_joint_position_target(full_q)

    elif args_cli.embodiment in ["dual_franka"]:
        target_dtype = scene["left_robot"].data.joint_pos_target.dtype
        if "left_robot" in scene.articulations:
            l_robot = scene["left_robot"]
            l_joint_ids = list(range(9)) 
            l_q = action_isaac[[0,1,2,3,4,5,6,14,15]].unsqueeze(0).to(target_dtype)
            l_robot.write_joint_position_to_sim(l_q, joint_ids=l_joint_ids)
            l_robot.write_joint_velocity_to_sim(torch.zeros_like(l_q))
            l_robot.set_joint_position_target(l_q, joint_ids=l_joint_ids)
        if "right_robot" in scene.articulations:
            r_robot = scene["right_robot"]
            r_joint_ids = list(range(9))
            r_q = action_isaac[[7,8,9,10,11,12,13,16,17]].unsqueeze(0).to(target_dtype)
            r_robot.write_joint_position_to_sim(r_q, joint_ids=r_joint_ids)
            r_robot.write_joint_velocity_to_sim(torch.zeros_like(r_q))
            r_robot.set_joint_position_target(r_q, joint_ids=r_joint_ids)
    
    scene.write_data_to_sim()
    for _ in range(30):
        env.step(action_isaac.unsqueeze(0))

def convert_hdf5_to_videos(hdf5_path: str, output_dir: str, env_cfg, next_demo_idx: int = 0):
    """Post-processing: Convert recorded HDF5 data to MP4."""
    if not os.path.exists(hdf5_path):
        print(f"[Error] HDF5 file not found: {hdf5_path}")
        return

    print(f"\n[Post-Process] Extracting videos from HDF5...")
    videos_dir = os.path.join(output_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)

    with h5py.File(hdf5_path, 'r') as f:
        if 'data' not in f:
            return
        
        data_grp = f["data"]
        demo_keys = sorted(
            data_grp.keys(),
            key=lambda k: int(k.split("_")[-1]) if k.startswith("demo_") and k.split("_")[-1].isdigit() else 10**18
        )

        for idx, demo_key in enumerate(demo_keys):
            demo_grp = f['data'][demo_key]
            save_demo_key = f"demo_{next_demo_idx + idx}" 
            
            # Locate observation group
            obs_grp = None
            if 'observations' in demo_grp:
                if 'camera_obs' in demo_grp['observations']:
                    obs_grp = demo_grp['observations']['camera_obs']
                else:
                    obs_grp = demo_grp['observations'] 
            
            if obs_grp is None: continue

            for key in obs_grp.keys():
                dataset = obs_grp[key]
                if dataset.ndim == 4 and dataset.shape[-1] == 3:
                    print(f"  -> Converting {save_demo_key}/{key} ...")
                    data = dataset[:] 
                    
                    video_name = f"{save_demo_key}_{key}.mp4"
                    save_path = os.path.join(videos_dir, video_name)
                    
                    fps = int(1.0 / (env_cfg.sim.dt * env_cfg.sim.render_interval * args_cli.step_skip))
                    if fps <= 0 or fps > 100: fps = 30 
                    
                    imageio.mimwrite(save_path, data, fps=fps, quality=8)
                    print(f"     Saved to {save_path}")

def images_encoding(imgs: np.ndarray):
    """cv2 jpg encode -> list[bytes], max_len"""
    encode_data = []
    max_len = 0
    for i in range(len(imgs)):
        ok, enc = cv2.imencode(".jpg", imgs[i])
        if not ok:
            raise RuntimeError(f"cv2.imencode failed at frame {i}")
        b = enc.tobytes()
        encode_data.append(b)
        max_len = max(max_len, len(b))
    return encode_data, max_len


def _copy_attrs(src_obj, dst_obj):
    for k, v in src_obj.attrs.items():
        dst_obj.attrs[k] = v


def _compress_camera_obs(src_cam_grp: h5py.Group, dst_cam_grp: h5py.Group):
    _copy_attrs(src_cam_grp, dst_cam_grp)

    for cam_key in src_cam_grp.keys():
        obj = src_cam_grp[cam_key]
        if isinstance(obj, h5py.Dataset) and obj.dtype == np.uint8 and obj.ndim == 4 and obj.shape[-1] == 3:
            imgs = obj[:]  # (T,H,W,3)
            encode_data, max_len = images_encoding(imgs)
            dst_cam_grp.create_dataset(cam_key, data=encode_data, dtype=f"S{max_len}")
        elif isinstance(obj, h5py.Dataset):
            ds = dst_cam_grp.create_dataset(cam_key, data=obj[()], dtype=obj.dtype)
            _copy_attrs(obj, ds)
        else:
            g = dst_cam_grp.create_group(cam_key)
            _copy_attrs(obj, g)


def _copy_group_with_compression(src_grp: h5py.Group, dst_grp: h5py.Group):
    _copy_attrs(src_grp, dst_grp)

    for key, obj in src_grp.items():
        if isinstance(obj, h5py.Group):
            if key == "camera_obs" and dst_grp.name.endswith("/observations"):
                cam_dst = dst_grp.create_group("camera_obs")
                _compress_camera_obs(obj, cam_dst)
            else:
                sub = dst_grp.create_group(key)
                _copy_group_with_compression(obj, sub)
        elif isinstance(obj, h5py.Dataset):
            ds = dst_grp.create_dataset(key, data=obj[()], dtype=obj.dtype)
            _copy_attrs(obj, ds)


def split_and_compress_big_hdf5(
    big_hdf5_path: str,
    out_dir: str,
    delete_big: bool = True,
    keep_data_root: bool = False,
    next_demo_idx: int = 0
):
    os.makedirs(out_dir, exist_ok=True)

    with h5py.File(big_hdf5_path, "r") as fin:
        if "data" not in fin:
            raise ValueError(f"{big_hdf5_path} missing top-level group 'data'")

        data_in = fin["data"]
        demo_keys = sorted(
            data_in.keys(),
            key=lambda k: int(k.split("_")[-1]) if k.startswith("demo_") and k.split("_")[-1].isdigit() else 10**18
        )
        print(f"[Post] Found {len(demo_keys)} demos in {big_hdf5_path}")

        for idx, demo_key in tqdm(enumerate(demo_keys), desc="[Post] Splitting+compressing"):
            demo_in = data_in[demo_key]
            out_path = os.path.join(out_dir, f"demo_{next_demo_idx + idx}.hdf5")

            with h5py.File(out_path, "w") as fout:
                _copy_attrs(fin, fout)

                if keep_data_root:
                    data_out = fout.create_group("data")
                    _copy_attrs(data_in, data_out)
                    demo_out = data_out.create_group(demo_key)
                    _copy_group_with_compression(demo_in, demo_out)
                else:
                    _copy_group_with_compression(demo_in, fout)

    if delete_big:
        os.remove(big_hdf5_path)
        print(f"[Post] Deleted big file: {big_hdf5_path}")

def setup_output_directories() -> tuple[str, str]:
    output_dir = args_cli.output
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"replay_data_{timestamp}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    return output_dir, output_filename

def create_environment_config(output_dir: str, output_file_name: str):
    try:
        arena_builder = get_arena_builder_from_cli(args_cli)
        env_name, env_cfg = arena_builder.build_registered()
    except Exception as e:
        omni.log.error(f"Failed to parse environment configuration: {e}")
        exit(1)

    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None 
    
    env_cfg.terminations.time_out = None
    env_cfg.sim.dt = 1 / 250
    env_cfg.sim.decimation = 5
    env_cfg.recorders = None 

    recorder_cfg = ActionStateRecorderManagerCfg(
        dataset_export_dir_path=output_dir,
        dataset_filename=output_file_name, 
        dataset_export_mode=DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    )
    recorder_cfg.observations = ObservationRecorderTermCfg()

    return env_cfg, env_name, success_term, recorder_cfg

def create_environment(env_cfg, env_name: str) -> gym.Env:
    try:
        env = gym.make(env_name, cfg=env_cfg).unwrapped
        return env
    except Exception as e:
        omni.log.error(f"Failed to create environment: {e}")
        exit(1)

def process_success_condition(env: gym.Env, success_term, recorder_manager: RecorderManager) -> bool:
    if success_term is None:
        print("No success term defined. Marking episode as SUCCESS (Blind Replay).")
        recorder_manager.record_pre_reset([0], force_export_or_skip=False)
        recorder_manager.set_success_to_episodes(
            [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
        )
        recorder_manager.export_episodes([0])
        return True

    try:
        is_success = success_term.func(env, **success_term.params)
        if isinstance(is_success, torch.Tensor):
            is_success = is_success.any().item()
            
        if is_success and args_cli.validate_block_stability and "stack_blocks_" in args_cli.environment:
            block_assets = [
                asset
                for name, asset in env.scene.rigid_objects.items()
                if name.startswith("block")
            ]
            block_assets.sort(key=lambda asset: float(asset.data.root_pos_w[0, 2]))
            positions = torch.stack([asset.data.root_pos_w[0] for asset in block_assets])
            quaternions = torch.stack([asset.data.root_quat_w[0] for asset in block_assets])
            linear_speeds = torch.stack(
                [torch.linalg.vector_norm(asset.data.root_lin_vel_w[0]) for asset in block_assets]
            )
            angular_speeds = torch.stack(
                [torch.linalg.vector_norm(asset.data.root_ang_vel_w[0]) for asset in block_assets]
            )
            centered_xy = float(torch.max(torch.abs(positions[1:, :2] - positions[:-1, :2])))
            # For wxyz quaternions, world-up dot local-up is 1 - 2(x^2 + y^2).
            upright_cos = 1.0 - 2.0 * (quaternions[:, 1] ** 2 + quaternions[:, 2] ** 2)
            min_upright_cos = float(torch.min(upright_cos))
            max_linear_speed = float(torch.max(linear_speeds))
            max_angular_speed = float(torch.max(angular_speeds))
            stable = (
                centered_xy <= 0.015
                and min_upright_cos >= 0.994
                and max_linear_speed <= 0.02
                and max_angular_speed <= 0.20
            )
            print(
                "[Stability] "
                f"centered_xy={centered_xy:.5f} m, "
                f"min_upright_cos={min_upright_cos:.6f}, "
                f"max_linear_speed={max_linear_speed:.5f} m/s, "
                f"max_angular_speed={max_angular_speed:.5f} rad/s, "
                f"pass={stable}"
            )
            is_success = stable

        if is_success:
            print(f"Success condition met! Exporting data...")
            recorder_manager.record_pre_reset([0], force_export_or_skip=False)
            recorder_manager.set_success_to_episodes(
                [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
            )
            recorder_manager.export_episodes([0])
            return True
        else:
            print("Success condition NOT met at the end of replay.")
            return False
            
    except Exception as e:
        print(f"Error checking success condition: {e}")
        return False

def handle_reset(env: gym.Env, recorder_manager: RecorderManager):
    print("Resetting environment...")
    env.sim.reset()
    recorder_manager.reset()
    env.reset()

def run_simulation_loop(
    env: gym.Env,
    teleop_interface: object | None,
    success_term: object | None,
    recorder_manager: RecorderManager | None,
    loader: RoboTwinDataLoader | None,
    next_demo_idx: int = 0,
) -> int:
    """Runs the main replay loop with checkpoint skipping."""
    
    total_successful_demos = 0
    env.sim.set_camera_view([0.0, -0.5, 1.75], [0.0, -0.05, 0.75])

    # 1. Load Processed History
    processed_indices = load_processed_log(args_cli.output)
    
    if args_cli.start_episode < 0:
        raise ValueError("--start_episode must be non-negative")
    ep_idx = args_cli.start_episode
    target_demos = args_cli.num_demos
    
    while True:
        if target_demos != -1 and total_successful_demos >= target_demos:
            print(f"Reached target number of successful demos ({target_demos}). Stopping.")
            break

        # 2. Checkpoint Logic: Skip if already processed
        if ep_idx in processed_indices:
            # We silently skip or log briefly. 
            # Note: We continue to increment ep_idx until we find an unprocessed one
            # or until the loader returns None (end of dataset).
            ep_idx += 1
            continue

        print(f"\n--- Processing Episode {ep_idx} ---")
        
        # 3. Load Data
        data = loader.load_episode(ep_idx)
        
        # If no data is returned, we have reached the end of the available RoboTwin dataset
        if not data:
            print(f"No more data found at episode {ep_idx}. Stopping loop.")
            break
        
        # 4. Reset & Teleport
        handle_reset(env, recorder_manager)
        teleport_entities(env, env.scene, data)
        env.scene.update(dt=env.physics_dt)
        env.observation_manager.compute()

        # 5. Start Recording
        recorder_manager.reset()
        recorder_manager.record_post_reset([0])

        step_count = 0
        
        # 6. Playback
        last_action_isaac = None
        for action_rt in data["actions"]:
            action_isaac = remap_action(env, action_rt, env.device).unsqueeze(0)
            # for _ in range(5):
            env.step(action_isaac)
            step_count += 1
            if (step_count % args_cli.step_skip == 0):
                recorder_manager.record_post_step()

            last_action_isaac = action_isaac

        if last_action_isaac is not None:
            for _ in range(max(args_cli.stability_steps, 0)):
                env.step(last_action_isaac)

        # 7. Check Success
        if process_success_condition(env, success_term, recorder_manager):
            demo_idx = next_demo_idx + total_successful_demos
            copy_instruction_json(args_cli.robotwin_data_root, args_cli.output, ep_idx, demo_idx)
            total_successful_demos += 1
            print(f"Total Successful: {total_successful_demos} / {target_demos if target_demos > 0 else 'Inf'}")
        else:
            print(f"Episode {ep_idx} discarded.")

        # 8. Mark as Processed (Log immediately to save state)
        update_processed_log(args_cli.output, ep_idx)
            
        ep_idx += 1

    return total_successful_demos

def main() -> None:
    output_dir, output_file_name = setup_output_directories()
    global env_cfg  
    env_cfg, env_name, success_term, recorder_cfg = create_environment_config(output_dir, output_file_name)

    env = create_environment(env_cfg, env_name)
    spawn_light_once()
    recorder_manager = RecorderManager(recorder_cfg, env)
    loader = RoboTwinDataLoader(root_dir=args_cli.robotwin_data_root, device=env.device)

    next_demo_idx = get_next_demo_idx_from_hdf5(args_cli.output)
    print(f"[Info] Next demo idx start from: {next_demo_idx}")

    total_successful_demos = run_simulation_loop(env, None, success_term, recorder_manager, loader, next_demo_idx)

    env.close()
    print(f"Recording session completed with {total_successful_demos} successful demonstrations")
    print(f"Demonstrations saved to: {args_cli.output}")

    full_hdf5_path = os.path.join(output_dir, output_file_name + '.hdf5')
    convert_hdf5_to_videos(full_hdf5_path, output_dir, env_cfg, next_demo_idx)

    split_dir = os.path.join(args_cli.output, "data")
    split_and_compress_big_hdf5(
        big_hdf5_path=full_hdf5_path,
        out_dir=split_dir,
        delete_big=True,
        keep_data_root=False,
        next_demo_idx=next_demo_idx
    )

if __name__ == "__main__":
    main()
    simulation_app.close()
