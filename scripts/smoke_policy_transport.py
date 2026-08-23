"""Send one real RoboTwin observation through the GR00T policy-server boundary."""

from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from twingr00t_lab.policy_bridge import build_groot_observation, groot_action_to_isaac
from twingr00t_lab.policy_client import PolicyTransportClient


def _rgb_frame(dataset: h5py.Dataset) -> np.ndarray:
    value = dataset[0]
    if dataset.ndim == 1:
        with Image.open(BytesIO(bytes(value))) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    return np.asarray(value, dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("demo", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--prompt", default="stack the three bowls")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with h5py.File(args.demo, "r") as handle:
        policy_group = handle["observations/policy"]
        camera_group = handle["observations/camera_obs"]
        policy = {
            key: np.asarray(policy_group[key][0], dtype=np.float32)
            for key in ("left_arm_pos", "left_gripper_pos", "right_arm_pos", "right_gripper_pos")
        }
        cameras = {
            key: _rgb_frame(camera_group[key])
            for key in ("head_camera_rgb", "left_camera_rgb", "right_camera_rgb")
        }

    observation = build_groot_observation(policy, cameras, args.prompt)
    with PolicyTransportClient(args.host, args.port) as client:
        if not client.ping():
            raise RuntimeError(f"GR00T policy server is not ready on {args.host}:{args.port}")
        client.reset({"episode_index": 0, "step_index": 0})
        action, info = client.get_action(observation)
    isaac_chunk = groot_action_to_isaac(action)
    result = {
        "status": "pass",
        "demo": str(args.demo),
        "server": f"{args.host}:{args.port}",
        "logical_action_keys": sorted(action),
        "isaac_action_shape": list(isaac_chunk.shape),
        "action_finite": bool(np.isfinite(isaac_chunk).all()),
        "server_info": info,
    }
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
