"""Compare a trained N1.7 policy with saved teacher actions on camera frames."""

from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from twingr00t_lab.dataset import collapse_isaac_action
from twingr00t_lab.n17_sampling import configure_n17_sampling
from twingr00t_lab.policy_bridge import build_groot_observation


def frame_rgb(dataset: h5py.Dataset, index: int, *, correct_legacy_jpeg_bgr: bool = False) -> np.ndarray:
    with Image.open(BytesIO(bytes(dataset[index]))) as image:
        frame = np.asarray(image.convert("RGB"), dtype=np.uint8)
        return np.ascontiguousarray(frame[..., ::-1] if correct_legacy_jpeg_bgr else frame)


def evaluate_episode(
    policy: object,
    path: Path,
    prompt: str,
    indices: list[int],
    *,
    correct_legacy_jpeg_bgr: bool = False,
) -> list[dict]:
    rows: list[dict] = []
    with h5py.File(path, "r") as handle:
        observed = handle["observations/policy"]
        cameras = handle["observations/camera_obs"]
        targets = collapse_isaac_action(np.asarray(observed["actions"][:], dtype=np.float32))
        for index in indices:
            if index >= len(targets) - 16:
                continue
            state = {
                key: observed[key][index]
                for key in ("left_arm_pos", "left_gripper_pos", "right_arm_pos", "right_gripper_pos")
            }
            images = {
                key: frame_rgb(cameras[key], index, correct_legacy_jpeg_bgr=correct_legacy_jpeg_bgr)
                for key in ("head_camera_rgb", "left_camera_rgb", "right_camera_rgb")
            }
            observation = build_groot_observation(state, images, prompt)
            prediction, _ = policy.get_action(observation)
            predicted = np.concatenate(
                (
                    prediction["left_arm"][0],
                    prediction["left_gripper"][0],
                    prediction["right_arm"][0],
                    prediction["right_gripper"][0],
                ),
                axis=1,
            )[:16]
            actual = targets[index : index + len(predicted)]
            rows.append(
                {
                    "frame": index,
                    "mae_14d": float(np.abs(predicted - actual).mean()),
                    "first_prediction": predicted[0].round(3).tolist(),
                    "first_teacher": actual[0].round(3).tolist(),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--train-demo", required=True, type=Path)
    parser.add_argument("--heldout-demo", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 50, 100, 200, 300])
    parser.add_argument("--correct-legacy-jpeg-bgr", action="store_true")
    parser.add_argument("--flow-steps", type=int, default=4)
    parser.add_argument("--sample-average", type=int, default=1)
    args = parser.parse_args()

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        model_path=str(args.checkpoint),
        device=args.device,
    )
    configure_n17_sampling(policy, flow_steps=args.flow_steps, sample_average=args.sample_average)
    result = {
        "checkpoint": str(args.checkpoint),
        "flow_steps": args.flow_steps,
        "sample_average": args.sample_average,
        "correct_legacy_jpeg_bgr": args.correct_legacy_jpeg_bgr,
        "train": evaluate_episode(
            policy,
            args.train_demo,
            args.prompt,
            args.indices,
            correct_legacy_jpeg_bgr=args.correct_legacy_jpeg_bgr,
        ),
        "heldout": evaluate_episode(
            policy,
            args.heldout_demo,
            args.prompt,
            args.indices,
            correct_legacy_jpeg_bgr=args.correct_legacy_jpeg_bgr,
        ),
    }
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
