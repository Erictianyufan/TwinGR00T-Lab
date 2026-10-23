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
from twingr00t_lab.seeding import configure_policy_seed, validate_seed


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
    observation_action_offset: int = 0,
    repeats: int = 1,
) -> list[dict]:
    if observation_action_offset < 0:
        raise ValueError("observation_action_offset must be non-negative")
    if any(index < 0 for index in indices):
        raise ValueError("Observation indices must be non-negative")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    rows: list[dict] = []
    with h5py.File(path, "r") as handle:
        observed = handle["observations/policy"]
        cameras = handle["observations/camera_obs"]
        targets = collapse_isaac_action(np.asarray(observed["actions"][:], dtype=np.float32))
        if observation_action_offset >= len(targets):
            raise ValueError("Action offset leaves no paired frames in the episode")
        for index in indices:
            target_start = index + observation_action_offset
            # Evaluate complete 16-step windows only. Equality is valid: the
            # final target is len(targets)-1, and no next episode is ever read.
            if target_start + 16 > len(targets):
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
            samples = []
            for _ in range(repeats):
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
                if predicted.shape != (16, 14) or not np.isfinite(predicted).all():
                    raise ValueError(f"Expected 16 finite 14-D predictions, got {predicted.shape}")
                samples.append(predicted)
            predictions = np.stack(samples)
            actual = targets[target_start : target_start + predictions.shape[1]]
            # Average errors, not predictions: opposite errors must not cancel.
            errors = np.abs(predictions - actual[None])
            arm_dims = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
            gripper_dims = [6, 13]
            # Population spread of repeated calls on the identical observation.
            # With one call this is zero, but is explicitly not a variance estimate.
            prediction_std = predictions.std(axis=0, ddof=0)
            prediction_variance = predictions.var(axis=0, ddof=0)
            rows.append(
                {
                    "frame": index,
                    "target_frame": target_start,
                    "observation_action_offset": observation_action_offset,
                    "compared_horizon": predictions.shape[1],
                    "repeat_count": repeats,
                    "mae_14d": float(errors.mean()),
                    "arm_mae_rad": float(errors[..., arm_dims].mean()),
                    "gripper_mae_m": float(errors[..., gripper_dims].mean()),
                    "first_step_mae_14d": float(errors[:, 0].mean()),
                    "first_step_arm_mae_rad": float(errors[:, 0, arm_dims].mean()),
                    "first_step_gripper_mae_m": float(errors[:, 0, gripper_dims].mean()),
                    "first_step_abs_error_14d": errors[:, 0].mean(axis=0).tolist(),
                    "variance_estimated": repeats > 1,
                    "prediction_variance_14d": prediction_variance.mean(axis=0).tolist(),
                    "prediction_std_14d": prediction_std.mean(axis=0).tolist(),
                    "prediction_std_arm_rad": float(prediction_std[:, arm_dims].mean()),
                    "prediction_std_gripper_m": float(prediction_std[:, gripper_dims].mean()),
                    "first_step_prediction_std_14d": prediction_std[0].tolist(),
                    "first_prediction": predictions[0, 0].round(3).tolist(),
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
    parser.add_argument(
        "--observation-action-offset",
        type=int,
        default=0,
        help="Compare observation i with teacher actions starting at i+offset",
    )
    parser.add_argument("--flow-steps", type=int, default=4)
    parser.add_argument("--sample-average", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Independent policy calls per identical observation for sampling-spread diagnostics",
    )
    args = parser.parse_args()
    if args.observation_action_offset < 0 or any(index < 0 for index in args.indices):
        parser.error("Observation/action offset and observation indices must be non-negative")
    if args.repeats < 1:
        parser.error("--repeats must be a positive integer")
    try:
        validate_seed(args.seed)
    except ValueError as exc:
        parser.error(str(exc))

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    seed_metadata = configure_policy_seed(args.seed)
    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        model_path=str(args.checkpoint),
        device=args.device,
    )
    configure_n17_sampling(policy, flow_steps=args.flow_steps, sample_average=args.sample_average)
    result = {
        **seed_metadata,
        "checkpoint": str(args.checkpoint),
        "flow_steps": args.flow_steps,
        "sample_average": args.sample_average,
        "repeats": args.repeats,
        "error_aggregation": "mean_absolute_error_over_independent_predictions",
        "prediction_spread": "population_std_over_identical_observation_calls",
        "mae_14d_units": "mixed_arm_radians_and_gripper_meters_use_grouped_metrics",
        "correct_legacy_jpeg_bgr": args.correct_legacy_jpeg_bgr,
        "observation_action_offset": args.observation_action_offset,
        "requested_indices": args.indices,
        "compared_horizon": 16,
        "incomplete_windows": "skipped",
        "train": evaluate_episode(
            policy,
            args.train_demo,
            args.prompt,
            args.indices,
            correct_legacy_jpeg_bgr=args.correct_legacy_jpeg_bgr,
            observation_action_offset=args.observation_action_offset,
            repeats=args.repeats,
        ),
        "heldout": evaluate_episode(
            policy,
            args.heldout_demo,
            args.prompt,
            args.indices,
            correct_legacy_jpeg_bgr=args.correct_legacy_jpeg_bgr,
            observation_action_offset=args.observation_action_offset,
            repeats=args.repeats,
        ),
    }
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
