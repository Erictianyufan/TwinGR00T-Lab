"""Audit XPolicyLab RoboTwin episodes before an N1.7 data conversion.

The public action stream contains *commands*, not measured joint positions.
This audit deliberately refuses to label it as ``observation.state``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

CAMERAS = ("head_camera", "left_camera", "right_camera")
ACTION_KEY = "joint_action/vector"
DEFAULT_STATE_KEY = "observation/joint_state/vector"


def audit_episode(path: Path, *, state_key: str = DEFAULT_STATE_KEY) -> dict[str, Any]:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - packaging check
        raise RuntimeError("Install the data extra: pip install -e '.[data]'") from exc

    problems: list[str] = []
    with h5py.File(path, "r") as episode:
        if ACTION_KEY not in episode:
            raise ValueError(f"{path}: missing {ACTION_KEY}")
        actions = episode[ACTION_KEY]
        if actions.ndim != 2 or actions.shape[1] != 14:
            raise ValueError(f"{path}: {ACTION_KEY} must have shape (frames, 14)")
        frames = int(actions.shape[0])
        if frames == 0:
            raise ValueError(f"{path}: empty episode")
        if not np.isfinite(actions[:]).all():
            problems.append("non-finite joint actions")

        camera_shapes: dict[str, list[int]] = {}
        for camera in CAMERAS:
            key = f"observation/{camera}/rgb"
            if key not in episode:
                problems.append(f"missing camera {key}")
                continue
            camera_shapes[camera] = list(episode[key].shape)
            if episode[key].shape[0] != frames:
                problems.append(f"frame count mismatch in {key}")

        if state_key not in episode:
            problems.append(f"missing measured joint state {state_key}")
            state_shape = None
        else:
            state_shape = list(episode[state_key].shape)
            if state_shape != [frames, 14]:
                problems.append(f"invalid measured joint state shape {state_shape}")
            elif not np.isfinite(episode[state_key][:]).all():
                problems.append("non-finite measured joint state")

    return {
        "episode": path.name,
        "frames": frames,
        "action_shape": [frames, 14],
        "camera_shapes": camera_shapes,
        "measured_state_shape": state_shape,
        "data_contract_ready": not problems,
        "problems": problems,
    }


def audit_dataset(directory: Path, *, state_key: str = DEFAULT_STATE_KEY) -> dict[str, Any]:
    paths = sorted((directory / "data").glob("episode*.hdf5"))
    if not paths:
        raise ValueError(f"No episode*.hdf5 files under {directory / 'data'}")
    episodes = [audit_episode(path, state_key=state_key) for path in paths]
    missing_instructions = [
        row["episode"]
        for row in episodes
        if not (directory / "instructions" / f"{Path(row['episode']).stem}.json").is_file()
    ]
    return {
        "source": str(directory),
        "episodes": len(episodes),
        "frames": sum(row["frames"] for row in episodes),
        "measured_state_key": state_key,
        "data_contract_ready": all(row["data_contract_ready"] for row in episodes) and not missing_instructions,
        "missing_instructions": missing_instructions,
        "action_is_not_measured_state": True,
        "episode_details": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--state-key", default=DEFAULT_STATE_KEY)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_dataset(args.dataset_dir, state_key=args.state_key)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
