"""Audit RoboTwin IsaacLab-Arena demonstrations before GR00T conversion."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from twingr00t_lab.contracts import CANONICAL_ACTION_DIM, ISAAC_ACTION_DIM

POLICY_ROOT = "observations/policy"
CAMERA_ROOT = "observations/camera_obs"
CAMERAS = ("head_camera_rgb", "left_camera_rgb", "right_camera_rgb")


def collapse_isaac_action(actions: np.ndarray, *, atol: float = 1e-6) -> np.ndarray:
    """Collapse 16 physical Aloha joints into the canonical 14 logical controls."""
    values = np.asarray(actions, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != ISAAC_ACTION_DIM:
        raise ValueError(f"Expected actions shaped (T, {ISAAC_ACTION_DIM}), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Actions contain NaN or Inf")
    if not np.allclose(values[:, 6], values[:, 7], atol=atol, rtol=0):
        raise ValueError("Left physical gripper joints disagree")
    if not np.allclose(values[:, 14], values[:, 15], atol=atol, rtol=0):
        raise ValueError("Right physical gripper joints disagree")
    return np.concatenate((values[:, :6], values[:, 6:7], values[:, 8:14], values[:, 14:15]), axis=1)


def canonical_proprio(policy_group: object) -> np.ndarray:
    """Build measured 14D proprioception; never substitute the previous action."""
    parts = (
        np.asarray(policy_group["left_arm_pos"][:], dtype=np.float32),
        np.asarray(policy_group["left_gripper_pos"][:], dtype=np.float32),
        np.asarray(policy_group["right_arm_pos"][:], dtype=np.float32),
        np.asarray(policy_group["right_gripper_pos"][:], dtype=np.float32),
    )
    state = np.concatenate(parts, axis=1)
    if state.ndim != 2 or state.shape[1] != CANONICAL_ACTION_DIM:
        raise ValueError(f"Expected state shaped (T, {CANONICAL_ACTION_DIM}), got {state.shape}")
    if not np.isfinite(state).all():
        raise ValueError("State contains NaN or Inf")
    return state


@dataclass(frozen=True)
class DemoAudit:
    path: str
    frames: int
    state_dim: int
    action_dim: int
    cameras: list[str]


def audit_demo(path: str | Path) -> DemoAudit:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - exercised by packaging, not unit tests
        raise RuntimeError("Install the data extra: pip install -e '.[data]'") from exc

    path = Path(path)
    with h5py.File(path, "r") as handle:
        policy = handle[POLICY_ROOT]
        state = canonical_proprio(policy)
        actions = collapse_isaac_action(np.asarray(policy["actions"][:], dtype=np.float32))
        if state.shape[0] != actions.shape[0]:
            raise ValueError(f"State/action frame mismatch: {state.shape[0]} != {actions.shape[0]}")

        camera_group = handle[CAMERA_ROOT]
        missing = [name for name in CAMERAS if name not in camera_group]
        if missing:
            raise ValueError(f"Missing policy cameras: {missing}")
        for name in CAMERAS:
            if len(camera_group[name]) != state.shape[0]:
                raise ValueError(f"Camera {name} frame count does not match state")

    return DemoAudit(
        path=str(path),
        frames=state.shape[0],
        state_dim=state.shape[1],
        action_dim=actions.shape[1],
        cameras=list(CAMERAS),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="RoboTwin demo HDF5 files")
    parser.add_argument("--output", type=Path, help="Optional JSON audit manifest")
    args = parser.parse_args()

    audits = [asdict(audit_demo(path)) for path in args.paths]
    payload = {"status": "pass", "demo_count": len(audits), "demos": audits}
    rendered = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
