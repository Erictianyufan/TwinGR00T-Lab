"""Derive a G1 plate seed by shifting annotated wrist targets in object space."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from twingr00t_lab.plate_seed import shift_targets_in_object_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--local-offset-xyz", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument(
        "--target-frame-rotation-y-deg",
        type=float,
        default=0.0,
        help="Rotate source wrist targets in object space before applying the local offset.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            shift_targets_in_object_frame(
                args.source,
                args.output,
                local_offset_xyz=tuple(args.local_offset_xyz),
                target_frame_rotation_y_deg=args.target_frame_rotation_y_deg,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
