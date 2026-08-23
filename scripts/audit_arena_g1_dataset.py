"""Write a strict JSON audit for an IsaacLab-Arena G1 Mimic dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from twingr00t_lab.arena_dataset import audit_arena_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_arena_dataset(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
