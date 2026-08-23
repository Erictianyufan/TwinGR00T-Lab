"""Combine per-seed G1 JSON manifests into one experiment summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from twingr00t_lab.aggregation import aggregate_g1_manifests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in args.manifests]
    summary = aggregate_g1_manifests(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
