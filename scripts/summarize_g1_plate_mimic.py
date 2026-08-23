"""Create an evidence manifest from fixed-attempt G1 plate Mimic trials."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from twingr00t_lab.mimic_results import summarize_fixed_mimic_trials


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", nargs=3, action="append", metavar=("SEED", "HDF5", "LOG"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trials = [(int(seed), Path(dataset), Path(log)) for seed, dataset, log in args.trial]
    result = summarize_fixed_mimic_trials(trials)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
