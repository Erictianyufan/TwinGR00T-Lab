"""Assemble new, unverified public RoboTwin episodes for Isaac replay."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from prepare_public_factory_replay import prepare


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()
    if args.start_episode < 0 or args.count < 1 or args.start_episode + args.count > 50:
        parser.error("Select 1-50 episodes within public clean-50")
    if args.output_dir.exists():
        parser.error(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{args.output_dir.name}-", dir=args.output_dir.parent) as raw:
        root = Path(raw)
        combined = root / "combined"
        for name in ("data", "instructions", "reports"):
            (combined / name).mkdir(parents=True)
        reports = []
        for episode in range(args.start_episode, args.start_episode + args.count):
            candidate = root / f"candidate-{episode}"
            report = prepare(args.archive, args.template, candidate, episode)
            for folder, extension in (("data", "hdf5"), ("instructions", "json"), ("reports", "json")):
                shutil.copy2(candidate / folder / f"demo_{episode}.{extension}", combined / folder)
            reports.append(report)
            print(f"[candidate only, not accepted] public episode {episode} prepared", flush=True)
        (combined / "candidate_manifest.json").write_text(
            json.dumps(
                {
                    "status": "candidate_teacher_replays_not_physically_verified",
                    "public_episodes": list(range(args.start_episode, args.start_episode + args.count)),
                    "training_eligible": False,
                    "pickle_loaded": False,
                    "reports": reports,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        combined.rename(args.output_dir)


if __name__ == "__main__":
    main()
