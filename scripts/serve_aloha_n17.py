"""Serve authentic GR00T N1.7 with explicit inference-sampling settings."""

from __future__ import annotations

import argparse
from pathlib import Path

from twingr00t_lab.n17_sampling import configure_n17_sampling


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--flow-steps", type=int, default=4)
    parser.add_argument("--sample-average", type=int, default=1)
    args = parser.parse_args()
    if not args.model_path.is_dir():
        parser.error("The GR00T checkpoint directory does not exist")

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.policy.server_client import PolicyServer

    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        model_path=str(args.model_path),
        device="cuda:0",
    )
    configure_n17_sampling(policy, flow_steps=args.flow_steps, sample_average=args.sample_average)
    print(
        f"[n17] model={args.model_path} flow_steps={args.flow_steps} "
        f"sample_average={args.sample_average}",
        flush=True,
    )
    with PolicyServer(policy=policy, host=args.host, port=args.port) as server:
        server.run()


if __name__ == "__main__":
    main()
