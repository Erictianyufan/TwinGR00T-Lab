"""Create an experimental N1.7 checkpoint between two compatible checkpoints.

This is parameter interpolation, not extra training or a claim of task success.
The destination must be new. Inputs are read as safetensors, never pickle.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def interpolate(first: Path, second: Path, output: Path, alpha: float) -> dict:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    indexes = [
        json.loads((path / "model.safetensors.index.json").read_text(encoding="utf-8"))
        for path in (first, second)
    ]
    if indexes[0]["weight_map"] != indexes[1]["weight_map"]:
        raise ValueError("The N1.7 checkpoints have different tensor-to-shard maps")
    shards = sorted(set(indexes[0]["weight_map"].values()))
    input_bytes = sum((second / name).stat().st_size for name in shards)
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < input_bytes + 1_000_000_000:
        raise OSError("Not enough free space for one complete output checkpoint")

    output.mkdir()
    for path in second.iterdir():
        destination = output / path.name
        if path.is_file() and path.name not in shards:
            shutil.copy2(path, destination)
        elif path.is_dir():
            shutil.copytree(path, destination)

    counts = {}
    for shard in shards:
        names = sorted(name for name, mapped in indexes[0]["weight_map"].items() if mapped == shard)
        tensors = {}
        with safe_open(first / shard, framework="pt", device="cpu") as left:
            with safe_open(second / shard, framework="pt", device="cpu") as right:
                if set(left.keys()) != set(right.keys()) or set(left.keys()) != set(names):
                    raise ValueError(f"Tensor names differ in {shard}")
                for name in names:
                    old = left.get_tensor(name)
                    new = right.get_tensor(name)
                    if old.shape != new.shape or old.dtype != new.dtype:
                        raise ValueError(f"Tensor shape/dtype differs: {name}")
                    if old.is_floating_point():
                        tensors[name] = torch.lerp(old.float(), new.float(), alpha).to(dtype=old.dtype).contiguous()
                    else:
                        if not torch.equal(old, new):
                            raise ValueError(f"Non-floating tensor differs: {name}")
                        tensors[name] = old.contiguous()
        save_file(tensors, output / shard)
        with safe_open(output / shard, framework="pt", device="cpu") as saved:
            if set(saved.keys()) != set(names):
                raise ValueError(f"Written shard is incomplete: {shard}")
        counts[shard] = len(names)
        print(f"[interpolate] wrote {shard}: {len(names)} tensors", flush=True)
        del tensors

    provenance = {
        "status": "experimental_parameter_interpolation_not_task_validation",
        "first": str(first.resolve()),
        "second": str(second.resolve()),
        "alpha_second": alpha,
        "formula": "output=(1-alpha_second)*first+alpha_second*second",
        "shard_tensor_counts": counts,
        "output": str(output.resolve()),
    }
    (output / "interpolation.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", required=True, type=Path)
    parser.add_argument("--second", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--alpha", required=True, type=float)
    args = parser.parse_args()
    print(json.dumps(interpolate(args.first, args.second, args.output, args.alpha), indent=2))


if __name__ == "__main__":
    main()
