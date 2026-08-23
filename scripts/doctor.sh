#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

python - <<'PY'
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path


def command_output(*args: str) -> str | None:
    if shutil.which(args[0]) is None:
        return None
    return subprocess.check_output(args, text=True).strip()


gpu_lines = command_output(
    "nvidia-smi",
    "--query-gpu=index,name,memory.total,driver_version",
    "--format=csv,noheader",
)
manifest = {
    "hostname": platform.node(),
    "os": platform.platform(),
    "python": platform.python_version(),
    "gpus": gpu_lines.splitlines() if gpu_lines else [],
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "disk_root": shutil.disk_usage("/")._asdict(),
}

output = Path("results/manifests/host.json")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(json.dumps(manifest, indent=2))
print(f"wrote {output}")
PY

cd "$project_root"
