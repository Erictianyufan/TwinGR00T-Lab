#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
versions_file="$project_root/configs/versions.env"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required; install it before running this script" >&2
    exit 1
fi

# shellcheck disable=SC1090
source "$versions_file"
cd "$project_root"

env_dir=${ROBOTWIN_ENV_DIR:-/home/ubuntu/env_robotwin}
vendor_dir=${ROBOTWIN_VENDOR_DIR:-$project_root/vendor/RoboTwin}
isaaclab_dir="$vendor_dir/submodule/isaaclab_arena/submodules/IsaacLab"

mkdir -p "$(dirname -- "$vendor_dir")"
if [[ ! -d "$vendor_dir/.git" ]]; then
    git clone --branch "$ROBOTWIN_BRANCH" "$ROBOTWIN_REPO" "$vendor_dir"
fi

git -C "$vendor_dir" fetch origin "$ROBOTWIN_BRANCH"
git -C "$vendor_dir" checkout --detach "$ROBOTWIN_REF"

# The upstream tree contains nested SSH submodule URLs. Apply an ephemeral HTTPS
# rewrite to this process and every recursive Git child without changing user config.
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=url.https://github.com/.insteadOf
export GIT_CONFIG_VALUE_0=git@github.com:
git -C "$vendor_dir" submodule sync --recursive
git -C "$vendor_dir" submodule update --init --recursive

arena_ref=$(git -C "$vendor_dir/submodule/isaaclab_arena" rev-parse HEAD)
if [[ "$arena_ref" != "$ROBOTWIN_ARENA_REF" ]]; then
    echo "unexpected IsaacLab-Arena revision: $arena_ref" >&2
    exit 1
fi
nested_isaaclab_ref=$(git -C "$isaaclab_dir" rev-parse HEAD)
if [[ "$nested_isaaclab_ref" != "$ROBOTWIN_NESTED_ISAACLAB_REF" ]]; then
    echo "unexpected nested Isaac Lab revision: $nested_isaaclab_ref" >&2
    exit 1
fi

if [[ ! -x "$env_dir/bin/python" ]]; then
    uv venv --python "$ROBOTWIN_PYTHON_VERSION" "$env_dir"
fi

py="$env_dir/bin/python"
uv pip install --python "$py" \
    "isaacsim[all,extscache]==$ISAAC_SIM_VERSION" \
    --extra-index-url https://pypi.nvidia.com
uv pip install --python "$py" \
    "torch==$ROBOTWIN_TORCH_VERSION" \
    "torchvision==$ROBOTWIN_TORCHVISION_VERSION" \
    "torchaudio==$ROBOTWIN_TORCH_VERSION" \
    --index-url "https://download.pytorch.org/whl/$ROBOTWIN_CUDA_WHEEL"

uv pip install --python "$py" \
    "setuptools==$ROBOTWIN_SETUPTOOLS_VERSION" wheel "packaging==23.0"
uv pip install --python "$py" --no-build-isolation "flatdict==4.0.1"
uv pip install --python "$py" \
    "click==8.1.7" "psutil==5.9.8" "typing-extensions==4.12.2" \
    "gymnasium==1.2.1" "prettytable==3.3" hidapi "pyglet<2" einops \
    "warp-lang==1.16" onnxruntime vuer lightwheel-sdk pin-pink

# Install the pinned Isaac Lab packages and the two RoboTwin packages editable.
# Core is no-deps because flatdict's legacy build metadata otherwise fails under
# modern isolated build tooling; its runtime dependencies were installed above.
uv pip install --python "$py" --no-deps -e "$isaaclab_dir/source/isaaclab"
for package in isaaclab_assets isaaclab_tasks isaaclab_rl isaaclab_mimic; do
    uv pip install --python "$py" -e "$isaaclab_dir/source/$package"
done
uv pip install --python "$py" -e "$vendor_dir/submodule/isaaclab_arena"
uv pip install --python "$py" -e "$vendor_dir/source/manip_eval_tasks"
uv pip install --python "$py" -e "$project_root[data]"

"$py" - <<'PY'
import json
from pathlib import Path

import isaaclab
import isaaclab_arena
import isaacsim
import manip_eval_tasks
import torch

manifest = {
    "status": "pass",
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "gpu_count": torch.cuda.device_count(),
    "modules": [
        isaaclab.__name__,
        isaaclab_arena.__name__,
        isaacsim.__name__,
        manip_eval_tasks.__name__,
    ],
}
output = Path("results/manifests/robotwin-imports.json")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(json.dumps(manifest, indent=2))
PY

echo "strict RoboTwin environment ready at $env_dir"
