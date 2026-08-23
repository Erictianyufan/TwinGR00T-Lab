#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck disable=SC1090
source "$project_root/configs/versions.env"
cd "$project_root"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required; install it before running this script" >&2
    exit 1
fi

env_dir=${GROOT_ENV_DIR:-/home/ubuntu/env_groot}
vendor_dir=${GROOT_VENDOR_DIR:-$project_root/vendor/Isaac-GR00T-N1.7}

if [[ ! -f "$vendor_dir/pyproject.toml" ]]; then
    if [[ -e "$vendor_dir" ]]; then
        echo "incomplete GR00T source directory exists: $vendor_dir" >&2
        exit 1
    fi
    archive=$(mktemp --suffix=-isaac-groot.tar.gz)
    unpack_dir=$(mktemp -d)
    cleanup() {
        rm -f -- "$archive"
        rm -rf -- "$unpack_dir"
    }
    trap cleanup EXIT

    curl -fL --retry 3 \
        "https://codeload.github.com/NVIDIA/Isaac-GR00T/tar.gz/$GROOT_REF" \
        -o "$archive"
    tar -xzf "$archive" --strip-components=1 -C "$unpack_dir"
    mkdir -p "$(dirname -- "$vendor_dir")"
    mv -- "$unpack_dir" "$vendor_dir"
    trap - EXIT
    rm -f -- "$archive"
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y ffmpeg
fi

if [[ -n "${GROOT_FLASH_ATTN_WHEEL:-}" ]]; then
    if [[ ! -f "$GROOT_FLASH_ATTN_WHEEL" ]]; then
        echo "flash-attn wheel not found: $GROOT_FLASH_ATTN_WHEEL" >&2
        exit 1
    fi
    if [[ ! -x "$env_dir/bin/python" ]]; then
        uv venv --python "$GROOT_PYTHON_VERSION" "$env_dir"
    fi
    py="$env_dir/bin/python"
    uv pip install --python "$py" --no-deps "$GROOT_FLASH_ATTN_WHEEL"
    uv pip install --python "$py" \
        --index-url "https://download.pytorch.org/whl/cu128" \
        "torch==$GROOT_TORCH_VERSION" "torchvision==$GROOT_TORCHVISION_VERSION"
    (
        cd "$vendor_dir"
        uv pip install --python "$py" -r <(
            uv export --frozen --no-hashes --no-emit-project --no-annotate \
                | grep -Ev '^(flash-attn|torch==|torchvision==|triton==)'
        )
        uv pip install --python "$py" --no-deps -e .
    )
else
    (
        cd "$vendor_dir"
        UV_PROJECT_ENVIRONMENT="$env_dir" uv sync --python "$GROOT_PYTHON_VERSION" --frozen
    )
fi

TWIN_GROOT_REF="$GROOT_REF" TWIN_GROOT_OUTPUT="$project_root/results/manifests/groot-imports.json" \
    "$env_dir/bin/python" - <<'PY'
import json
import os
from pathlib import Path

import gr00t
import flash_attn
import torch

manifest = {
    "status": "pass",
    "source_ref": os.environ["TWIN_GROOT_REF"],
    "gr00t_module": gr00t.__name__,
    "python": os.sys.version.split()[0],
    "torch": torch.__version__,
    "flash_attn": flash_attn.__version__,
    "cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu_count": torch.cuda.device_count(),
    "gpus": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
    "weights_loaded": False,
}
output = Path(os.environ["TWIN_GROOT_OUTPUT"])
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(json.dumps(manifest, indent=2))
PY

uv pip check --python "$env_dir/bin/python"

echo "GR00T environment ready at $env_dir; gated model weights are not downloaded"
