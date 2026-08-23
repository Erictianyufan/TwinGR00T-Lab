"""Check gated GR00T repositories and optionally load the base policy."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "nvidia/GR00T-N1.7-3B"
DEFAULT_BACKBONE = "nvidia/Cosmos-Reason2-2B"


def _write_manifest(path: Path | None, result: dict[str, Any]) -> None:
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered, flush=True)


def _repo_access(repo_id: str) -> tuple[bool, str | None]:
    from huggingface_hub import HfApi, hf_hub_download

    try:
        info = HfApi().model_info(repo_id)
        # Gated repositories expose metadata anonymously. Downloading a small,
        # required file proves that the active token has actual artifact access.
        hf_hub_download(repo_id=repo_id, filename="config.json", revision=info.sha)
    except Exception as exc:  # Hugging Face exposes several auth/network exception types.
        return False, type(exc).__name__
    return True, info.sha


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--embodiment-tag", default="XDOF")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--access-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result: dict[str, Any] = {
        "status": "running",
        "phase": "access",
        "model": args.model_path,
        "backbone": args.backbone,
        "device": args.device,
        "weights_loaded": False,
    }

    access: dict[str, dict[str, Any]] = {}
    for label, repo_id in (("model", args.model_path), ("backbone", args.backbone)):
        allowed, detail = _repo_access(repo_id)
        access[label] = {"repo_id": repo_id, "allowed": allowed}
        if allowed:
            access[label]["revision"] = detail
        else:
            access[label]["error_type"] = detail
    result["access"] = access

    if not all(entry["allowed"] for entry in access.values()):
        result.update(
            status="blocked",
            reason="huggingface_access_required",
            remediation=(
                "Verify the Hugging Face account, accept access for the gated backbone, "
                "then run `hf auth login` interactively."
            ),
        )
        _write_manifest(args.output, result)
        return 2

    if args.access_only:
        result.update(status="pass", phase="access")
        _write_manifest(args.output, result)
        return 0

    try:
        import torch
        from gr00t.policy import Gr00tPolicy

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        torch.cuda.reset_peak_memory_stats(args.device)
        started = time.perf_counter()
        policy = Gr00tPolicy(
            model_path=args.model_path,
            embodiment_tag=args.embodiment_tag,
            device=args.device,
            strict=True,
        )
        elapsed = time.perf_counter() - started
        modality = policy.get_modality_config()
        result.update(
            status="pass",
            phase="load",
            weights_loaded=True,
            load_seconds=round(elapsed, 3),
            peak_vram_gib=round(torch.cuda.max_memory_allocated(args.device) / 1024**3, 3),
            modality_keys=sorted(modality),
        )
    except BaseException as exc:
        result.update(
            status="fail",
            phase="load",
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        _write_manifest(args.output, result)
        return 1

    _write_manifest(args.output, result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
