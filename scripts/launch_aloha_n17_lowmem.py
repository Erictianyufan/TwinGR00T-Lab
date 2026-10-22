"""Experimental small-batch N1.7 post-training without stopping other GPU jobs.

This adapts NVIDIA's Apache-2.0 ``launch_finetune.py`` configuration path. It
does not change the model or use demonstration replay during deployment.
NVIDIA recommends at least 40 GB of *available* GPU memory for fine-tuning;
this variant is a guarded feasibility probe, not a supported memory guarantee.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from pathlib import Path


def free_vram_mib() -> int:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True,
    )
    return min(int(line.strip()) for line in output.splitlines() if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--modality-config-path", required=True, type=Path)
    parser.add_argument("--base-model-path", default="nvidia/GR00T-N1.7-3B")
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument(
        "--optimizer",
        choices=("adamw_torch_fused", "adamw_bnb_8bit", "adafactor"),
        default="adamw_torch_fused",
    )
    parser.add_argument("--min-free-mib", type=int, default=26000)
    parser.add_argument("--state-dropout-prob", type=float, default=0.2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not (args.dataset_path / "meta/stats.json").is_file():
        parser.error("GR00T dataset statistics are missing")
    if not (args.dataset_path / "meta/relative_stats.json").is_file():
        parser.error("GR00T relative statistics are missing")
    if not args.modality_config_path.is_file():
        parser.error("The ALOHA modality config is missing")
    if (
        min(args.max_steps, args.save_steps, args.save_total_limit, args.batch_size, args.gradient_accumulation_steps)
        < 1
    ):
        parser.error("Training counts and batch settings must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Refusing to overwrite a nonempty training output directory")
    if not 0.0 <= args.state_dropout_prob < 1.0:
        parser.error("state-dropout-prob must be in [0, 1)")

    available = free_vram_mib()
    if available < args.min_free_mib:
        parser.error(
            f"Only {available} MiB GPU memory is free; need {args.min_free_mib} MiB. "
            "Existing GPU processes will not be stopped."
        )

    # The custom tag must be registered before the data/model config is built.
    spec = importlib.util.spec_from_file_location("aloha_n17_modality_config", args.modality_config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the ALOHA modality config")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    from gr00t.configs.base_config import get_default_config
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.experiment.experiment import run

    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": [str(args.dataset_path.resolve())],
                        "mix_ratio": 1.0,
                        "embodiment_tag": EmbodimentTag.NEW_EMBODIMENT.value,
                    }
                ],
            }
        }
    )
    config.load_config_path = None
    config.model.tune_llm = False
    config.model.tune_visual = False
    config.model.tune_projector = True
    config.model.tune_diffusion_model = True
    config.model.load_bf16 = True
    config.model.reproject_vision = False
    config.model.model_name = "nvidia/Cosmos-Reason2-2B"
    config.model.backbone_trainable_params_fp32 = False
    config.model.use_relative_action = True
    config.model.state_dropout_prob = args.state_dropout_prob

    config.training.start_from_checkpoint = args.base_model_path
    config.training.optim = args.optimizer
    config.training.num_gpus = 1
    config.training.global_batch_size = args.batch_size
    config.training.gradient_accumulation_steps = args.gradient_accumulation_steps
    config.training.gradient_checkpointing = True
    config.training.dataloader_num_workers = 1
    config.training.max_steps = args.max_steps
    config.training.save_steps = args.save_steps
    config.training.save_total_limit = args.save_total_limit
    config.training.save_only_model = True
    config.training.output_dir = str(args.output_dir.resolve())
    config.training.use_wandb = False

    summary = {
        "status": "dry_run" if args.dry_run else "starting",
        "dataset": str(args.dataset_path.resolve()),
        "output": str(args.output_dir.resolve()),
        "base_model": args.base_model_path,
        "available_vram_mib": available,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_steps": args.max_steps,
        "save_total_limit": args.save_total_limit,
        "optimizer": args.optimizer,
        "state_dropout_prob": args.state_dropout_prob,
        "trainable": ["projector", "action_head"],
        "experimental_low_memory": True,
    }
    print(json.dumps(summary, indent=2), flush=True)
    if not args.dry_run:
        run(config)


if __name__ == "__main__":
    main()
