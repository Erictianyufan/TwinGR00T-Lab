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
import math
import subprocess
from pathlib import Path
from typing import Any

COLOR_JITTER_KEYS = ("brightness", "contrast", "saturation", "hue")


def free_vram_mib() -> int:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True,
    )
    return min(int(line.strip()) for line in output.splitlines() if line.strip())


def inspect_checkpoint_processor(checkpoint: str) -> dict:
    """Read the action contract from the processor, not the model config.

    N1.7's fine-tuning pipeline overrides the processor's relative-action flag
    independently of model.config. A false flag in final_model_config.json
    therefore does not establish that the learned actions are absolute.
    """
    root = Path(checkpoint).expanduser()
    if not root.exists():
        if root.is_absolute() or checkpoint.startswith("."):
            raise FileNotFoundError(f"Local starting checkpoint does not exist: {root}")
        return {"source": checkpoint, "status": "remote_not_inspected", "use_relative_action": None}
    # Match Gr00tPolicy's precedence when both layouts are present.
    config_path = root / "processor_config.json"
    if not config_path.is_file():
        config_path = root / "processor" / "processor_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Starting checkpoint has no processor_config.json: {root}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    processor = payload.get("processor_kwargs")
    if not isinstance(processor, dict):
        raise ValueError(f"Invalid processor_kwargs in {config_path}")
    # Older processor files omit the flag; the pinned N1.7 loader defaults to False.
    relative = processor.get("use_relative_action", False)
    if not isinstance(relative, bool):
        raise ValueError(f"use_relative_action must be a JSON boolean in {config_path}")
    return {
        "source": str(config_path.resolve()),
        "status": "inspected",
        "use_relative_action": relative,
        "relative_flag_explicit": "use_relative_action" in processor,
        "max_action_horizon": processor.get("max_action_horizon"),
        "state_dropout_prob": processor.get("state_dropout_prob"),
        "visual_augmentation": {
            key: processor.get(key)
            for key in (
                "color_jitter_params",
                "random_rotation_angle",
                "crop_fraction",
                "image_crop_size",
                "image_target_size",
                "use_albumentations",
            )
        },
    }


def configure_visual_augmentation(model_config: Any, mode: str, starting_processor: dict) -> dict:
    """Use only the pinned N1.7 processor's supported checkpoint overrides.

    ``setup._create_dataset`` forwards color_jitter_params and the processor's
    ``from_pretrained`` override_keys accepts it. None means "keep the saved
    value", so disabling color jitter requires an explicit non-None zero dict.
    crop_fraction is NOT in override_keys: changing only model_config would
    silently leave the saved crop active. Do not advertise this as "no aug".
    """
    if mode not in ("inherit", "color-off"):
        raise ValueError(f"Unknown visual augmentation mode: {mode}")
    override = None
    if mode == "color-off":
        override = dict.fromkeys(COLOR_JITTER_KEYS, 0.0)
        model_config.color_jitter_params = override.copy()
    return {
        "mode": mode,
        "color_jitter_override": override,
        "starting_checkpoint_visual_config": starting_processor.get("visual_augmentation"),
        "crop_and_rotation": "inherited_not_overridden",
        "scope": "color jitter only; spatial augmentation and model dropout are unchanged",
        "verification": "saved effective processor checked after training; not verified by dry-run",
    }


def verify_saved_visual_augmentation(output_dir: Path, mode: str) -> dict:
    """Check the effective serialized processor, never model config metadata."""
    if mode not in ("inherit", "color-off"):
        raise ValueError(f"Unknown visual augmentation mode: {mode}")
    contract = inspect_checkpoint_processor(str(output_dir))
    visual = contract.get("visual_augmentation", {})
    actual = visual.get("color_jitter_params")
    colors_disabled = isinstance(actual, dict) and all(actual.get(key) == 0.0 for key in COLOR_JITTER_KEYS)
    if mode == "color-off" and not colors_disabled:
        raise RuntimeError(
            "Requested color-off did not reach the saved effective processor: "
            f"{contract['source']} has color_jitter_params={actual!r}. "
            "Do not describe this run as color-augmentation-disabled."
        )
    return {
        "mode": mode,
        "status": "verified_color_off" if mode == "color-off" else "inherit_recorded",
        "effective_processor_source": contract["source"],
        "effective_visual_config": visual,
        "crop_and_rotation": "inherited_not_overridden",
    }


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
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--training-episode-id",
        type=int,
        help=(
            "Sample only this metadata episode_index; retain full-dataset stats "
            "(single-case overfit, not generalization)"
        ),
    )
    parser.add_argument(
        "--optimizer",
        choices=("adamw_torch_fused", "adamw_bnb_8bit", "adafactor"),
        default="adamw_torch_fused",
    )
    parser.add_argument("--min-free-mib", type=int, default=26000)
    parser.add_argument("--state-dropout-prob", type=float, default=0.2)
    parser.add_argument(
        "--visual-augmentation",
        choices=("inherit", "color-off"),
        default="inherit",
        help="Keep checkpoint augmentation, or disable only color jitter; crop/rotation remain inherited",
    )
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
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning-rate must be finite and positive")
    try:
        starting_processor = inspect_checkpoint_processor(args.base_model_path)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    sampling = {"mode": "all_episodes"}
    if args.training_episode_id is not None:
        from twingr00t_lab.training_selection import selection_plan

        try:
            sampling = selection_plan(args.dataset_path, args.training_episode_id)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))

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
    visual_augmentation = configure_visual_augmentation(config.model, args.visual_augmentation, starting_processor)

    config.training.start_from_checkpoint = args.base_model_path
    config.training.optim = args.optimizer
    config.training.learning_rate = args.learning_rate
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
        "learning_rate": args.learning_rate,
        "state_dropout_prob": args.state_dropout_prob,
        "visual_augmentation": visual_augmentation,
        "starting_checkpoint_processor": starting_processor,
        "requested_training_use_relative_action": config.model.use_relative_action,
        "effective_processor_config": str(args.output_dir.resolve() / "processor" / "processor_config.json"),
        "training_initialization": "checkpoint_weights_with_new_optimizer_and_schedule",
        "training_sampling": sampling,
        "trainable": ["projector", "action_head"],
        "experimental_low_memory": True,
    }
    print(json.dumps(summary, indent=2), flush=True)
    if not args.dry_run:
        restore_selection = None
        if args.training_episode_id is not None:
            import gr00t.data.dataset.factory as dataset_factory

            from twingr00t_lab.training_selection import install_factory_selection

            restore_selection = install_factory_selection(
                dataset_factory, sampling, args.output_dir.resolve() / "episode_selection_audit.json"
            )
        try:
            run(config)
        finally:
            if restore_selection is not None:
                restore_selection()
        if args.training_episode_id is not None and not (args.output_dir / "episode_selection_audit.json").is_file():
            raise RuntimeError("Training returned without verifying the requested episode sampling subset")
        visual_audit = verify_saved_visual_augmentation(args.output_dir, args.visual_augmentation)
        audit_path = args.output_dir / "visual_augmentation_audit.json"
        audit_path.write_text(json.dumps(visual_audit, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"visual_augmentation_audit": visual_audit, "audit_path": str(audit_path)}, indent=2))


if __name__ == "__main__":
    main()
