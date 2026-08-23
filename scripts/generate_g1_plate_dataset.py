"""Run Arena MimicGen with the local Galileo scene and serving-plate cargo.

The upstream generator has no CLI for replacing the background or manipulated
object.  This wrapper installs the same environment override used by the GR00T
evaluator, then executes the unmodified Arena generator.  The object keeps the
task-facing name ``brown_box`` so existing annotated subtask signals remain
compatible; generated episodes are still exported only when the simulator's
native success predicate passes.
"""

from __future__ import annotations

import os
from pathlib import Path


def main() -> None:
    cargo_value = os.environ.get("TWINGR00T_CARGO_USD")
    if not cargo_value:
        raise RuntimeError("Set TWINGR00T_CARGO_USD to the serving-plate USD")
    cargo_path = Path(cargo_value).resolve()
    if not cargo_path.is_file():
        raise FileNotFoundError(f"Serving-plate USD does not exist: {cargo_path}")

    scale = tuple(float(value) for value in os.environ.get("TWINGR00T_CARGO_SCALE", "1,1,1").split(","))
    if len(scale) != 3:
        raise ValueError("TWINGR00T_CARGO_SCALE must contain exactly three comma-separated values")

    upstream = (
        Path(__file__).resolve().parents[1]
        / "vendor"
        / "RoboTwin"
        / "submodule"
        / "isaaclab_arena"
        / "isaaclab_arena"
        / "scripts"
        / "generate_dataset.py"
    )
    if not upstream.is_file():
        raise FileNotFoundError(f"Arena generator does not exist: {upstream}")

    # Arena's script owns SimulationApp creation at module scope.  Inject the
    # environment patch immediately after that creation: importing Homie any
    # earlier fails because omni.client is not yet available, while launching a
    # second SimulationApp from a conventional wrapper is unsupported.
    source = upstream.read_text(encoding="utf-8")
    marker = "simulation_app = app_launcher.app"
    if source.count(marker) != 1:
        raise RuntimeError("Unexpected Arena generator layout; refusing an ambiguous source injection")
    injection = f"""{marker}

from run_g1_groot_closedloop import _use_local_homie_models, _use_simple_home_scene
_use_local_homie_models()
_use_simple_home_scene(Path({str(cargo_path)!r}), {scale!r})
"""
    source = source.replace(marker, injection)

    trial_marker = "env_cfg.datagen_config.generation_num_trials = generation_num_trials"
    if source.count(trial_marker) != 1:
        raise RuntimeError("Unexpected Arena trial configuration; refusing an ambiguous source injection")
    trial_injection = f"""{trial_marker}

    if __import__("os").environ.get("TWINGR00T_MIMIC_GUARANTEE", "1") == "0":
        env_cfg.datagen_config.generation_guarantee = False
        print("[TwinGR00T] Mimic fixed-attempt mode: failures count toward generation_num_trials")
    if __import__("os").environ.get("TWINGR00T_MIMIC_KEEP_FAILED", "0") == "1":
        env_cfg.datagen_config.generation_keep_failed = True
        print("[TwinGR00T] Mimic failed-episode export: enabled")
    _twingroot_seed = __import__("os").environ.get("TWINGR00T_MIMIC_SEED")
    if _twingroot_seed is not None:
        env_cfg.datagen_config.seed = int(_twingroot_seed)
        print(f"[TwinGR00T] Mimic seed: {{env_cfg.datagen_config.seed}}")
"""
    source = source.replace(trial_marker, trial_injection)
    namespace = {
        "__file__": str(upstream),
        "__name__": "__main__",
        "__package__": None,
        "Path": Path,
    }
    exec(compile(source, str(upstream), "exec"), namespace)


if __name__ == "__main__":
    main()
