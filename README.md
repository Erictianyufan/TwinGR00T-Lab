# TwinGR00T Lab

Reproducible RoboTwin 2.0 manipulation replay, GR00T N1.7 policy integration,
and presentation-grade multi-camera demos in NVIDIA Isaac Lab.

The current verified showcase focuses on ALOHA bimanual tasks. Stack demos are
accepted only after a post-action settling window verifies that every object is
centered, upright, and nearly stationary; a task's position-only success flag is
not sufficient. See [`docs/block-stability.md`](docs/block-stability.md).

The experimental **HomeRelay G1 + GR00T + RoboTwin** integration remains
available separately. It combines the official Unitree G1 articulation, an
ALOHA handoff, and bowl stacking in one Isaac Sim timeline; it is not included
in the verified ALOHA-only showcase. See
[`docs/g1_groot_merge.md`](docs/g1_groot_merge.md).

## Status

This repository is an experiment harness, not a claim that GR00T has reproduced RoboTwin results. A stage is marked complete only after its acceptance checks and artifacts exist.

Initial target:

- task: `stack_bowls_three`
- embodiment: Aloha AgileX
- simulator: Isaac Sim 5.1.0
- robot framework: Isaac Lab 2.3.x / Isaac Lab-Arena 0.1.1
- policy: `nvidia/GR00T-N1.7-3B`
- cameras: head, left wrist, right wrist
- canonical state/action: 14 dimensions

## Architecture

```text
GPU 0: RoboTwin + Isaac Sim + cameras
                 |
                 | policy request/response
                 v
GPU 1: GR00T N1.7 policy server

Separate 40 GB+ GPU job: GR00T post-training
```

The simulator and policy use separate Python environments. The preinstalled Isaac Sim 6 environment is retained for a later migration experiment and is not modified by the strict reproduction setup.

## Repository layout

```text
configs/                 pinned versions and experiment configuration
docs/                    experiment design and runbooks
scripts/                 remote diagnostics and deployment helpers
src/twingr00t_lab/       reusable data and policy contracts
tests/                   dependency-light contract tests
vendor/                  ignored upstream checkouts on the cloud host
artifacts/               ignored videos and model artifacts
results/                 small versionable metrics and manifests
```

Large datasets, checkpoints, generated videos, upstream repositories, and local
Isaac assets are intentionally excluded from Git. Each rendered artifact is
paired with a JSON manifest on the experiment host.

## Warehouse parcel-sorting showcase

`scripts/parcel_sorting_demo.py` loads the Isaac Digital Twin Warehouse and an
official conveyor asset, then presents a continuous three-parcel, two-bin ALOHA
sorting sequence with a cooperative handoff. The first version uses recorded
successful bimanual actions plus explicitly labelled deterministic parcel
choreography; it does not claim GR00T inference or conveyor surface physics.
See [`docs/parcel-sorting-demo.md`](docs/parcel-sorting-demo.md).

## Local checks

```bash
python -m pip install -e '.[dev]'
pytest
```

Audit downloaded demonstrations before conversion:

```bash
python -m pip install -e '.[data]'
twingr00t-audit vendor/RoboTwin/data/stack_bowls_three/data/demo_*.hdf5 \
  --output results/manifests/smoke-data.json
```

Convert a one-episode smoke set to GR00T-flavored LeRobot v2:

```bash
twingr00t-convert \
  --input-dir vendor/RoboTwin/data/stack_bowls_three \
  --output-dir data/lerobot/stack_bowls_three-smoke \
  --limit 1 --fps 31.25 \
  --manifest results/manifests/conversion-smoke.json
```

The converter writes 14-D measured state/action Parquet, three H.264 camera
streams, language metadata and the GR00T modality contract. It refuses to replace
an existing output directory.

Create leakage-safe episode datasets with source IDs 0-79, 80-89 and 90-99:

```bash
twingr00t-convert --input-dir vendor/RoboTwin/data/stack_bowls_three \
  --output-dir data/lerobot/stack_bowls_three-train --start-episode 0 --limit 80
twingr00t-convert --input-dir vendor/RoboTwin/data/stack_bowls_three \
  --output-dir data/lerobot/stack_bowls_three-val --start-episode 80 --limit 10
twingr00t-convert --input-dir vendor/RoboTwin/data/stack_bowls_three \
  --output-dir data/lerobot/stack_bowls_three-test --start-episode 90 --limit 10
```

Launch the strict RoboTwin Arena task on GPU 0:

```bash
source /home/ubuntu/env_robotwin/bin/activate
python scripts/smoke_arena.py \
  --headless --device cuda:0 --steps 50 \
  --environment manip_eval_tasks.examples.manipulation.stack_bowls_three_environment:StackBowlsThreeEnvironment \
  stack_bowls_three --embodiment aloha --enable_cameras True \
  --output results/manifests/arena-smoke.json
```

Do not set `CUDA_VISIBLE_DEVICES` for Isaac Sim; select the simulator GPU with  
`--device cuda:0`. The policy service will use GPU 1 from its separate process.

## Continuous home-service demo

The first `TwinGR00T HomeRelay` demo keeps a kitchen, living room, dining family, humanoid carrier, ALOHA workcell,
handoff, and bowl-stacking task in one continuous Isaac Sim timeline:

```bash
source /home/ubuntu/env_robotwin/bin/activate
python scripts/home_service_demo.py \
  vendor/RoboTwin/data/stack_bowls_three/data/demo_0.hdf5 \
  --output artifacts/videos/home_relay_demo_v1.mp4 \
  --headless --device cuda:0 --rendering_mode quality \
  --environment manip_eval_tasks.examples.manipulation.stack_bowls_three_environment:StackBowlsThreeEnvironment \
  stack_bowls_three --embodiment aloha --enable_cameras True
```

See [`docs/home_service_demo.md`](docs/home_service_demo.md) for the phase contract, acceptance criteria, and G1/GR00T
replacement boundary.

Install the isolated GR00T N1.7 runtime after the simulator check passes:

```bash
bash scripts/bootstrap_groot.sh
```

This completes a code/import smoke test without downloading gated model weights.

After verifying the Hugging Face account and accepting access to
`nvidia/Cosmos-Reason2-2B`, authenticate interactively and validate the base
policy on GPU 1:

```bash
/home/ubuntu/env_groot/bin/hf auth login
cd vendor/Isaac-GR00T-N1.7
/home/ubuntu/env_groot/bin/python ../../scripts/smoke_groot.py \
  --device cuda:1 --output ../../results/manifests/groot-weights.json
```

Use `scripts/run_finetune.sh` only on a GPU with at least 40 GB VRAM. The dual
4090 host is the intended closed-loop simulator/inference node, not a supported
full post-training node.

## Reproduction stages

1. Pin and validate the official RoboTwin IsaacLab-Arena environment.
2. Replay a five-demo smoke subset, then audit all 100 public demonstrations.
3. Convert RoboTwin HDF5 into GR00T-flavored LeRobot v2.
4. Run GR00T open-loop inference and a bounded post-training smoke test.
5. Evaluate a frozen checkpoint on held-out initial states and language prompts.

See `docs/experiment-design.md` for the protocol and acceptance gates and
`docs/deployment.md` for the cloud-host runbook.
