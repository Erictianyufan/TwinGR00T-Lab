# Cloud deployment runbook

This runbook targets the validated host shape: Ubuntu 22.04, two RTX 4090 GPUs,
128 GB RAM, and roughly 500 GB of local disk. It deliberately keeps two tracks:

- strict reproduction: `/home/ubuntu/env_robotwin`, Isaac Sim 5.1, Python 3.11;
- migration experiment: `/home/ubuntu/env_isaaclab`, preinstalled Isaac Sim 6.0.1.

Never install strict-track packages into the preinstalled Sim 6 environment.
The upstream RoboTwin `IsaacLab-Arena` snapshot is tied to Isaac Sim 5.1 and an
Isaac Lab 2.3 development snapshot; Sim 6 is a separate compatibility study.

## Validated layout

```text
/home/ubuntu/mycode/robotwin-gr00t-lab/       experiment harness
  vendor/RoboTwin/                            pinned upstream checkout
  results/manifests/                          machine-readable checks
/home/ubuntu/env_robotwin/                    strict simulator environment
/home/ubuntu/env_groot/                       GR00T N1.7 environment (separate)
/home/ubuntu/env_isaaclab/                    provider Sim 6 image (preserved)
```

## Bootstrap GR00T N1.7

After the strict simulator smoke test passes:

```bash
bash scripts/bootstrap_groot.sh
/home/ubuntu/env_groot/bin/python -c \
  "import gr00t, torch; print(torch.__version__, torch.cuda.device_count())"
```

This uses the upstream lockfile with Python 3.12, PyTorch 2.9 and CUDA 12.8
wheels. The source snapshot skips Git LFS demonstration media and optional
LIBERO/RoboCasa/SimplerEnv submodules because TwinGR00T Lab supplies its own
pinned RoboTwin integration.

Downloading model weights is intentionally separate. First accept access to the
gated `nvidia/Cosmos-Reason2-2B` model and authenticate with Hugging Face; then
run the weight smoke test. Never place the token in a script or manifest.

```bash
/home/ubuntu/env_groot/bin/hf auth login
cd /home/ubuntu/mycode/robotwin-gr00t-lab/vendor/Isaac-GR00T-N1.7
/home/ubuntu/env_groot/bin/python ../../scripts/smoke_groot.py \
  --device cuda:1 \
  --output ../../results/manifests/groot-weights.json
```

The first command prompts for the token without placing it in this repository.
The smoke test checks access to both gated repositories before downloading, then
loads the official base policy on GPU 1 and records load time and peak VRAM.

If the cloud host's GitHub Release CDN is slow, download the exact FlashAttention
wheel on a faster machine, copy it to the host, and run:

```bash
GROOT_FLASH_ATTN_WHEEL=/tmp/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl \
  bash scripts/bootstrap_groot.sh
```

This mode still exports every other exact dependency from the upstream lockfile;
it only replaces the transport used for that same FlashAttention wheel.

## Bootstrap the strict environment

From the project root:

```bash
bash scripts/bootstrap_robotwin.sh
source /home/ubuntu/env_robotwin/bin/activate
python scripts/smoke_arena.py \
  --headless --device cuda:0 --steps 5 \
  --output results/manifests/arena-smoke.json \
  --environment manip_eval_tasks.examples.manipulation.stack_bowls_three_environment:StackBowlsThreeEnvironment \
  stack_bowls_three --embodiment aloha --enable_cameras True
```

Isaac Sim should see both GPUs. Do not set `CUDA_VISIBLE_DEVICES`; it can confuse
Vulkan device enumeration. `--device cuda:0` is the simulator assignment.
Headless GLFW/default-display warnings are expected when no X server is running.

## Pinned runtime

| Component | Strict value |
| --- | --- |
| RoboTwin | `e19d288299e042b25d7c1eb116e0ac2bd9599250` |
| IsaacLab-Arena | `e25ccf554da92ccb19d3b08aeec43978746eac0b` |
| Nested Isaac Lab | `6acdd82a1633732d32bb575e3d792e34fdeb437e` |
| Isaac Sim | 5.1.0 |
| Python | 3.11 |
| PyTorch | 2.7.0+cu128 |
| torchvision | 0.22.0+cu128 |

The separate GR00T environment follows its own upstream lock: Python 3.12,
PyTorch 2.9.0 and CUDA 12.8 wheels. These versions must not be merged with the
simulator environment.

The bootstrap includes two upstream packaging workarounds discovered during the
deployment: `setuptools==80.10.2` keeps `pkg_resources` available for
`flatdict==4.0.1`, and the Isaac Lab core editable package is installed without
dependency resolution after the official installer has populated its runtime.

## Acceptance checks

Run checks in this order and retain every JSON manifest:

1. `bash scripts/doctor.sh` creates `results/manifests/host.json`.
2. `twingr00t-audit ...` verifies finite 14-D state/action and three policy cameras.
3. Five simulator steps prove imports and scene construction.
4. Fifty steps catch basic camera/physics instability.
5. Five hundred steps complete gate G0.

The first five downloaded public demonstrations are only a pipeline smoke set;
they are not sufficient for post-training. Download and audit all 100 episodes
before conversion or training.

After converting all episodes and generating `stats.json` plus
`relative_stats.json`, launch training on a 40 GB+ GPU host with:

```bash
DATASET_PATH=/path/to/data/lerobot/stack_bowls_three \
OUTPUT_DIR=/path/to/checkpoints/stack_bowls_three \
MAX_STEPS=2000 GLOBAL_BATCH_SIZE=32 NUM_GPUS=1 \
  bash scripts/run_finetune.sh
```

The launcher rejects GPUs below the official 40 GB minimum by default. DDP on
the two 24 GB 4090s duplicates the model and does not combine their memory, so
those cards remain assigned to simulator plus inference/closed-loop evaluation.

The complete split preparation is resumable and can be run unattended:

```bash
nohup bash scripts/prepare_splits.sh > results/logs/prepare-splits.log 2>&1 &
```

It requires exactly 100 source demonstrations, creates episode-disjoint
`train`/`val`/`test` datasets (80/10/10), and computes GR00T absolute and
relative statistics independently for every split.

Before gated weights are available, validate the process boundary with the
official GR00T `ReplayPolicy`: launch `gr00t/eval/run_gr00t_server.py` against
the converted smoke dataset on localhost, then run:

```bash
/home/ubuntu/env_robotwin/bin/python scripts/smoke_policy_transport.py \
  vendor/RoboTwin/data/stack_bowls_three/data/demo_0.hdf5 \
  --output results/manifests/policy-transport-smoke.json
```

The client is deliberately numeric-only, rejects object-dtype msgpack arrays,
and keeps GR00T model dependencies out of the simulator environment.

## GPU process split

Run Isaac Sim on GPU 0 and the GR00T policy server on GPU 1. The 24 GB card can
serve N1.7 with inference-oriented precision, but official full post-training is
not a safe fit. Submit post-training to a separate 48/80 GB GPU job and copy only
the resulting checkpoint back for closed-loop evaluation.

## Secrets

Supply SSH and Hugging Face credentials interactively or through the cloud
provider's secret store. Do not put them in this repository, shell scripts,
manifests, or command history.
