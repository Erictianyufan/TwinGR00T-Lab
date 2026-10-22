<div align="center">

# TwinGR00T Lab

**RoboTwin 2.0 × GR00T N1.7 × NVIDIA Isaac Lab**

An evidence-first toolkit for reproducible bimanual manipulation experiments,
contact-physics validation, and multi-camera digital-twin demos.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-5.1-76B900?logo=nvidia&logoColor=white)](docs/deployment.md)
[![Policy](https://img.shields.io/badge/GR00T-N1.7--3B-FFB000)](docs/experiment-design.md)
[![Quality](https://img.shields.io/badge/quality-pytest%20%2B%20Ruff-2EA44F)](CONTRIBUTING.md#quality-gates)
[![Author](https://img.shields.io/badge/author-Erictianyufan-181717?logo=github)](https://github.com/Erictianyufan)

<img src="docs/media/parcel_sorting.gif" alt="ALOHA sorting parcels in an Isaac Sim warehouse" width="900" />

[Quick start](#quick-start) · [Architecture](docs/architecture.md) ·
[Experiment protocol](docs/experiment-design.md) · [Deployment](docs/deployment.md) ·
[Contributing](CONTRIBUTING.md) · [Citation](CITATION.cff)

</div>

## What this repository provides

TwinGR00T Lab connects the RoboTwin task and demonstration format to an
Isaac Sim/Isaac Lab runtime, a 14-dimensional ALOHA action contract, an
optional GR00T N1.7 policy service, and machine-checkable experiment artifacts.
It is designed around one rule: a video is presentation evidence only after the
corresponding physics, task, and settling checks pass.

- **Reproducible data path:** audit RoboTwin HDF5 demonstrations and convert
  episode-disjoint splits to a GR00T-flavored LeRobot v2 dataset.
- **Dual-runtime isolation:** keep the Isaac simulator and GR00T dependencies in
  separate Python environments and assign them to separate GPUs.
- **Contact-aware demos:** retain rigid-body dynamics, collision, friction,
  gravity, and post-release settling during verified manipulation.
- **Auditable outputs:** pair videos with JSON manifests and compressed traces
  containing actions, object motion, poses, and contact diagnostics.
- **Clear claim boundaries:** distinguish verified recorded-action replay from
  experimental GR00T inference and post-training.

## Demo gallery

<table>
  <tr>
    <td width="50%" align="center"><strong>Two-bowl stacking</strong></td>
    <td width="50%" align="center"><strong>Stable three-block stacking</strong></td>
  </tr>
  <tr>
    <td><img src="docs/media/bowl_stacking.gif" alt="ALOHA stacking two bowls" width="100%" /></td>
    <td><img src="docs/media/three_block_stacking.gif" alt="ALOHA stacking three blocks" width="100%" /></td>
  </tr>
</table>

The block demos are accepted only after an additional settling window verifies
centering, uprightness, linear speed, and angular speed. The warehouse demo
adds a continuous physical infeed: three dynamic parcels travel about 0.414 m,
stop at the pickup zone, and are sorted through finger collision and friction.
See [block stability](docs/block-stability.md) and
[parcel sorting](docs/parcel-sorting-demo.md) for the acceptance contracts.

## G1 在线推理演示：纸箱从货架搬入蓝色料盘

[![点击观看 G1 在线推理双机位视频](docs/media/g1_n16_live_preview.png)](media/g1_n16_live_success_cutdown.mp4)

点击上图观看 [20 秒双机位视频](media/g1_n16_live_success_cutdown.mp4)；另有
[未剪辑的机器人头部相机视频](media/g1_n16_head_rgb_episode0.mp4)。这是
**GR00T N1.6 `gn1_6` 权重的在线闭环推理**，不是录制动作回放，也不是本仓库
RoboTwin ALOHA 路线尚在实验中的 N1.7 结果。

```mermaid
flowchart LR
    instruction["自然语言任务：从货架拿起纸箱，放入蓝色料盘"]
    observation["当前观测：头部 RGB + G1 机器人状态"]
    bridge["Arena 闭环策略接口：整理观测并请求动作"]
    model["GR00T N1.6 策略服务<br/>加载 gn1_6 模型权重"]
    control["预测动作 → G1 全身控制"]
    physics["Isaac Lab-Arena + Isaac Sim / PhysX<br/>执行并更新物理状态"]
    evidence["相机 MP4 + 运行 manifest + episode 结果"]

    instruction --> bridge
    observation --> bridge --> model --> control --> physics
    physics -->|"下一时刻 RGB 与机器人状态"| observation
    physics --> evidence
```

这条闭环在每次仿真推进后重新取得观测，并由模型给出后续动作；视频由仿真相机录制，
成功判定来自独立保存的任务结果。此次只核验了 **seed 10 的一个完整 episode**：
第 1001 步 `success=true`、`overall_score=1.0`。不能把单次结果当成成功率，
也不能推断 N1.7 或 ALOHA 双臂任务已成功。查看
[运行说明与命令](docs/g1-n16-single-gpu.md)、
[原始 manifest](docs/evidence/g1-n16-20261022/run_manifest.json) 和
[episode 结果](docs/evidence/g1-n16-20261022/episode_results_rank0.jsonl)。

## Project framework

```mermaid
flowchart LR
    subgraph inputs ["Data and configuration"]
        demos[(RoboTwin HDF5)]
        taskConfig["Task configs"]
        isaacAssets["Isaac assets"]
    end

    subgraph preparation ["Preparation pipeline"]
        audit["Dataset audit"]
        conversion["LeRobot v2 conversion"]
        checkpoint["GR00T post-training"]
    end

    subgraph runtime ["Dual-GPU runtime"]
        arena["RoboTwin Arena"]
        simulator["Isaac Sim and PhysX"]
        replay["Recorded ALOHA replay"]
        bridge["Policy bridge"]
        policy["GR00T N1.7 server"]
    end

    subgraph validation ["Validation and delivery"]
        observations[(RGB state contacts)]
        metrics["Acceptance gates"]
        artifacts["MP4, GIF, JSON, NPZ"]
    end

    demos --> audit --> conversion
    conversion -.->|"Training data"| checkpoint
    checkpoint -.->|"Experimental"| policy
    taskConfig --> arena -->|"Launches task"| simulator
    isaacAssets --> simulator
    demos -->|"Verified actions"| replay -->|"Joint targets"| simulator
    simulator -.->|"RGB and state"| bridge
    bridge -.->|"Policy request"| policy
    policy -.->|"14-D action"| bridge
    bridge -.->|"Experimental targets"| simulator
    simulator --> observations --> metrics -->|"Accepted runs"| artifacts

    classDef verified fill:#d3f9d8,stroke:#2b8a3e
    classDef experimental fill:#fff3bf,stroke:#e67700
    class replay,simulator,metrics verified
    class checkpoint,bridge,policy experimental
```

Green nodes are on the currently verified presentation path. Yellow nodes are
implemented or planned experiment interfaces and must not be interpreted as a
validated GR00T result for the RoboTwin ALOHA route. The separate G1 N1.6
inference result below does not validate that ALOHA route. The detailed component and deployment view is in
[`docs/architecture.md`](docs/architecture.md).

## Evidence and project status

| Area | State | Evidence boundary |
|---|---|---|
| Bowl and block stacking | Verified replay | RoboTwin joint/gripper targets, Isaac physics, task success, post-settle checks |
| Warehouse parcel sorting | Verified replay | Physical conveyor traction, finger contacts, lift, hold, release, and settling traces |
| G1 shelf-to-bin box transfer | **Verified live GR00T N1.6 inference** | Official Isaac Lab-Arena G1 task, seed 10; one completed episode at step 1001, simulator `success=true` and `overall_score=1.0`. [Video](media/g1_n16_live_success_cutdown.mp4) · [raw head camera](media/g1_n16_head_rgb_episode0.mp4) · [run record](docs/g1-n16-single-gpu.md) |
| Dataset audit and conversion | Implemented | 14-D state/action contract, camera and episode consistency checks |
| GR00T policy transport | Implemented interface | Numeric request/response boundary; not a closed-loop learning result |
| GR00T N1.7 post-training | Experimental | Requires a separate GPU with at least 40 GB VRAM |
| G1 home-service composition | Experimental archive | Kept outside the accepted ALOHA-only showcase |
| RoboTwin Isaac Sim 6 / Isaac Lab 3 migration | Planned track | The separate G1 Arena result must not be pooled with the strict RoboTwin Sim 5.1 track |

The verified demo controller uses successful recorded RoboTwin targets. It does
**not** claim that GR00T produced the actions shown in the current GIFs. By
contrast, the linked G1 shelf-to-bin video came from online GR00T N1.6 model
inference, not the recorded ALOHA replay. The
research question and gates for making that claim are documented in
[`docs/experiment-design.md`](docs/experiment-design.md).

## Quick start

Dependency-light development and contract tests run without Isaac Sim:

```bash
git clone https://github.com/Erictianyufan/TwinGR00T-Lab.git
cd TwinGR00T-Lab
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
pytest
ruff check .
```

For the validated cloud layout—Ubuntu 22.04, two RTX 4090 GPUs, 128 GB RAM,
and about 500 GB disk—follow the
[`docs/deployment.md`](docs/deployment.md) runbook. Keep the simulator on GPU 0
and the policy service on GPU 1; two 24 GB cards do not become one 48 GB device
for GR00T post-training.

### Audit and convert a smoke episode

```bash
python -m pip install -e '.[data]'
twingr00t-audit vendor/RoboTwin/data/stack_bowls_three/data/demo_0.hdf5 \
  --output results/manifests/smoke-data.json

twingr00t-convert \
  --input-dir vendor/RoboTwin/data/stack_bowls_three \
  --output-dir data/lerobot/stack_bowls_three-smoke \
  --limit 1 --fps 31.25 \
  --manifest results/manifests/conversion-smoke.json
```

### Run the strict simulator smoke test

```bash
source /home/ubuntu/env_robotwin/bin/activate
python scripts/smoke_arena.py \
  --headless --device cuda:0 --steps 50 \
  --output results/manifests/arena-smoke.json \
  --environment manip_eval_tasks.examples.manipulation.stack_bowls_three_environment:StackBowlsThreeEnvironment \
  stack_bowls_three --embodiment aloha --enable_cameras True
```

Do not set `CUDA_VISIBLE_DEVICES` for Isaac Sim on the validated host; use
`--device cuda:0` so Vulkan and CUDA enumerate the same simulator GPU.

## Repository map

```text
assets/                  lightweight project-authored USD assets
configs/                 pinned versions and experiment configuration
docs/                    architecture, protocols, runbooks, and checkpoints
remote_overrides/        reviewed patches copied into upstream environments
scripts/                 bootstrap, rendering, evaluation, and audit entrypoints
src/twingr00t_lab/       data contracts, conversion, policy bridge, and metrics
tests/                   dependency-light unit and contract tests
artifacts/               ignored generated media and large runtime outputs
results/                 ignored or selectively versioned metrics and manifests
vendor/                  ignored upstream checkouts on the experiment host
```

Start with the [`docs/` index](docs/README.md). Large datasets, checkpoints,
videos, upstream repositories, and provider assets are intentionally excluded
from Git; compact showcase GIFs are the exception.

## Reproducibility and responsible use

- Record exact source commits, runtime versions, commands, seeds, and hardware.
- Keep train, validation, and test episodes disjoint; never split by frame.
- Never commit SSH passwords, Hugging Face tokens, cloud credentials, or model
  access tokens. Follow [`SECURITY.md`](SECURITY.md) if a secret is exposed.
- Treat simulation success as simulation evidence, not proof of safe real-robot
  deployment. Add hardware limits, emergency stops, and independent validation
  before transferring any policy to a physical robot.
- Respect the licenses and access terms of RoboTwin, NVIDIA models, simulator
  assets, robot descriptions, and every downloaded dependency.

## Author, citation, and license

TwinGR00T Lab is authored and maintained by
[Erictianyufan](https://github.com/Erictianyufan). GitHub contribution credit is
derived from commit authorship; scholarly/software citation metadata is provided
in [`CITATION.cff`](CITATION.cff).

No project license has been selected yet. Until a `LICENSE` file is added, the
repository is source-visible but does not grant general reuse, modification, or
redistribution rights. Upstream code, models, and assets remain subject to their
own licenses and terms.
