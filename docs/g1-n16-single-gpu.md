# G1 GR00T N1.6 单卡真实推理 Demo

这条路线与仓库中 RoboTwin ALOHA 轨迹回放不同：使用 NVIDIA 公开的、针对 Isaac Lab-Arena G1 搬箱任务微调的 `nvidia/GN1x-Tuned-Arena-G1-Loco-Manipulation` **`gn1_6` 修订版**。GR00T 服务必须用 `--model-path` 加载权重；`--dataset-path` 会进入回放策略，不属于本实验。

## 环境与资源

- 仿真：Isaac Sim 6.0.1 / Isaac Lab 3.0 beta2 / Isaac Lab-Arena `release/0.3.0`，Python 3.12。
- 模型：Arena 指向的 Isaac-GR00T 子模块 `e29d8fc50b0e4745120ae3fb72447986fe638aa6`，独立 Python 3.10 环境。
- 两个进程共用单张 48 GB GPU。启动器要求开跑时至少留有 37,000 MiB 空闲显存；不会停止或覆盖另一项目进程。若另一个项目正在跑 Isaac Sim，请等待它完成，再启动本 Demo。
- 录制：Arena 原生相机视频；可选 viewport 视频。视频仅证明发生了闭环运行，任务成功以仿真结果和阶段日志为准。

## 准备

以下路径是云机部署布局，可用启动器参数调整：

```bash
cd /home/ubuntu/mycode/robotwin-gr00t-lab
git clone --depth 1 --branch release/0.3.0 \
  https://github.com/isaac-sim/IsaacLab-Arena.git vendor/IsaacLab-Arena-0.3.0
git -C vendor/IsaacLab-Arena-0.3.0 submodule update --init --recursive

# 模型分支必须是 gn1_6，不能用默认分支的旧模型配置。
HF_HUB_DISABLE_XET=1 hf download --revision gn1_6 \
  nvidia/GN1x-Tuned-Arena-G1-Loco-Manipulation \
  config.json model-00001-of-00002.safetensors \
  model-00002-of-00002.safetensors model.safetensors.index.json \
  processor_config.json embodiment_id.json statistics.json \
  --local-dir checkpoints/GN1x-Tuned-Arena-G1-Loco-Manipulation-gn1_6
```

Arena 的 Isaac-GR00T 子模块、两个虚拟环境及必要依赖须先安装。启动器会验证 checkpoint 结构和显存，不会代替安装。

## 启动与证据

```bash
# 先用 100 步验证模型加载、相机、通信及动作映射。
python scripts/run_g1_n16_live_demo.py --steps 100

# 首轮通过后录制完整闭环；viewport 会额外消耗显存。
python scripts/run_g1_n16_live_demo.py --steps 1500 --viewport-video
```

每次运行创建独立 `results/g1_live/<时间戳>/`，包含 `run_manifest.json`、`server.log`、`runner.log`、每相机 MP4、可选 viewport MP4 及 Arena 的 episode 结果。`run_manifest.json` 明确记录 `controller_source=GR00T_N1.6_model_inference`、`replay_used=false`、权重版本和进程退出码；这并不自动意味着任务成功，须检查 episode 成功判据与视频中的抓取、搬运和释放。

本路线只是 G1 官方搬箱任务的真实 GR00T 推理，不等同于仓库 ALOHA 三碗堆叠的 N1.7 后训练结果，也不等同于 G1＋ALOHA 双机器人完整交接。任何演示辅助或播放已保存轨迹的视频都必须分别标注。

## 已验证运行：2026-10-22

云机使用单张 48 GB RTX 4090；另一个项目的 π₀.₅ 服务仍在运行，本次没有停止它。执行命令为：

```bash
cd /home/ubuntu/mycode/robotwin-gr00t-lab
/home/ubuntu/env_isaaclab/bin/python scripts/run_g1_n16_live_demo.py \
  --steps 1600 --viewport-video
```

结果目录：`results/g1_live/20261022-195443/`。第一条完整 episode（seed 10）在第 1001 步满足仿真任务判据：`success=true`、`overall_score=1.0`、`all_complete=true`。记录的阶段事件为物体稳定（步 1）、抬起纸箱（步 59）、在目标位置满足接触和速度阈值（步 1001）。启动器记录 `runner_exit_code=0`、`replay_used=false` 和实际 `--model-path` 命令；权重配置为 `Gr00tN1d6`，配置 SHA-256 为 `acd7f0cd662a02641f191273ec887e5a9619d1b91148f4dcebc2bab675435772`。

- [20 秒双机位连续剪辑](../media/g1_n16_live_success_cutdown.mp4)：0–10 秒第三人称，10–20 秒机器人头部相机；两段来自同一在线 rollout，没有重放或改写动作。
- [未剪辑的头部相机 episode 视频](../media/g1_n16_head_rgb_episode0.mp4)。
- [原始运行 manifest](evidence/g1-n16-20261022/run_manifest.json) 与 [原始 episode 结果](evidence/g1-n16-20261022/episode_results_rank0.jsonl)。

本次只验证了 **1 条任务、1 个随机种子、1 个成功 episode**；不能据此声称总体成功率，也没有验证 N1.7、RoboTwin ALOHA 任务或跨机器人迁移。第三人称 viewport 在机器人离开货架后没有持续跟拍，因此放置过程以头部相机和仿真判据为准。
