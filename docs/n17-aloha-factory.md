# GR00T N1.7 × ALOHA 工厂分拣：正式推理验收路线

本路线的唯一正式策略是 **GR00T N1.7 在 ALOHA AgileX 双臂上的微调权重**。`scripts/parcel_sorting_physics_demo.py` 使用录制动作驱动仿真，只是物理场景预览，不是策略推理视频。G1/N1.6 实验也不能替代此验收。

## 数据来源与当前阻塞

公开数据采用 [RoboTwin2.0 的 `blocks_ranking_rgb/aloha-agilex_clean_50`](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0)；每个 HDF5 包含三路 RGB、14 维 `joint_action/vector` 和语言指令。**动作是关节目标命令，不等于执行后的实测关节位置。** 当前下载版本没有 `observation/joint_state/vector`。因此不能把 `joint_action` 改名为 `observation.state`，也不能声称基于这份原样数据训练了闭环策略。

审计命令：

```bash
python -m twingr00t_lab.xpolicylab_audit \
  data/robotwin_public/dataset/blocks_ranking_rgb/aloha-agilex_clean_50 \
  --output results/manifests/blocks_ranking_rgb_xpolicy_audit.json
```

`data_contract_ready=false` 表示数据还不能转换为本项目要求的 LeRobot `observation.state`。它不是模型质量指标。

## 从数据到可信视频

1. 在严格对应的 RoboTwin/Isaac Sim 版本中重放公开动作，**逐帧从仿真机器人关节读数记录真实状态**，同时记录实际 RGB、动作目标、对象位姿、接触和任务成功信号。只保留成功且时间戳一致的轨迹；必须保留原始来源、场景随机种子与重放误差。若视觉帧与原数据不一致，使用同一次重放采集的视觉帧，不混搭两次运行的图像与状态。

`blocks_ranking_rgb` 的回放初始方块位姿优先从每条轨迹自身的首帧 RGB 与相机标定估计。实测发现 episode 1 按归档 seed 重建的起点与首帧相差十几至几十厘米；所以不能默认 seed 路径可靠。视觉定位失败时才回退 seed，并在日志中明确标记。
2. 把同步采集的三路 RGB、14 维实测状态、14 维动作和指令转换为 GR00T-flavored LeRobot v2，并计算 `stats.json`、`relative_stats.json`。在官方数据加载器中先验证一条轨迹的维度、归一化、相机方向和动作语义；训练集与测试集按**轨迹**分开。
3. 从 `nvidia/GR00T-N1.7-3B` 加载基础权重，使用 `NEW_EMBODIMENT` 和本项目 ALOHA modality config 微调。先做开环动作曲线检查，再用未参与训练的场景做闭环评测；不能仅凭训练 loss 下降宣称任务成功。
4. 正式录制时每个控制周期保存：当期 RGB/实测状态、N1.7 请求与原始输出、动作后处理结果、发送给 Isaac 的目标、执行后关节与物体状态、模型调用延迟。完成红绿蓝分拣后检查每次抓取接触、物体最终区域、稳定性以及无脚本动作注入；三机位视频与相同轨迹的 JSON/NPZ 一起导出。

上游 N1.7 的 `Gr00tPolicy.get_action()` 已调用 `processor.decode_action(...)`；相对关节动作在解码时以当期实测状态恢复成**绝对关节目标**。仿真桥接层因此只做 14→16 维夹爪映射，不得再把返回值当增量累加一次。训练数据中的状态与动作仍都应存为绝对值，由上游处理器负责相对化和逆变换。

## 云机条件

- **访问权**：云机上 `hf auth whoami` 必须显示有权访问 `nvidia/Cosmos-Reason2-2B` 的账号；实际下载一个 gated 文件成功才算通过。网页登录本身不代表云机登录。
- **显存**：N1.7 微调按官方最低约 40 GB/卡规划。48 GB 卡与另一项目共享时，需先测量空闲显存，再约定独占训练窗口；不得中断另一项目的进程。
- **版本**：当前项目锁定 RoboTwin/Isaac Sim 5.1/Isaac Lab 2.3 与 GR00T N1.7 指定 commit。现有 Isaac Sim 6 环境不能未经迁移验证就视为等价。
- **磁盘**：下载模型和新建两套 Python/Isaac 环境前检查可用空间，并保留权重、缓存、日志、视频的余量。不得为腾空间清理其他项目的数据。

真正的正式成果必须给出 N1.7 微调 checkpoint、可复现的在线推理命令、闭环成功轨迹与多机位视频。上述任一项缺失时，只能标记为准备中或回放预览。

`scripts/prepare_aloha_factory_dataset.sh` 是早期至少 10 条轨迹的 80/10/10 划分工具，默认 50 fps；**不能直接套用当前实测 62.5 Hz 的录制数据**。当前九条训练轨迹不重新混入留出集，工厂重采与第四阶段续训使用下述专用入口。

## 第四阶段：完整工厂重采轨迹续训

```bash
cd /home/ubuntu/mycode/robotwin-gr00t-aloha
INPUT_DIR="$PWD/data/replayed/blocks_ranking_rgb-factory-v1" \
bash scripts/train_n17_factory_stage.sh
```

包装固定接收原训练来源 episode 0–8，保持完整三块分拣、双臂流程，不改为单物体任务。每条 `reports/demo_N.json` 必须标记 `teacher_replay_for_training`、工厂场景、物理验收通过，并与原始 HDF5、指令及新 HDF5 的 SHA256 一致。它还独立核查 14 维实测状态、16 维完整源动作、左右 EEF、三个方块及稳定尾窗，重新计算默认抓取验收；失败、缺条、错来源或伪造成功标志都会阻止训练。

新数据必须为 62.5 Hz、真实执行至少 1 秒尾段、执行后同帧记录。转换采用 `observation[i] → action[i+1]`；新 JPEG 使用正确 RGB 编码，**禁止再次做 legacy BGR 交换**。在新 LeRobot 目录调用锁定上游 `gr00t/data/stats.py` 生成绝对及相对统计量，不复用旧缓存。

默认从 `n17-aligned-v3-coloroff/checkpoint-4000` 续训至新目录 `n17-factory-v4`：4,000 步、每 1,000 步保存、仅保留 2 个检查点，batch 1、梯度累积 8、Adafactor、学习率 5e-5、state dropout 0、关闭颜色增强。启动训练前要求至少 40 GiB 可用磁盘、目标 GPU 无其他计算进程，随后再检查至少 22,000 MiB 空闲显存；不停止其他项目。

`PREPARE_ONLY=1` 只进行数据验证、转换和统计量生成。`INPUT_DIR`、`SOURCE_DIR`、`DATASET_PATH`、`OUTPUT_DIR`、`BASE_MODEL_PATH` 可指定路径；数据、审计文件及训练输出均不得覆盖已有结果。准备失败后保留诊断文件，修正原因后使用新输出路径重试。此阶段产生的是教师回放训练数据及模型权重，**不是已完成真实模型推理 Demo**；正式视频仍须另行闭环推理、验收和复核。

## 离线动作诊断的语言条件

`scripts/diagnose_n17_offline.py` 默认分别读取训练、留出 HDF5 同级数据集下的 `instructions/<demo文件名>.json`，使用各自首条 `seen` 指令。结果 JSON 的 `prompts.train`、`prompts.heldout` 保存实际文本及来源。缺少指令时明确报错，不自动借用另一条轨迹的语言。

两条轨迹可能分别要求左/右/左与右/左/左的双臂分工；对照各自教师动作时应保留该条件。`--train-prompt`、`--heldout-prompt` 可分别覆盖；兼容旧的 `--prompt` 同指令实验，但会标注为显式共用覆盖，且单独覆盖优先。此类条件变化下的逐关节误差不能直接归因于模型泛化能力。

```bash
python scripts/diagnose_n17_offline.py \
  --checkpoint artifacts/checkpoints/n17-aligned-v3-coloroff/checkpoint-4000 \
  --train-demo data/replayed/blocks_ranking_rgb-vision-batch/data/demo_0.hdf5 \
  --heldout-demo data/replayed/blocks_ranking_rgb-vision-ep1/data/demo_0.hdf5 \
  --correct-legacy-jpeg-bgr --observation-action-offset 1 \
  --flow-steps 8 --repeats 3 --seed 42 \
  --indices 0 40 50 60 100 180 190 200 210 300 350 400 440
```

`--indices` 支持密集检查闭爪、释放及末段动作；无法容纳完整 16 步目标窗口的帧会跳过。旧版共用指令的离线报告保留原样，不视为已经按上述方式重测。离线误差不是闭环抓取成功证据。
