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

当物理重放至少得到 10 条经审计的成功轨迹后，在单独的转换环境中运行 `scripts/prepare_aloha_factory_dataset.sh`。脚本按轨迹生成 80/10/10 训练、验证、测试集，使用与重放控制步长一致的 50 fps，并调用锁定版本的 N1.7 统计工具。它拒绝覆盖已有分割目录，防止追加重放后误用过期划分。训练时将 `DATASET_PATH` 指向 `data/lerobot/blocks_ranking_rgb-train`；测试轨迹不得进入训练。
