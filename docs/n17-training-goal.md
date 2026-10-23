# N1.7 在线抓取与正式发布目标

状态：进行中（2026-10-23）。

最终成果为 GR00T N1.7 驱动 ALOHA 双臂完成工厂分拣的在线闭环视频，以及中文技术框架图、可复现命令和物理验收证据。视频中的动作必须来自本轮观测条件下的模型预测。

## 训练与评估路线

1. 校正录制时序。现有 HDF5 在执行动作后记录状态与图像，因此新数据使用 `observation[i] -> action[i+1]`，逐轨迹裁尾，不跨轨迹拼接。保留原数据和原 checkpoint。
2. 在新目录重新编码 RGB、生成状态/动作统计与相对动作统计。对现有批次的关节位置和速度进行时间间隔诊断，结果支持约 16 ms 记录间隔，因此新元数据为 62.5 Hz。在线运行须同时记录实际 `env.step_dt`；视频帧率单独计算。
3. 从现存 RGB checkpoint-500 分段续训。首段 200 步、每 100 步保存，batch 1、梯度累积 8、Adafactor、BF16、梯度检查点、学习率 5e-5、state dropout 0。这是一张 24 GB RTX 4090 上的实验配置，是否可用以实际显存和梯度检查为准。
4. 先比较训练轨迹与留出轨迹的动作误差，再做闭环。检查点编号和训练 loss 不作为任务成功证据。若动作未抓住目标，分析输入、动作和物体轨迹后再调整训练。
5. 正式验收检查模型调用与执行动作的对应关系、物体抬起历史、双臂操作、最终目标排列与稳定窗口，并人工检查同一轨迹的完整视频。运动学证据不等同于直接的夹爪接触传感。
6. 成功后制作中文架构图，导出完整多机位视频，将 README、复现说明、模型版本和运行证据更新至 GitHub 主文档。失败诊断视频不得展示为已完成任务。

## 当前服务器工作目录

项目：`/home/ubuntu/mycode/robotwin-gr00t-aloha`

```bash
cd /home/ubuntu/mycode/robotwin-gr00t-aloha
bash scripts/train_n17_aligned_stage.sh
```

首段数据目录：`data/lerobot/blocks_ranking_rgb-aligned-v1`。

首段权重目录：`artifacts/checkpoints/n17-aligned-v1`。

首段运行日志：`results/logs/n17_aligned_stage1_20261023.log`。

完整训练集来自 `blocks_ranking_rgb-vision-batch` 的 source episode 0–8；留出场景和训练场景必须在结果中分别标注。后续发布需保存并核对训练、验证、测试的 source episode ID 和原始来源，不能只比较导出目录名。

## 2026-10-23 实际进展

- 首段 200 步训练完成，运行约 424 秒，观测显存约 20,380 MiB。
- `n17_aligned200_train0_20261023` 是真实模型在线诊断，不是成功成果。训练场景运行 1,500 控制步、188 次推理，没有满足抓取条件；红、绿方块未搬动，蓝方块最多发生约 5.5 cm 位移。
- 同一 checkpoint 的离线训练帧 0/50/100/200/300，16 步动作窗口混合维度 MAE 分别约 0.134/0.065/0.067/0.058/0.060。该指标混合弧度与米，不能解释为统一的物理误差，也不能替代闭环验收。
- 第二段从首段 checkpoint-200 加载权重，使用新优化器与调度器再训练 1,000 步，耗时约 1,860 秒、平均训练 loss 0.06147。输出 `artifacts/checkpoints/n17-aligned-v2/checkpoint-1000`，日志 `results/logs/n17_aligned_stage2_20261023.log`。
- 第二段在线评测 `n17_aligned1000_train0_20261023` 仍为 `task_not_completed`：1,500 控制步，没有满足抓取条件。不得把此次录像称为成功推理 Demo。训练帧重复三次推理的关节标准差约 0.031–0.054 rad，显示预测波动仍需降低。
- 正式录制证据升级到 schema 2：保存原始动作块、有效限幅边界、执行动作、物体与机器人实测状态、逐帧对应控制步、MP4 哈希和实际帧数。验收独立重算动作链路和抓取/稳定放置条件，最终还必须完成视频复核。

第二段复现命令：

```bash
BASE_MODEL_PATH="$PWD/artifacts/checkpoints/n17-aligned-v1/checkpoint-200" \
OUTPUT_DIR="$PWD/artifacts/checkpoints/n17-aligned-v2" \
MAX_STEPS=1000 SAVE_STEPS=500 SAVE_TOTAL_LIMIT=1 \
bash scripts/train_n17_aligned_stage.sh
```

训练脚本拒绝覆盖非空输出目录。以上记录只用于追踪训练与诊断，不是正式成功声明。

## 第三段：降低颜色增强并继续拟合

从第二段 checkpoint-1000 继续使用同一九条训练轨迹，新增 4,000 步；明确关闭亮度、对比度、饱和度、色相随机扰动，裁剪、旋转和网络 dropout 不变。此设置通过上游支持的 processor override 传入，训练完成后再次核验保存配置；不是仅修改一个无效的 model config 字段。每 1,000 步保存、保留四个阶段候选。尚未完成闭环验收，不能推断本段必然成功。

```bash
cd /home/ubuntu/mycode/robotwin-gr00t-aloha
export PYTHONPATH="$PWD/src:/home/ubuntu/mycode/robotwin-gr00t-lab/vendor/Isaac-GR00T-N1.7-src"
export NO_ALBUMENTATIONS_UPDATE=1 TOKENIZERS_PARALLELISM=false
/home/ubuntu/env_groot/bin/python scripts/launch_aloha_n17_lowmem.py \
  --dataset-path data/lerobot/blocks_ranking_rgb-aligned-v1 \
  --output-dir artifacts/checkpoints/n17-aligned-v3-coloroff \
  --base-model-path artifacts/checkpoints/n17-aligned-v2/checkpoint-1000 \
  --modality-config-path configs/groot_robotwin_aloha.py \
  --max-steps 4000 --save-steps 1000 --save-total-limit 4 \
  --batch-size 1 --gradient-accumulation-steps 8 \
  --optimizer adafactor --min-free-mib 22000 \
  --state-dropout-prob 0 --learning-rate 0.00005 --visual-augmentation color-off
```

日志：`results/logs/n17_aligned_stage3_coloroff_20261023.log`。

### 训练后的顺序诊断

`scripts/run_n17_posttrain_checks.sh` 等待指定训练进程退出，核对最终 checkpoint 步数、分片文件和实际颜色增强配置，再依次执行离线检查与两组在线录像。它不会把训练进程退出、录像完成或单纯的运动学通过当作正式成功，也不会自动发布结果。

- 两组均固定策略与仿真种子为 42，使用同一训练初始场景；分别为 `factory` 与 `no-factory-dressing`。后者仅移除新增工厂装饰，不代表已经重建原始采集环境，不能用于正式工厂验收。
- 每组保存三路实际传给策略的首帧 PNG、数值范围和文件哈希，用于排查颜色及图像输入问题。
- GPU 显存检查不通过时退出，不停止其他项目进程。最终仍需检查完整录像、独立验收，以及区分训练场景和留出场景。

2026-10-23 已完成云盘在线扩容：底层磁盘 450 GiB，根分区 ext4 文件系统显示约 436 GiB；扩容后核验约有 105 GiB 空闲。扩容未重启机器或中断训练，后续余量随 checkpoint 保存变化。

## 中文技术流程图

![N1.7 双臂抓取、分拣训练与在线推理流程](media/n17-framework-zh.png)

图为流程示意，不是任务成功证据。使用内置图片工具生成；箭头经过修订，模型检查点加载到策略模块，录像必须通过验收后才能作为正式成果发布。生成提示与修订记录见 [图像提示词](media/n17-framework-zh.prompt.txt)。
