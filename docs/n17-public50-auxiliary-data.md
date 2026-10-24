# N1.7 彩色积木辅助数据：来源、对齐与边界

这份数据用于改善双臂策略的空间泛化，**不是** GR00T N1.7 在 Isaac Sim 中完成分拣的证据。最终结果仍以本仓库的在线闭环视频、物理验收和人工逐帧复核为准。

## 来源与格式

- 来源：[RoboTwin 2.0 公共数据集](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0)，`dataset/blocks_ranking_rgb/aloha-agilex_clean_50.zip`，MIT 许可。
- 仅读取压缩包中 50 条 `data/episode*.hdf5` 与 `instructions/episode*.json`。`_traj_data/*.pkl` 没有加载或执行。
- 原始 `joint_action/vector` 虽名为 action，但 RoboTwin 录制器从机器人关节状态读取它；因此转换为 `观测与状态 i → 状态 i+1` 的模仿目标，不误称为执行命令。
- 原视频逐条核对均为 30 FPS（50/50），转换输出包含 50 个 episode、23,041 组配对帧、三路相机共 150 段视频；150 段视频的帧数均经独立 `ffprobe` 核对。原相机是旧版 JPEG：常规 JPEG 解码会反转红蓝通道，转换时做且只做一次通道修正；抽检 450 张相机帧均没有新版 `XPL-RGB1` 标记。
- 夹爪状态原值 `[0,1]` 映射到 Isaac ALOHA 近似测量范围 `[0.006,0.048]` 米；下一帧目标映射到执行命令范围 `[-0.010,0.048]` 米。转换器及输出清单都明确记录此假设。

## 与工厂场景的区别

公开数据来自 RoboTwin/SAPIEN 的白色场景、30 FPS；现有工厂教师数据来自 Isaac Sim，控制频率为 62.5 Hz，且 9 条轨迹经过 Isaac 物理验收。两者不能直接混为同域成功案例。新增数据先作为辅助微调，随后必须在工厂场景重新闭环测试；如未通过，保留失败记录，不发布“成功”结论。

转换命令（不覆盖已有输出）：

```bash
PYTHONPATH=src /home/ubuntu/env_convert/bin/python -m twingr00t_lab.public_robotwin \
  --archive data/thirdparty/robotwin2-clean50/dataset/blocks_ranking_rgb/aloha-agilex_clean_50.zip \
  --output-dir data/lerobot/blocks_ranking_rgb-public50-v1 \
  --episodes 50 --fps 30
```

输出 `meta/source_audit.json` 记录源文件 SHA-256、每条轨迹帧数、对齐方式和非 Isaac 验收状态。转换后单独运行 GR00T 统计生成器产生绝对/相对归一化参数。公开压缩包和训练后的权重不提交到 GitHub。
