# 确定性可训练 Mock 后端

## 设计目标

让整个训练/推理流程在没有 GPU、没有 torch 的环境里也能**真实、可复现、可量化**地运行。

## 核心概念

- `skill`：模型预测「正确下一 token」的概率。
- `world_target(key)`：确定性世界模型，给定前缀返回正确 token。
- `compute_loss = -log(skill)`。
- `train_step`：`skill += lr * (1 - skill)`，单调趋 1。

## 为什么可靠

训练信号有明确定义：skill 上升 ⇔ loss 下降 ⇔ 准确率上升。任何一次运行都能量化「学到了」，且结果跨机器、跨进程完全一致（md5 确定性）。
