# 教程：RL 后训练

```bash
python -m panguforge rl --config examples/config_rl.yaml
```

GRPO 对每个 prompt 采样 K 个回答，按组内相对优势更新策略，奖励单调上升。
