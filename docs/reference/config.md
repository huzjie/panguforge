# 配置参考

| 键 | 默认 | 说明 |
|---|---|---|
| `model.vocab_size` | 512 | 词表大小 |
| `model.dim` | 64 | 隐藏维度 |
| `model.num_heads` | 4 | 注意力头数 |
| `model.num_layers` | 2 | Transformer 层数 |
| `model.moe_hidden` | 256 | 专家 FFN 中间维度 |
| `model.num_experts` | 8 | 专家数量 |
| `model.top_k` | 2 | 每 token 激活专家数 |
| `backend` | mock | 后端选择 |
| `mock.skill_init` | 0.4 | 初始 skill |
| `mock.lr` | 0.05 | skill 学习率 |
| `train.steps` | 100 | 训练步数 |
| `rl.algorithm` | grpo | RL 算法 |
| `rl.steps` | 40 | RL 轮数 |
