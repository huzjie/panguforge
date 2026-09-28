# API 参考

## 命令行

| 命令 | 说明 |
|---|---|
| `panguforge doctor` | 环境与后端体检 |
| `panguforge pretrain` | 预训练 |
| `panguforge sft` | 监督微调 |
| `panguforge rl` | RL 后训练 |
| `panguforge serve` | 启动 HTTP 服务 |
| `panguforge bench` | 运行基准 |

## HTTP API（OpenAI 兼容）

### GET /health

```json
{"status": "ok", "backend": "mock"}
```

### POST /v1/completions

请求：
```json
{"prompt": "你好", "max_tokens": 16, "temperature": 1.0, "top_p": 1.0}
```

响应：
```json
{"id": "cmpl-0", "object": "text_completion",
 "choices": [{"text": "...", "index": 0, "finish_reason": "length"}]}
```

### POST /v1/chat/completions

请求：
```json
{"messages": [{"role": "user", "content": "你好"}], "max_tokens": 32}
```

### GET /metrics

返回引擎当前指标（loss / skill / reward）。

## Python API

```python
from panguforge.config import load_config
from panguforge.backends import get_backend

cfg = load_config("config.example.yaml")
engine = get_backend(cfg["backend"], cfg)
engine.train_step(batch)
engine.generate("你好", max_tokens=16)
```

## 核心类

| 类 | 位置 | 说明 |
|---|---|---|
| `PanguMoE` | `core/model.py` | MoE Transformer 模型 |
| `MoELayer` | `core/moe.py` | MoE FFN 层 |
| `TopKRouter` | `core/router.py` | Top-K 专家路由 |
| `MultiHeadAttention` | `core/attention.py` | 多头注意力 |
| `MockEngine` | `backends/mock.py` | 确定性可训练后端 |
| `Trainer` | `train/trainer.py` | 训练编排 |
| `RLLoop` | `train/rl.py` | RL 后训练循环 |
