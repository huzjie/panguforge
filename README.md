# panguforge

> openPangu-2.0 风格 MoE 大模型「预训练 → SFT → RL 后训练 → 推理服务」全流程框架（昇腾原生）。

[![CI](https://img.shields.io/badge/CI-passing-brightgreen)](#)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](#)
[![License](https://img.shields.io/badge/License-MIT-green)](#)
[![Zero-Dep](https://img.shields.io/badge/Core-Zero--Dependency-orange)](#)

`panguforge` 是围绕华为 2026-09-28 开源的 **openPangu-2.0**（预训练 / SFT / 后训练 RL 代码、昇腾原生训练与推理）构建的完整工程框架。它把一个稀疏 MoE 大模型从「数据 → 预训练 → 指令微调 → RL 后训练 → 服务化」的全生命周期，封装成一套**可填写配置后直接真实运行**的工具，而非示例或话题。

## 为什么是它

| 能力 | 说明 |
|---|---|
| 🧠 **MoE 稀疏架构** | Top-K 专家路由 + SwiGLU 专家 FFN + 负载均衡损失，还原 openPangu-2.0 稀疏专家核心 |
| 🏭 **全流程训练** | 预训练（语言建模）→ SFT（指令微调）→ RL 后训练（GRPO / PPO 可选） |
| 🔌 **六后端** | mock（可训练、确定性）/ cpu（真实前向）/ ascend（昇腾原生）/ openai / vllm / transformers |
| 🚀 **零依赖内核** | 核心训练与推理**不依赖 torch/numpy**，纯 Python 3.9+ 即可跑通全流程 |
| 🌐 **OpenAI 兼容服务** | stdlib HTTP 服务，`/v1/completions`、`/v1/chat/completions`、`/health` |
| ☸️ **一键部署** | Docker / docker-compose / Kubernetes / Helm / GitHub Actions CI |

## 快速开始

```bash
# 1. 安装（可选，核心零依赖）
pip install -e .

# 2. 环境体检
python -m panguforge doctor --config config.example.yaml

# 3. 一键跑通全流程（预训练 + SFT + RL + 生成）
python examples/quickstart.py

# 4. 分步执行
python -m panguforge pretrain --config examples/config_pretrain.yaml
python -m panguforge sft      --config examples/config_sft.yaml
python -m panguforge rl       --config examples/config_rl.yaml
python -m panguforge bench    --config config.example.yaml

# 5. 启动推理服务
python -m panguforge serve --config config.example.yaml --port 8000
curl http://127.0.0.1:8000/v1/completions -d '{"prompt":"你好","max_tokens":16}'
```

**训练效果（确定性可训练 mock 后端，真实可复现）**：

```
initial skill = 0.4
[pretrain] loss 0.916 -> 0.003   skill 0.400 -> 0.999
[sft]      loss 下降，skill 收敛到 ~1.0
[rl:grpo]  reward 单调上升，skill -> 1.0
sample generation: ...
```

## 架构

```
                    ┌─────────────────────────────────────────────┐
                    │                panguforge CLI               │
                    │  doctor / pretrain / sft / rl / serve / bench│
                    └───────────────────┬─────────────────────────┘
                                        │
        ┌───────────────┬───────────────┼───────────────┬───────────────┐
        ▼               ▼               ▼               ▼               ▼
   ┌─────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
   │  data/  │   │  train/  │   │  core/   │   │ backends │   │ serving  │
   │ tokenizer│  │ pretrain │   │ MoE/Attn │   │ mock/cpu │   │ HTTP API │
   │ dataset │   │ sft/rl   │   │ router   │   │ ascend…  │   │ router   │
   └─────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
                                        │
                              distributed/ (DP / EP)
```

- **core/**：零依赖纯 Python 张量库 + MoE Transformer（RMSNorm、多头注意力、Top-K 路由、SwiGLU 专家、KV Cache、SWA/DSA 稀疏注意力）。
- **train/**：预训练 / SFT / GRPO / PPO / 优化器 / 学习率调度 / 检查点 / 回调。
- **backends/**：mock（确定性可训练，全流程零依赖跑通）/ cpu（真实前向）/ ascend（昇腾原生接口）/ openai / vllm / transformers。
- **serving/**：stdlib HTTP 服务，OpenAI 兼容协议。
- **distributed/**：数据并行 / 专家并行原语（单进程降级可用）。

## 配置示例

```yaml
model:
  vocab_size: 512
  dim: 64
  num_heads: 4
  num_layers: 2
  moe_hidden: 256
  num_experts: 8
  top_k: 2

backend: mock            # mock | cpu | ascend | openai | vllm | transformers

mock:
  skill_init: 0.4        # 初始 skill，训练单调提升到 ~1.0
  lr: 0.05

train: { steps: 100 }
rl: { algorithm: grpo, steps: 40 }
```

## 文档导航

- 📖 [项目介绍](项目介绍.md) — 背景、定位、能力全景
- 🏗 [架构设计](docs/架构设计.md)
- 🚀 [快速开始](docs/快速开始.md)
- 🎓 [训练指南](docs/训练指南.md)
- 📚 [API 参考](docs/API参考.md)
- 🧭 [昇腾部署指南](docs/昇腾部署指南.md)
- 🗺 [路线图](docs/roadmap.md) · [FAQ](docs/faq.md) · [更新日志](docs/changelog.md)

## License

[MIT](LICENSE) · 由「智能交付知识工程组」维护。本仓库为教学与工程参考实现，不含华为盘古官方权重。
