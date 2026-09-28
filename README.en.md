# panguforge

> An openPangu-2.0 style MoE LLM full-pipeline framework: pretrain -> SFT -> RL -> serving, Ascend-native.

## Highlights

- **MoE sparse architecture**: Top-K expert routing + SwiGLU experts + load-balancing loss.
- **Full lifecycle**: pretraining, SFT, RL post-training (GRPO/PPO), inference serving.
- **Six backends**: mock (deterministic & trainable), cpu, ascend, openai, vllm, transformers.
- **Zero-dependency core**: run the whole pipeline with pure Python 3.9+ (no torch/numpy).
- **OpenAI-compatible HTTP API** via the stdlib server.
- **Deploy**: Docker / K8s / Helm / CI.

## Quickstart

```bash
pip install -e .
python -m panguforge doctor --config config.example.yaml
python examples/quickstart.py
python -m panguforge serve --config config.example.yaml --port 8000
```

## License

MIT. Educational/engineering reference; does not include official Pangu weights.
