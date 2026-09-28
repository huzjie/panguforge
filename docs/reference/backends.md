# 后端参考

| 后端 | 注册名 | 训练 | 推理 | 依赖 |
|---|---|---|---|---|
| mock | `mock` | ✅ 可训练 | ✅ 确定性 | 无 |
| cpu | `cpu` | 部分 | ✅ 真实前向 | 无 |
| ascend | `ascend` | 契约 | 契约 | CANN/torch-npu |
| openai | `openai` | ❌ | ✅ | API key |
| vllm | `vllm` | ❌ | ✅ | vllm |
| transformers | `transformers` | ❌ | ✅ | transformers |
