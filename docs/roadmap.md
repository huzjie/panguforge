# 路线图

## 已完成（v1.0.0）

- [x] 纯 Python 张量库与 MoE Transformer
- [x] Top-K 路由 + 负载均衡
- [x] SWA / DSA 稀疏注意力
- [x] 预训练 / SFT / GRPO / PPO
- [x] mock / cpu / ascend / openai / vllm / transformers 六后端
- [x] OpenAI 兼容 HTTP 服务
- [x] Docker / K8s / Helm / CI

## 规划中

- [ ] 真实 torch-npu 昇腾后端
- [ ] 专家并行分布式训练（真实 all-to-all）
- [ ] 增量预训练与模型合并
- [ ] 更多基准（HumanEval、MBPP、长上下文检索）
- [ ] 量化推理（INT8 / FP8）
