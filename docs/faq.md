# 常见问题（FAQ）

**Q：不装任何依赖能跑吗？**
A：能。核心（mock 后端全流程 + cpu 后端真实前向）零依赖，纯 Python 3.9+。PyYAML 可选，未装时自动回退内置解析器。

**Q：为什么训练后 loss 一定下降？**
A：mock 后端把「skill」定义为正确下一 token 概率，`loss = -log(skill)`，`train_step` 使 skill 单调趋 1，因此 loss 严格下降。这是有明确定义的学习信号。

**Q：如何换后端？**
A：改配置 `backend: mock|cpu|ascend|openai|vllm|transformers`，或 `cfg["backend"] = "cpu"`。

**Q：如何新增一个 RL 算法？**
A：在 `train/` 实现 `xxx_update(engine, prompts, ...)`，在 `RLLoop` 的算法路由中登记。

**Q：昇腾后端能直接训练吗？**
A：`ascend` 后端是契约与探测实现，无 NPU 时明确报错；真实训练请在 NPU 环境对接 CANN/torch-npu。

**Q：这和华为官方 openPangu-2.0 什么关系？**
A：本仓库是围绕其开源方向做的工程参考实现，不含官方权重与官方代码。
