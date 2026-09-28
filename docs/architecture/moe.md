# MoE 稀疏专家详解

Mixture-of-Experts 将 FFN 替换为「路由 + 专家」：

1. **路由**：`TopKRouter` 对每个 token 输出 K 个专家及其权重（softmax 后的 top-k 门控）。
2. **专家**：`SwiGLUExpert` 是 SwiGLU 激活的 FFN。
3. **加权求和**：K 个专家的输出按权重相加。
4. **负载均衡**：`load_balancing_loss` 惩罚专家利用率不均，鼓励均匀调度。

相比稠密 FFN，MoE 只激活 K 个专家，参数量大幅增加而计算量基本不变——这正是 openPangu-2.0 稀疏架构的核心思想。
