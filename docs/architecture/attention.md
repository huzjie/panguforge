# 注意力机制

## 全注意力（MultiHeadAttention）

缩放点积注意力 + 因果 mask。

## 滑动窗口注意力（SWA）

每个 query 只看窗口内（`i - j < window`）的 key，计算量从 O(n²) 降为 O(n·w)，是超长上下文的关键手段。

## DeepSeek 稀疏注意力（DSA）

按 top-n 选择相关 key，进一步稀疏化。

三者共用同一 `MultiHeadAttention` 基类，通过 mask 切换。
