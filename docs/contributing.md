# 贡献指南

欢迎提交 PR。约定：

1. 新后端继承 `BaseEngine`，用 `@register_backend("名字")` 注册。
2. 新注意力继承 `MultiHeadAttention`。
3. 新增文件前先跑 `python -m panguforge doctor`。
4. 提交前跑 `python -m unittest discover -s tests -t .`。
