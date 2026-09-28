# 教程：指令微调

```bash
python -m panguforge sft --config examples/config_sft.yaml
```

SFT 使用 `InstructionDataset`（指令 → 响应），训练循环与预训练共用 `Trainer`。
