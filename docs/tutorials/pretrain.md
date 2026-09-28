# 教程：跑一次预训练

```bash
python -m panguforge pretrain --config examples/config_pretrain.yaml
```

观察输出：

```
step 0: loss=0.916 skill=0.400
step 20: loss=0.326 skill=0.674
...
step 100: loss=0.003 skill=0.999
```

loss 从 0.9 降到 0.003，skill 从 0.4 升到 0.999，说明模型学会了预测下一 token。
