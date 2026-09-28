# 教程：部署推理服务

```bash
python -m panguforge serve --config config.example.yaml --port 8000
```

```bash
curl http://127.0.0.1:8000/v1/completions -d '{"prompt":"你好","max_tokens":16}'
```
