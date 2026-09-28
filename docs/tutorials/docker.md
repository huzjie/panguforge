# 教程：Docker 部署

```bash
docker build -t panguforge .
docker run -p 8000:8000 panguforge
curl http://127.0.0.1:8000/health
```
