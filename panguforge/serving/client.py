"""Minimal HTTP client for the serving API."""
import json
import urllib.request


class Client:
    def __init__(self, base_url="http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")

    def complete(self, prompt, max_tokens=32):
        body = json.dumps({"prompt": prompt, "max_tokens": max_tokens}).encode()
        req = urllib.request.Request(self.base_url + "/v1/completions", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())

    def health(self):
        with urllib.request.urlopen(self.base_url + "/health", timeout=10) as r:
            return json.loads(r.read().decode())
