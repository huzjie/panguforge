"""Stdlib HTTP server exposing an OpenAI-compatible API."""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

from .router import Router
from .protocol import CompletionRequest, CompletionResponse, CompletionChoice


def make_handler(engine):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/health":
                self._send(200, {"status": "ok", "backend": engine.name})
            elif path == "/metrics":
                self._send(200, engine.metrics())
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            path = urlparse(self.path).path
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            if path == "/v1/completions":
                req = CompletionRequest.from_dict(body)
                tokens = engine.generate(req.prompt, max_tokens=req.max_tokens)
                text = "".join(chr(t % 128) for t in tokens) if isinstance(tokens, list) else str(tokens)
                resp = CompletionResponse(id="cmpl-0", choices=[CompletionChoice(text=text)])
                self._send(200, resp.to_dict())
            elif path == "/v1/chat/completions":
                prompt = ""
                for m in body.get("messages", []):
                    prompt += str(m.get("content", ""))
                tokens = engine.generate(prompt, max_tokens=int(body.get("max_tokens", 32)))
                text = "".join(chr(t % 128) for t in tokens) if isinstance(tokens, list) else str(tokens)
                self._send(200, {"id": "chatcmpl-0", "choices": [{"message": {"role": "assistant", "content": text}}]})
            else:
                self._send(404, {"error": "not found"})

        def log_message(self, *args):
            pass

    return Handler


def create_server(engine, host="127.0.0.1", port=8000):
    return HTTPServer((host, port), make_handler(engine))
