"""Request router: maps an HTTP route to a handler."""
import json


class Router:
    def __init__(self):
        self.routes = {}

    def add(self, path, handler):
        self.routes[path] = handler

    def dispatch(self, path, body):
        handler = self.routes.get(path)
        if handler is None:
            return 404, {"error": f"unknown route {path}"}
        try:
            return 200, handler(body)
        except Exception as e:
            return 500, {"error": str(e)}
