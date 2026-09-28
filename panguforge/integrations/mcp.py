"""Model Context Protocol (MCP) tool server for panguforge."""
import json


class MCPToolServer:
    """Expose panguforge capabilities as MCP tools."""

    def __init__(self, engine):
        self.engine = engine

    def tools(self):
        return [
            {"name": "panguforge_complete", "description": "Complete a text prompt using the panguforge model",
             "inputSchema": {"type": "object", "properties": {"prompt": {"type": "string"}, "max_tokens": {"type": "integer"}}}},
            {"name": "panguforge_status", "description": "Get current engine metrics", "inputSchema": {"type": "object"}},
        ]

    def call(self, name, arguments):
        if name == "panguforge_complete":
            return {"text": self.engine.generate(arguments.get("prompt", ""), int(arguments.get("max_tokens", 32)))}
        if name == "panguforge_status":
            return self.engine.metrics()
        raise ValueError(f"unknown tool {name}")
