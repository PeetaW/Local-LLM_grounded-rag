"""Optional MCP transport adapter for the ResearchWorkflow tool interface.

Default workflow uses the in-process allowlisted router for low latency.
Set router=MCPToolBackend(root) to use the official MCP stdio client instead.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from .mcp_client import call_private_stdio


MCP_MAP = {
    "spectroscopy.report": "spectroscopy_report",
    "presentation.from_discussion": "presentation_from_discussion",
}


class MCPToolBackend:
    def __init__(self, workspace_root: str | Path):
        self.workspace_root = Path(workspace_root).resolve(strict=True)

    def run(self, tool_name: str, arguments: dict) -> dict:
        if tool_name not in MCP_MAP:
            raise ValueError("Not an approved MCP tool name")
        # This synchronous adapter is intended for CLI/backgroundless local
        # agent flows, not for invocation inside an already-running event loop.
        return asyncio.run(call_private_stdio(
            self.workspace_root, MCP_MAP[tool_name], arguments
        ))
