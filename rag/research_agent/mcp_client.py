"""MCP v2 client for the local, explicitly allowlisted research tools.

Both in-process tests and a real stdio subprocess use the official protocol.
Do not use remote network transports or permit model-chosen commands/servers.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


TOOL_NAMES = frozenset({"spectroscopy_report", "presentation_from_discussion"})


def _extract(result) -> dict:
    if getattr(result, "is_error", False):
        raise RuntimeError("MCP tool returned an error")
    data = getattr(result, "structured_content", None)
    if isinstance(data, dict):
        # SDK may wrap non-structured return values.
        if isinstance(data.get("result"), dict):
            return data["result"]
        return data
    import json
    for part in result.content:
        raw = getattr(part, "text", None)
        if raw:
            value = json.loads(raw)
            if isinstance(value, dict):
                return value
    raise RuntimeError("MCP did not return a JSON object")


async def call_inprocess(server, name: str, arguments: dict[str, Any]) -> dict:
    from mcp import Client
    if name not in TOOL_NAMES:
        raise ValueError("Unapproved MCP tool")
    async with Client(server, raise_exceptions=True) as client:
        result = await client.call_tool(name, arguments)
        return _extract(result)


async def call_private_stdio(
    workspace_root: str | Path, name: str, arguments: dict[str, Any]
) -> dict:
    """Start ONLY this repository's server with this Python interpreter."""
    from mcp import Client, StdioServerParameters
    if name not in TOOL_NAMES:
        raise ValueError("Unapproved MCP tool")
    root = Path(workspace_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Private research workspace must exist")
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "rag.research_agent.mcp_server"],
        env={"RESEARCH_WORKSPACE": str(root)},
    )
    async with Client(params) as client:
        return _extract(await client.call_tool(name, arguments))
