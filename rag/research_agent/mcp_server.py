"""Local MCP v2 stdio server: precisely two sandboxed research tools.

Optional SDK: pip install 'mcp>=2,<3'. Start with:
  RESEARCH_WORKSPACE=<private-dir> python -m rag.research_agent.mcp_server
Never run this server from an untrusted remote HTTP endpoint.
"""
from __future__ import annotations

import os
from pathlib import Path

from .tool_router import ResearchToolRouter


def create_server(workspace_root: str | Path):
    from mcp.server import MCPServer
    router = ResearchToolRouter(workspace_root)
    server = MCPServer("Local Research Tools")

    @server.tool()
    def spectroscopy_report(
        measurements: list[dict], output_dir: str = "reports",
        deck_title: str = "Research spectra",
        template_path: str | None = None, create_pptx: bool = True,
    ) -> dict:
        """Analyze XRD/Raman/FTIR text XY spectra. Exploratory peaks only, no phase wt.%."""
        return router.run("spectroscopy.report", {
            "measurements": measurements, "output_dir": output_dir,
            "deck_title": deck_title, "template_path": template_path,
            "create_pptx": create_pptx,
        })

    @server.tool()
    def presentation_from_discussion(
        title: str, findings: list[dict],
        output_path: str = "reports/research.pptx",
        template_path: str | None = None,
        next_steps: list[str] | None = None,
        unresolved_questions: list[str] | None = None,
    ) -> dict:
        """Create evidence-tagged editable PPTX from supplied research findings."""
        return router.run("presentation.from_discussion", {
            "title": title, "findings": findings, "output_path": output_path,
            "template_path": template_path,
            "next_steps": next_steps or [],
            "unresolved_questions": unresolved_questions or [],
        })

    return server


def main() -> None:
    workspace = os.environ.get("RESEARCH_WORKSPACE")
    if not workspace:
        raise SystemExit("Set RESEARCH_WORKSPACE to an existing private directory")
    server = create_server(workspace)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
