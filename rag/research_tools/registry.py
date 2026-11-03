"""Explicit allowlist of offline research tools. No arbitrary code execution.

The caller supplies validated input dictionaries and must sandbox allowed file
paths before exposing these functions to an LLM. This is not an MCP server yet.
"""
from __future__ import annotations

from .report_pipeline import analyze_spectra_to_deck
from .presentation import DeckSpec, SlideSpec, export_research_deck
from .slide_planner import ResearchFinding, plan_deck_from_discussion


def run_named_tool(tool_name: str, arguments: dict):
    if tool_name == "spectroscopy.report":
        return analyze_spectra_to_deck(**arguments)
    if tool_name == "presentation.from_discussion":
        findings = [
            ResearchFinding(
                topic=f["topic"],
                observation=f["observation"],
                interpretation=f.get("interpretation", ""),
                evidence_refs=tuple(f.get("evidence_refs", ())),
                status=f.get("status", "observed"),
                figure_paths=tuple(f.get("figure_paths", ())),
            )
            for f in arguments["findings"]
        ]
        deck = plan_deck_from_discussion(
            arguments["title"], findings,
            next_steps=arguments.get("next_steps", ()),
            unresolved_questions=arguments.get("unresolved_questions", ()),
        )
        path = export_research_deck(deck, arguments["output_path"],
                                    template_path=arguments.get("template_path"))
        return {"pptx": str(path)}
    raise ValueError(f"Unknown or unauthorized research tool: {tool_name!r}")
