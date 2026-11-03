"""Explicit, inspectable intent routing for the local research assistant.

Do not trust model-chosen commands, file paths, or scientific method labels.
The router checks actual input availability separately from textual intent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .controller import ResearchMode, ResearchRequest
from .outcome_planner import Intent, detect_intent


@dataclass(frozen=True)
class RouteDecision:
    intent: Intent
    mode: ResearchMode
    permitted_task_types: tuple[str, ...]
    reason: str


_TASKS = {
    Intent.QUESTION: ("rag.query",),
    Intent.SPECTROSCOPY: ("spectroscopy.report",),
    Intent.PRESENTATION: ("presentation.from_discussion",),
    Intent.RESEARCH_REPORT: ("spectroscopy.report", "presentation.from_discussion"),
}


class IntentRouter:
    def route(self, request: ResearchRequest, inputs: Mapping[str, Any]) -> RouteDecision:
        kind = detect_intent(request.question, inputs)
        reason = {
            Intent.QUESTION: "Literature question: requires the existing indexed RAG",
            Intent.SPECTROSCOPY: "Spectroscopy request: requires caller-supplied instrument files",
            Intent.PRESENTATION: "Slide output: requires caller-supplied findings",
            Intent.RESEARCH_REPORT: "Spectroscopy plus editable slides: requires measured data",
        }[kind]
        return RouteDecision(kind, request.mode, _TASKS[kind], reason)
