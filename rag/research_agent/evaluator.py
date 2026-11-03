"""Deterministic artifact/evidence gate for Research Agent attempts.

Passing means required files and source pointers exist, NOT that a scientific
interpretation or a citation is scientifically correct. Benchmark gold and
expert checks remain separate from runtime checks.
"""
from __future__ import annotations

import json
from pathlib import Path

from .controller import AgentOutput, Assessment, FailureKind, ResearchRequest


class ResearchEvaluator:
    def __call__(self, request: ResearchRequest, output: AgentOutput) -> Assessment:
        meta = output.metadata or {}
        if meta.get("missing_input"):
            return Assessment(False, FailureKind.MISSING_INPUT, str(meta.get("reason", "Input missing")))
        if meta.get("tool_error"):
            return Assessment(False, FailureKind.TOOL_ERROR, str(meta["tool_error"])[:300])
        if not output.answer.strip():
            return Assessment(False, FailureKind.OMISSION, "Empty response")
        kind = meta.get("intent", "question")
        if kind == "question":
            if not output.evidence_refs or not meta.get("evidence_available"):
                return Assessment(False, FailureKind.EVIDENCE_GAP,
                                  "No traceable retrieved paper evidence")
        else:
            if kind in ("spectroscopy", "research_report"):
                path = Path(meta.get("analysis_json") or "")
                if not path.is_file():
                    return Assessment(False, FailureKind.OMISSION, "Spectral analysis JSON missing")
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    if not payload.get("measurements"):
                        raise ValueError("No spectral measurements")
                except (OSError, ValueError, TypeError) as exc:
                    return Assessment(False, FailureKind.OMISSION,
                                      f"Invalid analysis report: {type(exc).__name__}")
            if kind in ("presentation", "research_report"):
                path = Path(meta.get("pptx") or "")
                if not path.is_file() or path.stat().st_size < 1000:
                    return Assessment(False, FailureKind.OMISSION, "Editable PPTX missing")
        return Assessment(True)


class IndependentReviewer:
    """Read-only scientific review with optional LLM advisory, not a fake judge."""
    def __init__(self, llm=None):
        self.llm = llm

    def review(self, question: str, output: AgentOutput, profiles: tuple[str, ...]) -> dict:
        cautions = []
        if not output.evidence_refs:
            cautions.append("No source references; scientific claims require corroboration.")
        if output.metadata.get("analysis_json"):
            cautions.append("Candidate spectral peaks are not validated phase or mass fractions.")
        guidance = " ".join(profiles)[:2500]
        report = {
            "checks": list(profiles),
            "cautions": cautions,
            "advisory": "",
            "independent_evidence_verified": False,
        }
        if self.llm is not None:
            try:
                response = self.llm.structured(
                    "Act as a skeptical scientific peer reviewer, not a fact-checking oracle. "
                    "Identify assumptions, alternate explanations, and missing measurements. "
                    "Never claim to have consulted external evidence. Return JSON only.",
                    f"Question: {question[:1800]}\nDraft: {output.answer[:4000]}\nChecks: {guidance}",
                    {
                        "type": "object",
                        "properties": {"review": {"type": "string"}},
                        "required": ["review"],
                        "additionalProperties": False,
                    }
                )
                report["advisory"] = str(response["review"])[:2000]
            except (TypeError, ValueError, KeyError, OSError, RuntimeError):
                report["cautions"].append("Model-based advisory review unavailable")
        return report
