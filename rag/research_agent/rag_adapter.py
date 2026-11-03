"""Lazy adapter to the ORIGINAL seven-stage academic RAG pipeline.

The evidence origin is visible in returned metadata. This adapter does not
claim that selected source titles establish correctness of individual claims.
No mutation of production config, indexes or query flow is performed.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .controller import AgentOutput


class GroundedRagAdapter:
    def __init__(self, paper_engines: dict, query_fn: Callable | None = None):
        if not isinstance(paper_engines, dict):
            raise ValueError("paper_engines must be mapping of indexed papers")
        self.paper_engines = paper_engines
        self._query_fn = query_fn

    def query(self, question: str) -> AgentOutput:
        if not self.paper_engines:
            return AgentOutput(
                "No indexed papers are available. Load a corpus first.",
                metadata={"missing_input": True, "rag_backend": "seven-stage"},
            )
        if self._query_fn is None:
            # Heavy LlamaIndex/torch imports only occur during the actual local run.
            from rag.query_pipeline import execute_structured_query
            fn = execute_structured_query
        else:
            fn = self._query_fn

        artifacts: dict[str, Any] = {}
        def capture(name: str, value: Any):
            # Capture bounded metadata, not full scientific corpora in the trace.
            if name in ("stage2_evidence", "planning_selected_papers", "stage6_grounding_report"):
                artifacts[name] = value
        answer = fn(question, self.paper_engines, on_artifact=capture)
        evidence = str(artifacts.get("stage2_evidence") or "")
        chosen = artifacts.get("planning_selected_papers") or []
        if not isinstance(chosen, list):
            chosen = []
        # Only selected sources for which actual evidence was produced are
        # exposed as pointers; don't claim this proves each cited statement.
        refs = tuple(str(s) for s in chosen if str(s) in evidence)[:20]
        has_evidence = bool(evidence.strip()) and bool(refs)
        return AgentOutput(
            answer=str(answer or ""),
            evidence_refs=refs,
            # Existing RAG does not return a reliable aggregate per-request
            # token count: zero means 'not measured', NOT 'zero GPU usage'.
            tokens_used=0,
            metadata={
                "source": "seven-stage-rag",
                "evidence_available": has_evidence,
                "token_usage_measured": False,
                "grounding_report_available": bool(artifacts.get("stage6_grounding_report")),
            },
        )
