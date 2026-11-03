"""Opt-in Research Agent orchestration: plan -> execute -> evaluate -> repair.

A complete mock-testable workflow. LLMs cannot choose arbitrary file paths
or command names. The existing seven-stage RAG is called only via a bridge,
and remains unchanged. File tools execute through the sandboxed router.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .controller import (
    AgentOutput, FailureKind, ResearchAgent, ResearchRequest,
    RepairDirective, ResearchResult,
)
from .evaluator import IndependentReviewer, ResearchEvaluator
from .intent_router import IntentRouter
from .expert_profiles import profiles_for
from .outcome_planner import OutcomePlan, OutcomePlanner
from .tool_router import ResearchToolRouter


class ResearchWorkflow:
    def __init__(
        self, workspace_root: str | Path,
        *,
        rag_backend=None,
        planning_llm=None,
        review_llm=None,
        planner: OutcomePlanner | None = None,
        router: ResearchToolRouter | None = None,
        evaluator=None,
    ):
        self.router = router or ResearchToolRouter(workspace_root)
        self.rag_backend = rag_backend
        self.llm = planning_llm
        self.intent_router = IntentRouter()
        self.planner = planner or OutcomePlanner(planning_llm)
        self.reviewer = IndependentReviewer(review_llm)
        self.evaluator = evaluator or ResearchEvaluator()
        self.last_plan: OutcomePlan | None = None

    def _run_steps(
        self, request: ResearchRequest, inputs: Mapping[str, Any],
        plan: OutcomePlan, repair: RepairDirective | None,
    ) -> AgentOutput:
        if plan.missing_inputs:
            return AgentOutput(
                answer="Required inputs are missing: " + "; ".join(plan.missing_inputs),
                metadata={"missing_input": True, "reason": "; ".join(plan.missing_inputs),
                          "intent": plan.intent.value},
            )
        products: dict[str, Any] = {}
        references: list[str] = []
        parts: list[str] = []
        meta = {"intent": plan.intent.value, "plan_outcome": plan.outcome,
                "verification_level": "artifact_and_provenance_only",
                "completed_tasks": [], "review": None,
                "evidence_available": False}
        try:
            for step in plan.steps:
                if not all(dep in products for dep in step.depends_on):
                    raise ValueError(f"Missing predecessor task for {step.id}")
                if step.kind == "rag.query":
                    if self.rag_backend is None:
                        return AgentOutput(
                            "RAG backend is not configured",
                            metadata={"missing_input": True, "reason": "RAG backend missing",
                                      "intent": plan.intent.value},
                        )
                    q = request.question
                    if repair is not None and repair.failure in (
                        FailureKind.EVIDENCE_GAP, FailureKind.CONTRADICTION,
                        FailureKind.OMISSION,
                    ):
                        # Bounded targeted evidence request; not arbitrary model code.
                        q += "\nRecheck missing/contradicted evidence: " + repair.explanation[:280]
                    result = self.rag_backend.query(q)
                    if not isinstance(result, AgentOutput):
                        raise TypeError("RAG adapter must return AgentOutput")
                    parts.append(result.answer)
                    references.extend(result.evidence_refs)
                    meta["evidence_available"] = bool(result.metadata.get("evidence_available"))
                    if result.metadata.get("missing_input"):
                        meta["missing_input"] = True
                        meta["reason"] = result.answer
                    products[step.id] = result

                elif step.kind == "spectroscopy.report":
                    out_dir = str(inputs.get("output_dir") or "results")
                    result = self.router.run("spectroscopy.report", {
                        "measurements": list(inputs["measurements"]),
                        "output_dir": out_dir,
                        "deck_title": str(inputs.get("deck_title") or request.question)[:150],
                        "template_path": inputs.get("template_path"),
                        "create_pptx": plan.intent.value == "spectroscopy" and bool(
                            inputs.get("create_pptx", True)
                        ),
                    })
                    products[step.id] = result
                    meta["analysis_json"] = result["json"]
                    if result.get("pptx"):
                        meta["pptx"] = result["pptx"]
                    meta["plots"] = result.get("plots", [])
                    parts.append(f"Spectral exploratory analysis: {result['json']}")
                    references.extend(str(item.get("path")) for item in inputs["measurements"]
                                      if item.get("path"))

                elif step.kind == "presentation.from_discussion":
                    findings = [dict(f) for f in inputs.get("findings", [])]
                    if plan.intent.value == "research_report":
                        import json
                        analysis = products["T1"]
                        with open(analysis["json"], encoding="utf-8") as stream:
                            report = json.load(stream)
                        for measurement, plot in zip(report["measurements"], analysis["plots"]):
                            labels = [f"{p['x']:.2f}" for p in measurement.get("peaks", [])[:8]]
                            unit = measurement.get("x_unit", "")
                            peaks_text = ", ".join(labels) if labels else "none above threshold"
                            findings.append({
                                "topic": (f"{measurement['sample_id']} "
                                          f"{measurement['technique'].upper()} data"),
                                "observation": f"Screened peak candidates ({unit}): {peaks_text}",
                                "interpretation": "Peak screening does not establish phase identity or purity",
                                "evidence_refs": [Path(measurement["source_path"]).name],
                                "status": "tentative",
                                "figure_paths": [plot],
                            })
                    result = self.router.run("presentation.from_discussion", {
                        "title": str(inputs.get("deck_title") or plan.outcome)[:150],
                        "findings": findings,
                        "next_steps": list(inputs.get("next_steps", [])),
                        "unresolved_questions": list(inputs.get("unresolved_questions", [])),
                        "output_path": str(inputs.get("pptx_output") or "results/research_deck.pptx"),
                        "template_path": inputs.get("template_path"),
                    })
                    products[step.id] = result
                    meta["pptx"] = result["pptx"]
                    parts.append(f"Editable research presentation: {result['pptx']}")
                    for f in findings:
                        references.extend(str(x) for x in f.get("evidence_refs", []))

                elif step.kind == "evidence.review":
                    r = self.reviewer.review(
                        request.question,
                        AgentOutput("\n".join(parts), tuple(references),
                                    metadata=meta),
                        profiles_for(
                            [str(x.get("technique", "")) for x in inputs.get("measurements", [])],
                            presentation=plan.intent.value in ("presentation", "research_report"),
                        ),
                    )
                    products[step.id] = r
                    meta["review"] = r
                    parts.append("Reviewer advisory: " + (
                        r["advisory"] or "; ".join(r["cautions"]) or "Check source claims manually"
                    ))

                else:
                    raise RuntimeError(f"Unknown planned task: {step.kind}")
                meta["completed_tasks"].append(step.id)
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
            # Return a typed failure rather than pretending a completed report.
            meta["tool_error"] = f"{type(exc).__name__}: {str(exc)[:240]}"
            return AgentOutput("The research workflow failed safely.", tuple(references), metadata=meta)

        return AgentOutput(
            "\n".join(parts), tuple(dict.fromkeys(references)),
            metadata=meta,
        )

    def run(self, request: ResearchRequest, inputs: Mapping[str, Any]) -> ResearchResult:
        # No unbounded task expansion, and plan is fixed across repairs.
        validated = dict(inputs)
        validated["rag_available"] = self.rag_backend is not None
        route = self.intent_router.route(request, validated)
        plan = self.planner.plan(request, validated)
        if plan.intent != route.intent:
            raise ValueError('Planner intent diverged from validated route')
        self.last_plan = plan
        planning_tokens = int(getattr(self.llm, "last_tokens", 0) or 0) if self.llm else 0
        first = True

        def execute(req: ResearchRequest, repair: RepairDirective | None) -> AgentOutput:
            nonlocal first
            before = int(getattr(self.llm, "total_tokens", 0) or 0) if self.llm else 0
            output = self._run_steps(req, validated, plan, repair)
            after = int(getattr(self.llm, "total_tokens", 0) or 0) if self.llm else 0
            tokens = max(0, after - before) + (planning_tokens if first else 0)
            first = False
            return AgentOutput(output.answer, output.evidence_refs,
                               tokens_used=tokens, metadata=output.metadata)

        return ResearchAgent(execute, self.evaluator).run(request)
