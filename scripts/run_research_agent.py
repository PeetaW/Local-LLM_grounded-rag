"""Opt-in local Research Agent CLI (never modifies production API or model settings).

Example:
python -m scripts.run_research_agent --workspace E:/PrivateResearch \
  --question "Analyze XRD Raman and make PPTX" \
  --manifest E:/PrivateResearch/manifest.json \
  --template E:/PrivateResearch/owner_template.pptx --mode deep

Research-only, not a replacement for the frozen V12.1 pipeline benchmark.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from rag.research_agent.controller import ResearchMode, ResearchRequest, StopReason
from rag.research_agent.workflow import ResearchWorkflow
from rag.research_agent.mcp_backend import MCPToolBackend
from rag.research_agent.ollama_adapter import OllamaAdapter
from rag.research_agent.rag_adapter import GroundedRagAdapter
from scripts.run_research_tools_smoke import _read_manifest


def run_cli(args) -> dict:
    root = Path(args.workspace).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Research workspace must be a directory")
    manifest = Path(args.manifest).resolve(strict=True) if args.manifest else None
    template = Path(args.template).resolve(strict=True) if args.template else None
    inputs = {
        "measurements": _read_manifest(manifest) if manifest else [],
        "template_path": str(template) if template else None,
        "output_dir": args.output_dir,
        "pptx_output": args.pptx_output,
        "deck_title": args.deck_title,
        "findings": [],
    }
    if args.findings_json:
        path = Path(args.findings_json).resolve(strict=True)
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("findings"), list):
            raise ValueError("findings-json must contain a findings array")
        inputs["findings"] = data["findings"]
        inputs["next_steps"] = data.get("next_steps", [])
        inputs["unresolved_questions"] = data.get("unresolved_questions", [])
    planning_llm = (OllamaAdapter(model=args.model) if args.ollama_planner else None)
    rag = None
    if args.use_rag:
        # Heavy initialization, indexes and GPU models only when explicitly asked.
        from main import paper_engines
        rag = GroundedRagAdapter(paper_engines)
    router = MCPToolBackend(root) if args.mcp else None
    workflow = ResearchWorkflow(
        root, rag_backend=rag, planning_llm=planning_llm,
        review_llm=planning_llm if args.mode == "deep" else None,
        router=router,
    )
    result = workflow.run(
        ResearchRequest(question=args.question, mode=ResearchMode(args.mode)), inputs
    )
    payload = {
        "outcome": workflow.last_plan.outcome,
        "intent": workflow.last_plan.intent.value,
        "planned_tasks": [{"id": x.id, "kind": x.kind,
                           "depends_on": list(x.depends_on)}
                          for x in workflow.last_plan.steps],
        "stop_reason": result.stop_reason.value,
        "assessment": {"passed": result.assessment.passed,
                       "failure": result.assessment.failure.value,
                       "reason": result.assessment.explanation},
        "attempts": result.attempts, "reported_tokens": result.total_tokens,
        "answer": result.output.answer,
        "references": list(result.output.evidence_refs),
        "artifacts": {k: result.output.metadata.get(k)
                      for k in ("analysis_json", "pptx", "plots", "review")},
        "trace": list(result.trace),
    }
    path = root / "agent_run_result.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "result_file": str(path),
        "stop_reason": payload["stop_reason"],
        "attempts": payload["attempts"],
        "artifacts": payload["artifacts"],
    }, indent=2, ensure_ascii=False))
    return payload


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", required=True)
    p.add_argument("--question", required=True)
    p.add_argument("--manifest", help="Private data manifest JSON")
    p.add_argument("--findings-json", help="Private research discussion findings JSON")
    p.add_argument("--template", help="Private PPTX")
    p.add_argument("--output-dir", default="reports")
    p.add_argument("--pptx-output", default="reports/research_deck.pptx")
    p.add_argument("--deck-title", default="Research Progress Report")
    p.add_argument("--mode", choices=[x.value for x in ResearchMode], default="adaptive")
    p.add_argument("--ollama-planner", action="store_true", help="Use local Ollama JSON goal planner")
    p.add_argument("--model", default="qwen3.5:27b")
    p.add_argument("--use-rag", action="store_true", help="Load original indexed RAG; can be slow")
    p.add_argument("--mcp", action="store_true", help="Invoke tools through local MCP stdio")
    args = p.parse_args(argv)
    report = run_cli(args)
    return 0 if report["stop_reason"] == StopReason.SUCCESS.value else 2


if __name__ == "__main__":
    raise SystemExit(main())
