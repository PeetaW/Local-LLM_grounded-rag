"""Optional local-only Research Agent API (separate from the original api.py).

Requires RESEARCH_AGENT_WORKSPACE and RESEARCH_AGENT_TOKEN environment variables.
Run: python -m rag.research_agent.http_app
Only binds 127.0.0.1:8765. Never expose without separate production-grade auth.
The frozen production seven-stage endpoints are not modified.
"""
from __future__ import annotations

import hmac
import os
from pathlib import Path

from .controller import ResearchMode, ResearchRequest
from .workflow import ResearchWorkflow


def create_app(workspace_root: str | Path, auth_token: str, *,
               rag_backend=None, planning_llm=None, router=None):
    from fastapi import FastAPI, Header, HTTPException
    from pydantic import BaseModel, Field
    from starlette.concurrency import run_in_threadpool

    if not isinstance(auth_token, str) or len(auth_token) < 24:
        raise ValueError("Research Agent token must be >=24 characters")
    root = Path(workspace_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Research Agent workspace does not exist")
    workflow = ResearchWorkflow(root, rag_backend=rag_backend,
                                planning_llm=planning_llm, router=router)

    class AgentInput(BaseModel):
        question: str = Field(min_length=1, max_length=6000)
        mode: ResearchMode = ResearchMode.ADAPTIVE
        inputs: dict = Field(default_factory=dict)

    # Pydantic resolves type annotations inside FastAPI route closures from
    # module globals. Set a stable public class name for this generated model.
    AgentInput.__name__ = "AgentInput"

    app = FastAPI(title="Local Research Agent", docs_url=None, redoc_url=None)

    @app.get("/health")
    def health():
        return {"status": "ready", "scope": "local-research-agent"}

    @app.post("/agent/run")
    async def run_agent(
        body: AgentInput,
        x_research_agent_token: str = Header(default=""),
    ):
        if not hmac.compare_digest(x_research_agent_token, auth_token):
            raise HTTPException(status_code=401, detail="Not authorized")
        try:
            request = ResearchRequest(body.question, mode=body.mode)
            result = await run_in_threadpool(workflow.run, request, body.inputs)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=type(exc).__name__) from exc
        return {
            "status": result.stop_reason.value,
            "attempts": result.attempts,
            "answer": result.output.answer,
            "sources": list(result.output.evidence_refs),
            "assessment": {"passed": result.assessment.passed,
                           "failure": result.assessment.failure.value,
                           "reason": result.assessment.explanation},
            "artifacts": {k: result.output.metadata.get(k)
                          for k in ("analysis_json", "pptx", "plots", "review")},
            "verification_level": result.output.metadata.get("verification_level"),
            "trace": list(result.trace),
        }

    return app


def main() -> None:
    workspace = os.environ.get("RESEARCH_AGENT_WORKSPACE")
    token = os.environ.get("RESEARCH_AGENT_TOKEN")
    if not workspace or not token:
        raise SystemExit("Set RESEARCH_AGENT_WORKSPACE and RESEARCH_AGENT_TOKEN")
    from .ollama_adapter import OllamaAdapter
    planning_llm = None
    if os.environ.get("RESEARCH_AGENT_ENABLE_OLLAMA") == "1":
        planning_llm = OllamaAdapter(model=os.environ.get("RESEARCH_AGENT_MODEL",
                                                          "qwen3.5:27b"))
    rag_backend = None
    if os.environ.get("RESEARCH_AGENT_ENABLE_RAG") == "1":
        from main import paper_engines
        from .rag_adapter import GroundedRagAdapter
        rag_backend = GroundedRagAdapter(paper_engines)
    app = create_app(workspace, token, rag_backend=rag_backend,
                     planning_llm=planning_llm)
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="warning")


if __name__ == "__main__":
    main()
