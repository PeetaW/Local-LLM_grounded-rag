# rag/query_types.py
# Shared dataclasses for pipeline stage inputs/outputs.
# No LLM calls, no I/O — pure data structures.

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SubqueryTask:
    idx: int
    label: str
    engine: Any
    sub_q: str


@dataclass
class SubqueryResult:
    idx: int
    label: str
    result: str


@dataclass
class PipelineContext:
    """Carries all intermediate state across pipeline stages."""
    question: str
    memory_context: str = ""
    paper_names: list = field(default_factory=list)
    paper_engines_to_use: dict = field(default_factory=dict)
    sub_questions: list = field(default_factory=list)
    sub_answers: list = field(default_factory=list)
    rag_found_anything: bool = False
    knowledge_base: str = ""
    synthesis_prompt: str = ""
    fallback_notice: str = ""
    full_text: str = ""
    nli_report: str = ""
