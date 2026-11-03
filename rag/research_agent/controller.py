"""Bounded, failure-aware research workflow. No model/tool side effects in this module."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


class ResearchMode(str, Enum):
    FAST = "fast"
    ADAPTIVE = "adaptive"
    DEEP = "deep"


class FailureKind(str, Enum):
    NONE = "none"
    EVIDENCE_GAP = "evidence_gap"
    OMISSION = "omission"
    CONTRADICTION = "contradiction"
    TOOL_ERROR = "tool_error"
    MISSING_INPUT = "missing_input"
    UNSUPPORTED = "unsupported"
    OTHER = "other"


class StopReason(str, Enum):
    SUCCESS = "success"
    NEEDS_INPUT = "needs_input"
    UNSUPPORTED = "unsupported"
    NO_REPAIR_AVAILABLE = "no_repair_available"
    NO_PROGRESS = "no_progress"
    REVISION_LIMIT = "revision_limit"
    TIME_BUDGET = "time_budget"
    TOKEN_BUDGET = "token_budget"


@dataclass(frozen=True)
class ResearchRequest:
    question: str
    outcome: str = ""
    acceptance_criteria: tuple[str, ...] = ()
    mode: ResearchMode = ResearchMode.ADAPTIVE

    def __post_init__(self):
        if not self.question.strip():
            raise ValueError("question cannot be blank")


@dataclass(frozen=True)
class AgentOutput:
    answer: str
    evidence_refs: tuple[str, ...] = ()
    tokens_used: int = 0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.tokens_used < 0:
            raise ValueError("tokens_used must be nonnegative")


@dataclass(frozen=True)
class Assessment:
    passed: bool
    failure: FailureKind = FailureKind.NONE
    explanation: str = ""

    def __post_init__(self):
        if self.passed != (self.failure == FailureKind.NONE):
            raise ValueError("passed and failure must agree")


@dataclass(frozen=True)
class IterationBudget:
    max_revisions: int
    max_total_tokens: Optional[int] = None
    max_elapsed_seconds: Optional[float] = None

    def __post_init__(self):
        if self.max_revisions < 0:
            raise ValueError("max_revisions cannot be negative")


@dataclass(frozen=True)
class RepairDirective:
    failure: FailureKind
    action: str
    explanation: str


@dataclass(frozen=True)
class ResearchResult:
    output: AgentOutput
    assessment: Assessment
    stop_reason: StopReason
    total_tokens: int
    elapsed_seconds: float
    attempts: int
    trace: tuple[dict, ...]


DEFAULT_BUDGETS = {
    ResearchMode.FAST: IterationBudget(0),
    ResearchMode.ADAPTIVE: IterationBudget(1, 32000, 1200.0),
    ResearchMode.DEEP: IterationBudget(2, 48000, 1800.0),
}

# Whitelisted actions: the LLM never supplies executable shell instructions.
REPAIR_ACTIONS = {
    FailureKind.EVIDENCE_GAP: "targeted_evidence_retrieval",
    FailureKind.OMISSION: "complete_missing_requirements",
    FailureKind.CONTRADICTION: "recheck_claim_against_source",
    FailureKind.TOOL_ERROR: "retry_failed_tool",
}


class ResearchAgent:
    """Coordinate injected executor/evaluator; does not claim model self-scores are truth.

    Token/time budgets are checked between synchronous calls only.
    Instrumentation must use measured tokens, not fabricated estimates.
    """
    def __init__(
        self,
        executor: Callable[[ResearchRequest, Optional[RepairDirective]], AgentOutput],
        evaluator: Callable[[ResearchRequest, AgentOutput], Assessment],
        budgets: Optional[dict[ResearchMode, IterationBudget]] = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.executor = executor
        self.evaluator = evaluator
        self.budgets = dict(DEFAULT_BUDGETS if budgets is None else budgets)
        self.clock = clock

    def run(self, request: ResearchRequest) -> ResearchResult:
        budget = self.budgets[request.mode]
        start = self.clock()
        trace: list[dict] = []
        tokens = 0
        directive = None
        seen_failures: set[tuple[str, str]] = set()
        while True:
            output = self.executor(request, directive)
            if not isinstance(output, AgentOutput):
                raise TypeError("executor must return AgentOutput")
            tokens += output.tokens_used
            assessment = self.evaluator(request, output)
            if not isinstance(assessment, Assessment):
                raise TypeError("evaluator must return Assessment")
            trace.append({
                "attempt": len(trace) + 1,
                "repair_action": directive.action if directive else None,
                "failure": assessment.failure.value,
                "passed": assessment.passed,
            })
            elapsed = max(0.0, self.clock() - start)
            if assessment.passed:
                reason = StopReason.SUCCESS
            elif assessment.failure == FailureKind.MISSING_INPUT:
                reason = StopReason.NEEDS_INPUT
            elif assessment.failure == FailureKind.UNSUPPORTED:
                reason = StopReason.UNSUPPORTED
            elif assessment.failure not in REPAIR_ACTIONS:
                reason = StopReason.NO_REPAIR_AVAILABLE
            elif len(trace) - 1 >= budget.max_revisions:
                reason = StopReason.REVISION_LIMIT
            elif budget.max_elapsed_seconds is not None and elapsed >= budget.max_elapsed_seconds:
                reason = StopReason.TIME_BUDGET
            elif budget.max_total_tokens is not None and tokens >= budget.max_total_tokens:
                reason = StopReason.TOKEN_BUDGET
            else:
                fingerprint = (assessment.failure.value, assessment.explanation.strip())
                if fingerprint in seen_failures:
                    reason = StopReason.NO_PROGRESS
                else:
                    seen_failures.add(fingerprint)
                    directive = RepairDirective(
                        assessment.failure, REPAIR_ACTIONS[assessment.failure], assessment.explanation
                    )
                    continue
            return ResearchResult(output, assessment, reason, tokens, elapsed, len(trace), tuple(trace))
