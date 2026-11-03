"""Opt-in Research Agent; independent of the production RAG path."""
from .controller import ResearchAgent, ResearchRequest, AgentOutput, Assessment, FailureKind, StopReason, ResearchMode, IterationBudget

__all__ = ["ResearchAgent", "ResearchRequest", "AgentOutput", "Assessment", "FailureKind", "StopReason", "ResearchMode", "IterationBudget"]
