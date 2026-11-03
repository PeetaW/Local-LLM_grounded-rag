"""Offline workflow contract tests; run with unittest, without Ollama or a GPU."""
import unittest

from rag.research_agent import (
    ResearchAgent, ResearchRequest, AgentOutput, Assessment,
    FailureKind, StopReason, ResearchMode, IterationBudget,
)


class AgentTests(unittest.TestCase):
    def test_fast_no_revision(self):
        calls = []
        def executor(req, directive):
            calls.append(directive)
            return AgentOutput("draft")
        agent = ResearchAgent(executor, lambda req, out: Assessment(False, FailureKind.OMISSION, "missing"))
        result = agent.run(ResearchRequest("Explain", mode=ResearchMode.FAST))
        self.assertEqual(result.stop_reason, StopReason.REVISION_LIMIT)
        self.assertEqual(calls, [None])

    def test_adaptive_targeted_repair(self):
        actions = []
        def executor(req, directive):
            actions.append(None if directive is None else directive.action)
            return AgentOutput("good" if directive else "bad", tokens_used=10)
        def evaluator(req, out):
            if out.answer == "good":
                return Assessment(True)
            return Assessment(False, FailureKind.EVIDENCE_GAP, "need source")
        result = ResearchAgent(executor, evaluator).run(ResearchRequest("Research"))
        self.assertEqual(result.stop_reason, StopReason.SUCCESS)
        self.assertEqual(actions, [None, "targeted_evidence_retrieval"])
        self.assertEqual(result.total_tokens, 20)

    def test_stops_missing_data(self):
        agent = ResearchAgent(
            lambda req, direction: AgentOutput("no data"),
            lambda req, out: Assessment(False, FailureKind.MISSING_INPUT, "upload spectrum"),
        )
        self.assertEqual(agent.run(ResearchRequest("Analyze")).stop_reason, StopReason.NEEDS_INPUT)

    def test_repeat_fails_without_loop(self):
        agent = ResearchAgent(
            lambda req, direction: AgentOutput("unchanged"),
            lambda req, out: Assessment(False, FailureKind.CONTRADICTION, "same source"),
        )
        result = agent.run(ResearchRequest("Check", mode=ResearchMode.DEEP))
        self.assertEqual(result.stop_reason, StopReason.NO_PROGRESS)
        self.assertEqual(result.attempts, 2)

    def test_token_budget(self):
        agent = ResearchAgent(
            lambda req, direction: AgentOutput("draft", tokens_used=22),
            lambda req, out: Assessment(False, FailureKind.TOOL_ERROR, "timeout"),
            budgets={ResearchMode.ADAPTIVE: IterationBudget(2, max_total_tokens=10)},
        )
        self.assertEqual(agent.run(ResearchRequest("Test")).stop_reason, StopReason.TOKEN_BUDGET)

    def test_validation(self):
        with self.assertRaises(ValueError):
            ResearchRequest(" ")
        with self.assertRaises(ValueError):
            AgentOutput("bad", tokens_used=-1)
        with self.assertRaises(ValueError):
            Assessment(True, FailureKind.OMISSION)


if __name__ == "__main__":
    unittest.main()
