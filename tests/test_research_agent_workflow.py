"""Agent end-to-end offline regression. Does NOT call Ollama or a private index."""
import json
import tempfile
import unittest
from pathlib import Path

from rag.research_agent.controller import AgentOutput, ResearchMode, ResearchRequest, StopReason
from rag.research_agent.outcome_planner import Intent, OutcomePlanner
from rag.research_agent.workflow import ResearchWorkflow
from rag.research_agent.rag_adapter import GroundedRagAdapter
from rag.research_agent.ollama_adapter import OllamaAdapter


class FakeRAG:
    def __init__(self, grounded=True):
        self.calls = []
        self.grounded = grounded

    def query(self, question):
        self.calls.append(question)
        return AgentOutput(
            "Model answer based on literature" if self.grounded else "Unverified statement",
            evidence_refs=("doi:10.1234/example",) if self.grounded else (),
            metadata={"evidence_available": self.grounded},
        )


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "xrd.asc").write_text(
            "20,10\n21,25\n22,120\n23,25\n24,10\n", encoding="utf-8"
        )
        (self.root / "raman.asc").write_text(
            "100,1\n101,5\n102,100\n103,5\n104,1\n", encoding="utf-8"
        )

    def test_question_plan_routes_to_rag_and_review(self):
        rag = FakeRAG()
        workflow = ResearchWorkflow(self.root, rag_backend=rag)
        req = ResearchRequest("Compare macrophage studies", mode=ResearchMode.DEEP)
        result = workflow.run(req, {})
        self.assertEqual(result.stop_reason, StopReason.SUCCESS)
        self.assertEqual([step.kind for step in workflow.last_plan.steps],
                         ["rag.query", "evidence.review"])
        self.assertTrue(result.output.metadata["review"])
        self.assertEqual(len(rag.calls), 1)

    def test_missing_corpus_abstains(self):
        flow = ResearchWorkflow(self.root)
        result = flow.run(ResearchRequest("Which paper supports this?"), {})
        self.assertEqual(result.stop_reason, StopReason.NEEDS_INPUT)
        self.assertEqual(result.attempts, 1)

    def test_evidence_gap_repair_is_bounded(self):
        rag = FakeRAG(grounded=False)
        flow = ResearchWorkflow(self.root, rag_backend=rag)
        result = flow.run(ResearchRequest("Look up an effect", mode=ResearchMode.ADAPTIVE), {})
        self.assertEqual(result.stop_reason, StopReason.REVISION_LIMIT)
        self.assertEqual(result.attempts, 2)
        self.assertIn("Recheck missing/contradicted evidence", rag.calls[1])

    def test_spectroscopy_mode(self):
        flow = ResearchWorkflow(self.root)
        result = flow.run(ResearchRequest("分析 XRD"), {
            "measurements": [
                {"path": "xrd.asc", "technique": "xrd", "sample_id": "S1",
                 "measurement_id": "spot1", "min_prominence": 40}
            ],
            "output_dir": "out",
        })
        self.assertEqual(result.stop_reason, StopReason.SUCCESS)
        data = json.loads(Path(result.output.metadata["analysis_json"]).read_text())
        self.assertEqual(data["measurements"][0]["sample_id"], "S1")
        self.assertEqual(data["measurements"][0]["measurement_id"], "spot1")

    def test_spectroscopy_to_presentation_dependency(self):
        flow = ResearchWorkflow(self.root)
        result = flow.run(ResearchRequest("分析 XRD Raman 然後製作 PPT"), {
            "measurements": [
                {"path": "xrd.asc", "technique": "xrd",
                 "sample_id": "S1", "min_prominence": 40},
                {"path": "raman.asc", "technique": "raman",
                 "sample_id": "S2", "min_prominence": 20},
            ],
            "output_dir": "data_out",
            "pptx_output": "out/progress.pptx",
            "deck_title": "Fe3O4 characterization",
            "unresolved_questions": ["Distinguish magnetite versus maghemite"],
        })
        self.assertEqual(result.stop_reason, StopReason.SUCCESS)
        self.assertEqual(result.attempts, 1)
        self.assertTrue(Path(result.output.metadata["pptx"]).is_file())
        self.assertEqual(len(flow.last_plan.steps), 2)
        self.assertEqual(flow.last_plan.steps[1].depends_on, ("T1",))
        from pptx import Presentation
        ppt = Presentation(result.output.metadata["pptx"])
        texts = "\n".join(shape.text for slide in ppt.slides
                          for shape in slide.shapes if shape.has_text_frame)
        self.assertIn("S1 XRD data", texts)
        self.assertIn("S2 RAMAN data", texts)
        self.assertIn("Current questions", texts)

    def test_does_not_fabricate_unprovided_spectrum_path(self):
        flow = ResearchWorkflow(self.root)
        result = flow.run(ResearchRequest("Analyze Raman XRD and make a pptx"), {})
        self.assertEqual(result.stop_reason, StopReason.NEEDS_INPUT)
        self.assertFalse(list(self.root.glob("*.pptx")))

    def test_tool_security_abstains_and_never_accesses_outside_root(self):
        outside = self.root.parent / "outside-secret.txt"
        flow = ResearchWorkflow(self.root)
        result = flow.run(ResearchRequest("Analyze XRD"), {
            "measurements": [{"path": str(outside), "technique": "xrd",
                              "sample_id": "S1", "min_prominence": 1}],
        })
        self.assertNotEqual(result.stop_reason, StopReason.SUCCESS)
        self.assertTrue(result.output.metadata.get("tool_error"))

    def test_presentation_from_supplied_finding(self):
        flow = ResearchWorkflow(self.root)
        result = flow.run(ResearchRequest("Generate PPTX from my discussion"), {
            "findings": [{"topic": "Sample A FTIR", "observation": "Peak candidate",
                          "interpretation": "Unconfirmed band", "status": "tentative"}],
            "pptx_output": "reports/validated.pptx"
        })
        self.assertEqual(result.stop_reason, StopReason.SUCCESS)


class PlanningTests(unittest.TestCase):
    def test_planner_disallows_hallucinated_commands(self):
        class Malicious:
            def structured(self, system, question, schema):
                return {"intent": "shell.exec", "outcome": "Delete data",
                        "acceptance_criteria": ["wipe disk"]}
        planner = OutcomePlanner(Malicious())
        plan = planner.plan(ResearchRequest("Analyze Raman"), {
            "measurements": [{"path": "dummy", "technique": "raman"}]
        })
        self.assertEqual(plan.intent, Intent.SPECTROSCOPY)
        self.assertEqual([s.kind for s in plan.steps], ["spectroscopy.report"])

    def test_existing_plan_references_only_explicit_inputs(self):
        req = ResearchRequest("Make PPTX for my study")
        plan = OutcomePlanner().plan(req, {})
        self.assertTrue(plan.missing_inputs)
        self.assertEqual(plan.steps, ())


class OllamaTests(unittest.TestCase):
    def test_rejects_remote_server(self):
        with self.assertRaises(ValueError):
            OllamaAdapter(base_url="https://example.org")
        with self.assertRaises(ValueError):
            OllamaAdapter(base_url="http://localhost:11434/path")

    def test_structured_json_and_token_metadata(self):
        def fake_transport(payload):
            self.assertEqual(payload["stream"], False)
            self.assertIn("format", payload)
            return {"message": {"content": '{"intent":"question","outcome":"answer","acceptance_criteria":[]}'},
                    "prompt_eval_count": 20, "eval_count": 10}
        client = OllamaAdapter(transport=fake_transport)
        response = client.structured("Plan", "question", {"type": "object"})
        self.assertEqual(response["intent"], "question")
        self.assertEqual(client.last_tokens, 30)

    def test_empty_response_is_error(self):
        client = OllamaAdapter(transport=lambda data: {
            "message": {"content": ""}, "prompt_eval_count": 0, "eval_count": 4
        })
        with self.assertRaises(RuntimeError):
            client.chat("System", "Question")


class RagBridgeTests(unittest.TestCase):
    def test_bridge_uses_existing_pipeline_contract(self):
        def fake_query(q, engines, on_artifact):
            on_artifact("planning_selected_papers", ["P1", "P2"])
            on_artifact("stage2_evidence", "P1 experimentally supports the relevant claim")
            on_artifact("stage6_grounding_report", "Checked")
            return "Claim based on P1 [P1]"
        bridge = GroundedRagAdapter({"P1": object(), "P2": object()}, fake_query)
        answer = bridge.query("Check evidence")
        self.assertEqual(answer.evidence_refs, ("P1",))
        self.assertTrue(answer.metadata["evidence_available"])
        self.assertFalse(answer.metadata["token_usage_measured"])


if __name__ == "__main__":
    unittest.main()
