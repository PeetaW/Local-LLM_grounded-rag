"""Local authenticated API tests without model/GPU or original RAG index."""
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from rag.research_agent.http_app import create_app


class AgentHttpTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.token = "local-testing-token-not-for-production-1234"
        self.app = create_app(self.root, self.token)
        self.client = TestClient(self.app)

    def test_missing_token_is_rejected(self):
        response = self.client.post("/agent/run", json={"question": "Analyze Raman"})
        self.assertEqual(response.status_code, 401)

    def test_incorrect_token_is_rejected(self):
        response = self.client.post("/agent/run",
                                    headers={"x-research-agent-token": "bad"},
                                    json={"question": "Analyze Raman"})
        self.assertEqual(response.status_code, 401)

    def test_missing_data_abstention_is_truthful(self):
        response = self.client.post(
            "/agent/run",
            headers={"x-research-agent-token": self.token},
            json={"question": "Analyze Raman", "mode": "deep"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "needs_input")
        self.assertFalse(data["assessment"]["passed"])
        self.assertEqual(data["attempts"], 1)

    def test_authenticated_presentation_creates_artifact(self):
        response = self.client.post(
            "/agent/run",
            headers={"x-research-agent-token": self.token},
            json={
                "question": "Generate a research PPTX",
                "inputs": {
                    "findings": [{
                        "topic": "Raman preliminary results",
                        "observation": "Peak position awaits confirmation",
                        "status": "tentative",
                    }],
                    "pptx_output": "out/research.pptx",
                }
            },
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["verification_level"], "artifact_and_provenance_only")
        self.assertTrue((self.root / "out/research.pptx").is_file())

    def test_too_short_token_fails_boot(self):
        with self.assertRaises(ValueError):
            create_app(self.root, "123")


if __name__ == "__main__":
    unittest.main()
