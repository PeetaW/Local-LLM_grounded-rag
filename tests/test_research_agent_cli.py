"""Local CLI end-to-end without Ollama or private PDF indexes."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_research_agent import run_cli


class FullAgentCLITests(unittest.TestCase):
    def test_cli_spectra_to_editable_pptx(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "sample.asc"
            source.write_text("100,1\n101,5\n102,90\n103,5\n104,1", encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"measurements": [{
                "path": "sample.asc", "technique": "raman", "sample_id": "S1",
                "measurement_id": "spot1", "min_prominence": 12,
            }]}), encoding="utf-8")
            args = argparse.Namespace(
                workspace=str(root), question="Analyze Raman and make PPTX",
                manifest=str(manifest), findings_json=None, template=None,
                output_dir="results", pptx_output="results/report.pptx",
                deck_title="Raman progress", mode="deep",
                ollama_planner=False, model="qwen3.5:27b",
                use_rag=False, mcp=False
            )
            result = run_cli(args)
            self.assertEqual(result["stop_reason"], "success")
            self.assertEqual(len(result["planned_tasks"]), 3)
            self.assertTrue((root / "results/report.pptx").is_file())
            self.assertTrue((root / "agent_run_result.json").is_file())

    def test_cli_does_not_claim_success_without_spectrum(self):
        with tempfile.TemporaryDirectory() as temp:
            args = argparse.Namespace(
                workspace=temp, question="Analyze Raman", manifest=None,
                findings_json=None, template=None, output_dir="results",
                pptx_output="results/report.pptx", deck_title="Research",
                mode="adaptive", ollama_planner=False, model="qwen3.5:27b",
                use_rag=False, mcp=False
            )
            result = run_cli(args)
            self.assertEqual(result["stop_reason"], "needs_input")


if __name__ == "__main__":
    unittest.main()
