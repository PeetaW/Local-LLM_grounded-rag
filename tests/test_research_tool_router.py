"""Fail-closed tests for future Research Agent tool routing."""
import os
import tempfile
import unittest
from pathlib import Path

from rag.research_agent.tool_router import ResearchToolRouter, ToolSecurityError


class RouterSecurityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.workspace = root / "workspace"
        self.workspace.mkdir()
        self.router = ResearchToolRouter(self.workspace)
        self.source = self.workspace / "sample.asc"
        self.source.write_text("100,1\n101,4\n102,40\n103,4\n104,1", encoding="utf-8")

    def test_allowlisted_spectrum_report(self):
        result = self.router.run("spectroscopy.report", {
            "measurements": [{"path": "sample.asc", "technique": "raman",
                              "sample_id": "S1", "min_prominence": 12}],
            "output_dir": "reports",
            "create_pptx": False,
        })
        self.assertTrue(Path(result["json"]).is_file())
        self.assertIsNone(result["pptx"])

    def test_unknown_tool_rejected(self):
        with self.assertRaises(ToolSecurityError):
            self.router.run("shell.exec", {"command": "echo should_not_execute"})

    def test_path_traversal_rejected(self):
        external = Path(self.temp.name) / "private.asc"
        external.write_text("1,2\n2,3\n3,4", encoding="utf-8")
        with self.assertRaises(ToolSecurityError):
            self.router.run("spectroscopy.report", {
                "measurements": [{"path": "../private.asc", "technique": "raman",
                                  "sample_id": "S1", "min_prominence": 10}],
                "output_dir": "out",
            })

    def test_output_escape_rejected(self):
        with self.assertRaises(ToolSecurityError):
            self.router.run("spectroscopy.report", {
                "measurements": [{"path": "sample.asc", "technique": "raman",
                                  "sample_id": "S1", "min_prominence": 10}],
                "output_dir": "../outside",
            })

    def test_unknown_argument_rejected(self):
        with self.assertRaises(ToolSecurityError):
            self.router.run("spectroscopy.report", {
                "measurements": [{"path": "sample.asc", "technique": "raman",
                                  "sample_id": "S1", "min_prominence": 10}],
                "output_dir": "out", "script": "rm -rf ...",
            })

    def test_symlink_escape_rejected(self):
        external = Path(self.temp.name) / "private.asc"
        external.write_text("1,2\n2,3\n3,4", encoding="utf-8")
        link = self.workspace / "link.asc"
        try:
            os.symlink(external, link)
        except (OSError, NotImplementedError):
            self.skipTest("symlink unavailable")
        with self.assertRaises(ToolSecurityError):
            self.router.run("spectroscopy.report", {
                "measurements": [{"path": "link.asc", "technique": "raman",
                                  "sample_id": "S1", "min_prominence": 10}],
                "output_dir": "out",
            })

    def test_presentation_rejects_external_figure(self):
        f = Path(self.temp.name) / "chart.png"
        f.write_bytes(b"not a real png")
        with self.assertRaises(ToolSecurityError):
            self.router.run("presentation.from_discussion", {
                "title": "Test", "output_path": "deck.pptx",
                "findings": [{"topic": "Raman", "observation": "Candidate",
                              "figure_paths": [str(f)]}],
            })


if __name__ == "__main__":
    unittest.main()
