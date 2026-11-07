"""Manifest reader and local CLI offline checks, no Ollama required."""
import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_research_tools_smoke import _read_manifest, main


class ResearchCLIWorkflowTests(unittest.TestCase):
    def test_missing_file_fails_before_processing(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "manifest.json"
            p.write_text(json.dumps({"measurements": [
                {"path": "missing.csv", "technique": "xrd", "sample_id": "S1", "min_prominence": 10}
            ]}), encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                _read_manifest(p)

    def test_ft_ir_mode_is_required(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "ftir.csv").write_text("1,10\n2,20\n3,10\n", encoding="utf-8")
            p = root / "manifest.json"
            p.write_text(json.dumps({"measurements": [
                {"path": "ftir.csv", "technique": "ftir", "sample_id": "S1", "min_prominence": 3}
            ]}), encoding="utf-8")
            with self.assertRaises(ValueError):
                _read_manifest(p)

    def test_analysis_only_exits_successfully(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "sample.asc").write_text("100\t1\n101\t3\n102\t30\n103\t3\n104\t1", encoding="utf-8")
            p = root / "manifest.json"
            p.write_text(json.dumps({"measurements": [
                {"path": "sample.asc", "technique": "raman", "sample_id": "S1", "measurement_id":"spot1", "min_prominence": 10}
            ]}), encoding="utf-8")
            result = main(["--manifest", str(p), "--output", str(root/"out"), "--analysis-only"])
            self.assertEqual(result, 0)
            self.assertTrue((root/"out"/"analysis.json").is_file())
            self.assertFalse((root/"out"/"research_report.pptx").exists())


if __name__ == "__main__":
    unittest.main()
