"""Offline integration test for data -> JSON / plot / editable PPTX.

Uses synthetic spectra with instrument-export-like headers and sample IDs.
No user's private raw spectra or slide masters are committed.
"""
import json
import tempfile
import unittest
from pathlib import Path

from rag.research_tools.report_pipeline import analyze_spectra_to_deck


class ReportWorkflowTests(unittest.TestCase):
    def test_two_independent_samples_and_multi_spot(self):
        try:
            from pptx import Presentation
            import scipy  # noqa: F401
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("Install optional research-tool dependencies")
        with tempfile.TemporaryDirectory() as base:
            root = Path(base)
            s1 = root / "xrd_sample_1.txt"
            s2 = root / "xrd_sample_2.txt"
            ftir = root / "ftir_s1_spot2.csv"
            s1.write_text(
                ";RAW4.00\n[RawHeader]\n[RangeHeader]\nStart=5\n"
                "[Data]\nAngle, PSD,\n"
                "20,10,\n21,20,\n22,100,\n23,20,\n24,10,\n",
                encoding="utf-8",
            )
            s2.write_text(
                ";RAW4.00\n[RawHeader]\n[RangeHeader]\nStart=5\n"
                "[Data]\nAngle, PSD,\n"
                "20,15,\n21,35,\n22,95,\n23,35,\n24,15,\n",
                encoding="utf-8",
            )
            ftir.write_text(
                "Created as New Dataset, Sample 001\ncm-1,%T\n"
                "4004,90\n4003,80\n4002,20\n4001,80\n4000,90\n",
                encoding="utf-8",
            )
            # Two independent samples, second spot is a measurement within S1.
            samples = [
                {"path": str(s1), "technique": "xrd",
                 "sample_id": "S1", "measurement_id": "xrd1", "min_prominence": 40},
                {"path": str(s2), "technique": "xrd",
                 "sample_id": "S2", "measurement_id": "xrd1", "min_prominence": 40},
                {"path": str(ftir), "technique": "ftir", "signal_mode": "transmittance",
                 "sample_id": "S1", "measurement_id": "ftir_spot2", "min_prominence": 20},
            ]
            out = analyze_spectra_to_deck(
                measurements=samples, output_dir=root / "out",
                deck_title="Fe3O4 experimental spectra",
            )
            self.assertTrue(Path(out["pptx"]).is_file())
            report = json.loads(Path(out["json"]).read_text(encoding="utf-8"))
            self.assertEqual(report["comparisons"]["xrd"]["sample_ids"], ["S1", "S2"])
            self.assertEqual(report["comparisons"]["xrd"]["measurement_count"], 2)
            self.assertEqual(
                report["measurements"][2]["measurement_id"], "ftir_spot2"
            )
            self.assertIn("Fe3O4/Fe2O3 percentages", report["not_estimated"])
            prs = Presentation(out["pptx"])
            # Cover, three experimental pages, conclusion/future work
            self.assertEqual(len(prs.slides), 5)
            self.assertTrue(all(Path(p).is_file() for p in out["plots"]))

    def test_uses_original_template_without_changing_it(self):
        from pptx import Presentation
        with tempfile.TemporaryDirectory() as base:
            root = Path(base)
            template = root / "template.pptx"
            prs = Presentation()
            prs.slides.add_slide(prs.slide_layouts[0])
            prs.slides.add_slide(prs.slide_layouts[6])
            prs.save(template)
            original_size = template.stat().st_size
            spectra = root / "raman.asc"
            spectra.write_text("100\t1\n101\t5\n102\t50\n103\t5\n104\t1", encoding="utf-8")
            out = analyze_spectra_to_deck(
                [{"path": str(spectra), "technique": "raman", "sample_id": "R",
                  "min_prominence": 10}],
                root / "out", template_path=template,
            )
            generated = Presentation(out["pptx"])
            self.assertEqual(len(generated.slides), 3)
            self.assertEqual(len(Presentation(template).slides), 2)
            self.assertEqual(template.stat().st_size, original_size)


if __name__ == "__main__":
    unittest.main()
