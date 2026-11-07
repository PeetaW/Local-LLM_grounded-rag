"""Synthetic offline tests; no experimental reference phases are fabricated."""
import tempfile
import unittest
from pathlib import Path

from rag.research_tools.spectroscopy import load_spectrum, analyze_spectrum, compare_spectra


class SpectrumTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def write(self, content, ext=".csv"):
        path = Path(self.temp.name) / ("data" + ext)
        path.write_text(content, encoding="utf-8")
        return path

    def test_raman_maximum(self):
        path = self.write("Shift,Intensity\n100,1\n101,2\n102,9\n103,2\n104,1\n")
        s = load_spectrum(path, "Raman", "sample_1")
        result = analyze_spectrum(s, min_prominence=4)
        self.assertEqual([p.x for p in result.peaks], [102.0])
        self.assertEqual(result.sample_id, "sample_1")

    def test_ftir_transmittance_minimum(self):
        path = self.write("wavenumber,transmittance\n400,90\n401,85\n402,20\n403,85\n404,90")
        s = load_spectrum(path, "ftir", "sample_2", signal_mode="transmittance")
        result = analyze_spectrum(s, min_prominence=30)
        self.assertEqual([p.x for p in result.peaks], [402.0])

    def test_xrd_whitespace_asc(self):
        path = self.write("# XRD\n20 2\n21 10\n22 2\n23 3\n24 1\n", ".asc")
        result = analyze_spectrum(load_spectrum(path, "xrd", "one"), 4)
        self.assertEqual(result.peaks[0].x, 21.0)

    def test_independent_samples(self):
        path = self.write("1,1\n2,4\n3,1\n")
        a = analyze_spectrum(load_spectrum(path, "raman", "A"), 1)
        b = analyze_spectrum(load_spectrum(path, "raman", "B"), 1)
        self.assertEqual(compare_spectra([a,b])["sample_ids"], ["A","B"])
        self.assertEqual(compare_spectra([a,b])["measurement_count"], 2)

    def test_binary_vendor_file_rejected(self):
        path = self.write("binary", ".sif")
        with self.assertRaises(ValueError):
            load_spectrum(path, "raman", "one")

    def test_ftir_mode_mandatory(self):
        path = self.write("1,1\n2,4\n3,1\n")
        with self.assertRaises(ValueError):
            load_spectrum(path, "ftir", "one")

    def test_duplicate_axis_rejected(self):
        path = self.write("1,1\n2,2\n2,3\n")
        with self.assertRaises(ValueError):
            load_spectrum(path, "raman", "one")


    def test_vendor_raw4_text_xrd_section(self):
        # Synthetic representation of real Bruker RAW4.00 text export.
        raw = (
            ";RAW4.00\n[RawHeader]\nDate=11/02/2026\n"
            "[RangeHeader]\nStart=5\nSteps=5\n[Data]\n"
            "Angle, PSD,\n5.0, 2,\n5.1, 8,\n5.2, 120,\n5.3, 8,\n5.4, 2,\n"
        )
        path = self.write(raw, ".txt")
        spectrum = load_spectrum(path, "xrd", "Fe3O4-S1")
        result = analyze_spectrum(spectrum, min_prominence=50)
        self.assertEqual(spectrum.x, (5.0, 5.1, 5.2, 5.3, 5.4))
        self.assertEqual([round(p.x, 1) for p in result.peaks], [5.2])

    def test_broad_peak_survives_noise(self):
        from math import exp
        # Alternating noise that defeated the original neighbor-prominence detector.
        lines = []
        for i in range(250):
            x = 20 + 0.05 * i
            y = 100 + 600 * exp(-0.5 * ((x - 28) / 0.35) ** 2) + (10 if i % 2 else -10)
            lines.append(f"{x:.4f},{y:.3f}")
        spectrum = load_spectrum(self.write("\n".join(lines)), "xrd", "noisy")
        result = analyze_spectrum(spectrum, min_prominence=120)
        self.assertTrue(any(abs(p.x - 28) < 0.2 for p in result.peaks))
        self.assertLessEqual(len(result.peaks), 2)

    def test_metadata_header_and_ftir_mode(self):
        raw = (
            "Created as New Dataset,Sample 001 By PEService\n"
            "cm-1,%T\n"
            "4004,90\n4003,88\n4002,40\n4001,88\n4000,90\n"
        )
        spectrum = load_spectrum(
            self.write(raw), "ftir", "Fe3O4-S2",
            signal_mode="transmittance", measurement_id="spot2"
        )
        result = analyze_spectrum(spectrum, min_prominence=20)
        self.assertEqual(result.measurement_id, "spot2")
        self.assertEqual([p.x for p in result.peaks], [4002.0])

    def test_irregular_x_reports_warning(self):
        spectrum = load_spectrum(
            self.write("1,1\n2,5\n3,2\n4,7\n5,1\n6,3\n7,2\n25,6\n26,1\n"),
            "raman", "spot"
        )
        result = analyze_spectrum(spectrum, min_prominence=1)
        self.assertTrue(any("Irregular X spacing" in w for w in result.warnings))


if __name__ == "__main__":
    unittest.main()
