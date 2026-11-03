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


if __name__ == "__main__":
    unittest.main()
