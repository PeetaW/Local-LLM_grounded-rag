"""Offline schema tests; optional python-pptx tests run when installed."""
import tempfile
import unittest
from pathlib import Path
from rag.research_tools.presentation import DeckSpec, SlideSpec, STYLE, export_research_deck, _layout_for


class PresentationTests(unittest.TestCase):
    def test_saved_style_profile(self):
        self.assertEqual(STYLE["font"], "Times New Roman")
        self.assertEqual(STYLE["presenter"], "Shuo-Hung, Wang")
        self.assertEqual(STYLE["content_title_rect"], (1.53, 0.15, 10.89, 0.74))

    def test_validation(self):
        with self.assertRaises(ValueError):
            DeckSpec("", ())
        with self.assertRaises(ValueError):
            SlideSpec(" ")


    def test_selects_title_only_content_and_title_slide_cover(self):
        try:
            from pptx import Presentation
        except ImportError:
            self.skipTest("Install optional python-pptx")
        prs = Presentation()
        self.assertEqual(_layout_for(prs, 'content').name.lower(), 'title only')
        self.assertEqual(_layout_for(prs, 'cover').name.lower(), 'title slide')

    def test_editable_deck(self):
        try:
            from pptx import Presentation
        except ImportError:
            self.skipTest("Install optional python-pptx")
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "report.pptx"
            export_research_deck(
                DeckSpec(title="Fe3O4 research progress", slides=(
                    SlideSpec("Fe3O4 XRD data", ("Two independent samples",)),
                ), conclusion=("Phase fractions remain undetermined",)),
                output
            )
            prs = Presentation(output)
            self.assertEqual(len(prs.slides), 3)
            self.assertTrue(any("Fe3O4 XRD data" in shape.text
                                for shape in prs.slides[1].shapes if shape.has_text_frame))


if __name__ == "__main__":
    unittest.main()
