#!/usr/bin/env python3
import os
import sys
import types
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    sys.modules["torch"] = types.ModuleType("torch")

from rag.citation_grounding import _find_lexical_support


class TestLexicalGrounding(unittest.TestCase):
    def test_matches_adjacent_pdf_sentences_with_line_break_hyphenation(self):
        claim = (
            "10B is separated from 11B generally by chemical exchange distillation and then "
            "converted to boronic esters or hydrolyzed to 10B boric acid."
        )
        chunks = [{
            "id": "review-1",
            "source": "review",
            "text": (
                "The 10B is separated from the more common 11B generally by chemical exchange "
                "distil- lation. This is then directly converted to various boronic esters, or "
                "hydrolyzed to 10B boric acid, a precursor to other boron compounds."
            ),
        }]

        result = _find_lexical_support(claim, chunks, ("review",))

        self.assertIsNotNone(result)
        self.assertEqual(result[0], "review-1")
        self.assertGreaterEqual(result[1], 0.8)


if __name__ == "__main__":
    unittest.main()
