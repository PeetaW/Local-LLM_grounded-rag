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

from rag.citation_grounding import _find_lexical_support, _preprocess_for_nli


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

    def test_matches_scientific_claim_across_fig_abbreviation_and_pdf_ligature(self):
        claim = (
            "The addition of preincubation significantly augmented inhibition potency to an "
            "IC50 of 34.2 ± 3.6 nM in HT-29 cells."
        )
        chunks = [{
            "id": "paper-1",
            "source": "paper",
            "text": (
                "Combination assays were performed using HT-29 cells. As shown in Fig. 5, while "
                "the IC50 was 99.2 ± 11.0 nM, the addition of preincubation signiﬁcantly "
                "augmented its inhibition potency (IC50 ¼ 34.2 ± 3.6 nM)."
            ),
        }]

        result = _find_lexical_support(claim, chunks, ("paper",))

        self.assertIsNotNone(result)
        self.assertEqual(result[0], "paper-1")

    def test_matches_five_token_verbatim_scientific_claim(self):
        result = _find_lexical_support(
            "high cost of isotopically enriched 10B.",
            [{
                "id": "review-1",
                "source": "review",
                "text": (
                    "The review highlights limitations regarding scalability and cost-effectiveness, "
                    "especially considering the high cost of isotopically enriched 10B."
                ),
            }],
            ("review",),
        )

        self.assertEqual(result, ("review-1", 1.0))

    def test_matches_positive_clause_after_unrelated_negated_clause(self):
        result = _find_lexical_support(
            "The synthesis of L-BPA has been approached through multiple routes.",
            [{
                "id": "review-1",
                "source": "review",
                "text": (
                    "There is no consensus approach to making it—the synthesis of L-BPA "
                    "has been approached through multiple routes, reflecting purity challenges."
                ),
            }],
            ("review",),
        )

        self.assertEqual(result, ("review-1", 1.0))

    def test_skips_non_substantive_source_preamble(self):
        sentence = (
            "Based on the provided data from a single source, the following information addresses "
            "the degradation products and storage stability of BPA."
        )

        self.assertEqual(_preprocess_for_nli(sentence), "")


if __name__ == "__main__":
    unittest.main()
