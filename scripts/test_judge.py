#!/usr/bin/env python3
import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))
import judge


def _response(payload):
    response = MagicMock()
    response.json.return_value = {"response": json.dumps(payload)}
    return response


def _text_response(text):
    response = MagicMock()
    response.json.return_value = {"response": text}
    return response


class TestStructuredJudge(unittest.TestCase):
    def test_candidate_items_keep_substantive_tradeoff_heading(self):
        items = judge._candidate_items(
            "Comparison scaffold:\n"
            "- Route evidence [PaperA].\n"
            "Central trade-off (purity versus scalability and cost-effectiveness):"
        )
        self.assertEqual(
            [item["text"] for item in items],
            [
                "Route evidence [PaperA].",
                "Central trade-off (purity versus scalability and cost-effectiveness):",
            ],
        )

    def test_candidate_items_split_multiple_sentences(self):
        items = judge._candidate_items(
            "JPH203 occupies the LAT1 substrate pocket. It blocks amino-acid transport."
        )
        self.assertEqual(len(items), 2)
        self.assertEqual(items[1]["text"], "It blocks amino-acid transport.")

    def test_negative_verdict_is_rechecked_with_candidate_ids(self):
        candidate = (
            "Producing high-purity, isotopically enriched 10B material is a challenge.\n"
            "The major cost comes from the isotope starting material."
        )
        first = {"facts": [
            {"id": "F1", "verdict": "missing", "evidence_ids": [], "reason": "not found"},
            {"id": "F2", "verdict": "covered", "evidence_ids": ["C2"], "reason": "stated"},
        ]}
        review = {"facts": [{
            "id": "F1",
            "verdict": "covered",
            "evidence_ids": ["C1"],
            "reason": "the first pass missed the explicit sentence",
        }]}

        client = MagicMock()
        client.post.side_effect = [_response(first), _response(review)]
        with patch.object(judge, "requests", client):
            result = judge.judge_correctness(
                "question",
                candidate,
                "reference",
                model="test",
                base_url="http://test",
                reference_facts=[
                    "High-purity isotopically enriched 10B material is difficult to produce.",
                    "The isotope starting material is the major cost driver.",
                ],
            )

        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["raw"], 5)
        self.assertEqual(result["reviewed_ids"], ["F1"])
        self.assertEqual(result["fact_audit"][0]["initial_verdict"], "missing")
        self.assertEqual(
            result["fact_audit"][1]["evidence"],
            ["The major cost comes from the isotope starting material."],
        )

    def test_non_verbatim_evidence_is_rejected(self):
        facts = judge._fact_items(["A fact"])
        _, errors = judge._validate_fact_audit({"facts": [{
            "id": "F1",
            "verdict": "covered",
            "evidence": ["invented quote"],
            "reason": "",
        }]}, facts, "actual candidate", stable_protocol=False)
        self.assertTrue(errors)

    def test_negative_review_does_not_change_negative_verdict_type(self):
        first = {"facts": [{
            "id": "F1", "verdict": "missing", "evidence_ids": [], "reason": "not stated",
        }]}
        review = {"facts": [{
            "id": "F1", "verdict": "contradicted", "evidence_ids": ["C1"], "reason": "opposite",
        }]}
        client = MagicMock()
        client.post.side_effect = [_response(first), _response(review)]
        with patch.object(judge, "requests", client):
            result = judge.judge_correctness(
                "question",
                "The candidate states a different fact.",
                "Reference fact.",
                model="test",
                base_url="http://test",
                reference_facts=["Reference fact."],
            )

        self.assertEqual(result["fact_audit"][0]["verdict"], "missing")
        self.assertEqual(result["fact_audit"][0]["review_verdict"], "contradicted")

    def test_unknown_candidate_id_is_rejected(self):
        facts = judge._fact_items(["A fact"])
        _, errors = judge._validate_fact_audit({"facts": [{
            "id": "F1",
            "verdict": "covered",
            "evidence_ids": ["C99"],
            "reason": "",
        }]}, facts, "actual candidate", stable_protocol=True)
        self.assertTrue(any("unknown evidence_ids" in error for error in errors))

    def test_invalid_structured_output_falls_back_to_holistic_score(self):
        invalid = {"facts": [{
            "id": "F1",
            "verdict": "covered",
            "evidence_ids": ["C99"],
            "reason": "bad id",
        }]}
        client = MagicMock()
        client.post.side_effect = [
            _response(invalid),
            _response(invalid),
            _text_response("SCORE: 5\nREASON: all reference facts are covered."),
        ]
        with (
            patch.object(judge, "requests", client),
            patch.object(judge.cfg, "STRUCTURED_JUDGE_STABLE_PROTOCOL_ENABLED", True),
        ):
            result = judge.judge_correctness(
                "question",
                "Actual candidate fact.",
                "Reference fact.",
                model="test",
                base_url="http://test",
                reference_facts=["Reference fact."],
            )

        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["mode"], "legacy_holistic_fallback")
        self.assertIn("unknown evidence_ids", result["structured_error"])


class TestTranslationJudge(unittest.TestCase):
    def test_translation_fidelity_uses_separate_scoring_contract(self):
        client = MagicMock()
        client.post.return_value = _response({"errors": [{
            "type": "mistranslation",
            "severity": "material",
            "source_ids": ["S1"],
            "target_ids": ["T1"],
            "reason": "chymotrypsin was mistranslated as trypsin",
        }]})
        with patch.object(judge, "requests", client):
            result = judge.judge_translation_fidelity(
                "The route uses chymotrypsin-catalysed hydrolysis.",
                "此路線使用胰蛋白酶催化水解。",
                model="test",
                base_url="http://test",
            )

        self.assertEqual(result["score"], 0.5)
        self.assertEqual(result["mode"], "translation_fidelity_v2")
        payload = client.post.call_args.kwargs["json"]
        self.assertIn("technical term left in English", payload["system"])
        self.assertIn("ENGLISH SOURCE SENTENCES", payload["prompt"])
        self.assertEqual(payload["format"], "json")

    def test_translation_fidelity_accepts_retained_english_terms(self):
        client = MagicMock()
        client.post.return_value = _response({"errors": []})
        with patch.object(judge, "requests", client):
            result = judge.judge_translation_fidelity(
                "JPH203 binds LAT1.",
                "JPH203 與 LAT1 結合。",
                model="test",
                base_url="http://test",
            )

        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["error_audit"], [])


if __name__ == "__main__":
    unittest.main()
