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

    def test_fact_contract_rejects_partial_numeric_and_dependent_coverage(self):
        facts = judge._fact_items([
            "JPH203 inhibition is concentration-dependent and time-dependent.",
            "The drug product reaches about 1% at 40 C over 6 months.",
        ])
        candidate = (
            "JPH203 preincubation significantly augmented inhibition potency. "
            "The drug product was incubated at 4, 25, and 40 C for several months."
        )
        audit, errors = judge._validate_fact_audit(
            {"facts": [
                {"id": "F1", "verdict": "covered", "evidence_ids": ["C1"], "reason": "related"},
                {"id": "F2", "verdict": "covered", "evidence_ids": ["C2"], "reason": "setup"},
            ]},
            facts,
            candidate,
            stable_protocol=True,
        )

        self.assertEqual(errors, [])
        self.assertEqual([item["verdict"] for item in audit], ["missing", "missing"])
        self.assertEqual([item["judge_verdict"] for item in audit], ["covered", "covered"])

    def test_contract_numbers_ignore_identifier_digits_and_latex_parameter_subscripts(self):
        plain = "JPH203 preincubation alone had an IC50 of 193 +/- 50 nM."
        latex = r"JPH203 preincubation alone had an $\text{IC}_{50}$ of $193 \pm 50$ nM."
        identifiers = "JPH203 binds LAT1-4F2hc, while 10B is the enriched isotope."

        self.assertEqual(judge._contract_numbers(plain), {"193", "50"})
        self.assertEqual(judge._contract_numbers(latex), {"193", "50"})
        self.assertEqual(judge._contract_numbers(identifiers), set())

    def test_fact_contract_requires_same_scope_for_contradiction(self):
        facts = judge._fact_items([
            "BPA degrades to tyrosine under alkaline and oxidative conditions, extremely rapidly."
        ])
        candidate = "Raw BPA powder remained stable during dry storage at 40 C."
        audit, errors = judge._validate_fact_audit(
            {"facts": [{
                "id": "F1",
                "verdict": "contradicted",
                "evidence_ids": ["C1"],
                "reason": "stable",
            }]},
            facts,
            candidate,
            stable_protocol=True,
        )

        self.assertEqual(errors, [])
        self.assertEqual(audit[0]["verdict"], "missing")
        self.assertEqual(audit[0]["judge_verdict"], "contradicted")

    def test_fact_contract_requires_condition_and_outcome_scope(self):
        degradation = judge._apply_fact_contract(
            "BPA degrades to tyrosine under alkaline and oxidative conditions, extremely rapidly.",
            "covered",
            ["A mechanistic pathway shows oxidative degradation of BPA to tyrosine."],
        )
        stability = judge._apply_fact_contract(
            "BPA is stable in the tested acidic and FeCl3 solutions.",
            "covered",
            ["BPA was tested in 100 mM HCl and 5% FeCl3."],
        )
        standalone = judge._apply_fact_contract(
            "Preincubation alone has an IC50 of 193 ± 50 nM.",
            "contradicted",
            ["With the addition of preincubation, the combined IC50 was 34.2 ± 3.6 nM."],
        )
        cooperative = judge._apply_fact_contract(
            "The lower combined IC50 shows that preincubation significantly enhances potency.",
            "covered",
            ["The preincubation effect synergistically enhances the co-incubation inhibitory effects."],
        )

        self.assertEqual(degradation[0], "missing")
        self.assertIn("alkaline", degradation[1])
        self.assertIn("rapid", degradation[1])
        self.assertEqual(stability[0], "missing")
        self.assertIn("stable", stability[1])
        self.assertEqual(standalone[0], "missing")
        self.assertIn("standalone", standalone[1])
        self.assertEqual(cooperative[0], "covered")

    def test_fact_contract_recovers_only_explicit_positive_relation(self):
        facts = judge._fact_items([
            "The lower combined IC50 shows that preincubation significantly enhances JPH203 inhibitory potency."
        ])
        numeric_only = "The combined IC50 was 34.2 nM versus 99.2 nM for co-incubation alone."
        candidate = numeric_only + " " + (
            "While the co-incubation IC50 was 99.2 nM, the addition of preincubation "
            "signifi- cantly augmented its inhibition potency (IC50 34.2 nM)."
        )
        missing = {"facts": [{
            "id": "F1", "verdict": "missing", "evidence_ids": [], "reason": "not explicit",
        }]}

        audit, errors = judge._validate_fact_audit(
            missing, facts, candidate, stable_protocol=True,
        )
        inferred, inferred_errors = judge._validate_fact_audit(
            missing, facts, numeric_only, stable_protocol=True,
        )

        self.assertEqual(errors, [])
        self.assertEqual(audit[0]["verdict"], "covered")
        self.assertEqual(audit[0]["judge_verdict"], "missing")
        self.assertEqual(audit[0]["evidence_ids"], ["C2"])
        self.assertEqual(inferred_errors, [])
        self.assertEqual(inferred[0]["verdict"], "missing")


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
        self.assertIn("Taiwan amino-acid names", payload["system"])
        self.assertIn("ENGLISH SOURCE SENTENCES", payload["prompt"])
        self.assertIn("not a one-to-one mapping", payload["prompt"])
        self.assertEqual(payload["format"]["required"], ["errors"])
        self.assertEqual(payload["format"]["properties"]["errors"]["type"], "array")

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

    def test_translation_fidelity_discards_identical_number_unit_false_positive(self):
        client = MagicMock()
        client.post.return_value = _response({"errors": [{
            "type": "number_unit",
            "severity": "material",
            "source_ids": ["S1"],
            "target_ids": ["T1"],
            "reason": "the values were interpreted incorrectly",
        }]})
        with (
            patch.object(judge, "requests", client),
            patch.object(judge.cfg, "TRANSLATION_EXACT_VALUE_FILTER_ENABLED", True),
        ):
            result = judge.judge_translation_fidelity(
                "The Ki values were 0.37 mM and 0.46 mM.",
                "Ki 值分別為 0.37 mM 與 0.46 mM。",
                model="test",
                base_url="http://test",
            )

        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["error_audit"], [])

        with (
            patch.object(judge, "requests", client),
            patch.object(judge.cfg, "TRANSLATION_EXACT_VALUE_FILTER_ENABLED", False),
        ):
            control = judge.judge_translation_fidelity(
                "The Ki values were 0.37 mM and 0.46 mM.",
                "Ki 值分別為 0.37 mM 與 0.46 mM。",
                model="test",
                base_url="http://test",
            )

        self.assertEqual(control["score"], 0.5)

    def test_translation_value_filter_ignores_different_citation_syntax(self):
        source = "The Ki values were 0.37 mM and 0.46 mM [1-s2.0-S1347861320300633-main]."
        target = "Ki 值分別為 0.37 mM 與 0.46 mM【1-s2.0-S1347861320300633-main】。"
        audit, errors = judge._validate_translation_audit(
            {"errors": [{
                "type": "number_unit",
                "severity": "material",
                "source_ids": ["S1"],
                "target_ids": ["T1"],
                "reason": "wrong unit",
            }]},
            source,
            target,
        )

        self.assertEqual(errors, [])
        self.assertEqual(audit, [])

    def test_translation_audit_accepts_taiwan_terms_and_merged_sentences(self):
        source = (
            "Tyrosine is a BPA impurity. "
            "Approximately 1% of phenylalanine formed at 40 C over 6 months."
        )
        target = (
            "tyrosine (酪胺酸) 是 BPA 雜質，且在 40 C 儲存 6 個月後形成約 1% 的 "
            "phenylalanine (苯丙胺酸)。"
        )
        audit, errors = judge._validate_translation_audit(
            {"errors": [
                {
                    "type": "mistranslation",
                    "severity": "material",
                    "source_ids": ["S1"],
                    "target_ids": ["T1"],
                    "reason": "Tyrosine is mistranslated as 酪胺酸 instead of 酪氨酸.",
                },
                {
                    "type": "mistranslation",
                    "severity": "material",
                    "source_ids": ["S2"],
                    "target_ids": [],
                    "reason": "Phenylalanine is mistranslated as 苯丙胺酸 instead of 苯丙氨酸.",
                },
                {
                    "type": "omission",
                    "severity": "material",
                    "source_ids": ["S2"],
                    "target_ids": [],
                    "reason": (
                        "The content appears merged into T1, but the source sentence structure "
                        "implies two distinct elements."
                    ),
                },
            ]},
            source,
            target,
        )

        self.assertEqual(errors, [])
        self.assertEqual(audit, [])

    def test_translation_audit_keeps_actual_semantic_omission(self):
        audit, errors = judge._validate_translation_audit(
            {"errors": [{
                "type": "omission",
                "severity": "material",
                "source_ids": ["S1"],
                "target_ids": ["T1"],
                "reason": "The numerical yield of 82% is absent from the target.",
            }]},
            "The isolated reaction yield was reported as 82% after purification.",
            "該反應經純化後的最終產率被描述為相當高，但沒有提供任何具體數值。",
        )

        self.assertEqual(errors, [])
        self.assertEqual(len(audit), 1)

    def test_translation_audit_discards_omission_when_named_witness_exists(self):
        source = (
            "Boc protection can be removed with TFA within 5-10 min at room temperature "
            "on a bench scale. The treatment is followed by later-stage cell membrane disruption."
        )
        target = (
            "Boc 保護可在實驗室規模 (bench scale) 下使用 TFA，於室溫 5-10 min 內去除。"
            "此處理隨後在後期導致細胞膜破裂。"
        )
        audit, errors = judge._validate_translation_audit(
            {"errors": [
                {
                    "type": "omission",
                    "severity": "material",
                    "source_ids": ["S1"],
                    "target_ids": ["T1"],
                    "reason": "The phrase 'in 5-10 min at room temperature on a bench scale' is omitted.",
                },
                {
                    "type": "omission",
                    "severity": "material",
                    "source_ids": ["S2"],
                    "target_ids": ["T1"],
                    "reason": "The detail 'later-stage cell membrane disruption' is omitted.",
                },
            ]},
            source,
            target,
        )

        self.assertEqual(errors, [])
        self.assertEqual(audit, [])

    def test_translation_audit_ignores_citation_only_omissions(self):
        source = "The major cost comes from isotope starting material [ReviewA]."
        target = "主要成本來自同位素起始原料。"
        audit, errors = judge._validate_translation_audit(
            {"errors": [{
                "type": "omission",
                "severity": "material",
                "source_ids": ["S1"],
                "target_ids": ["T1"],
                "reason": "The citation reference '[ReviewA]' is omitted from the target.",
            }]},
            source,
            target,
        )

        self.assertEqual(errors, [])
        self.assertEqual(audit, [])

    def test_translation_audit_recognizes_comparison_and_supplementary_figure(self):
        source = "LAT1 is the primary transporter compared to ATB0,+ and LAT2."
        target = "與 ATB0,+ 和 LAT2 相比，LAT1 是主要轉運蛋白。"
        audit, errors = judge._validate_translation_audit(
            {"errors": [{
                "type": "omission",
                "severity": "material",
                "source_ids": ["S1"],
                "target_ids": ["T1"],
                "reason": "The phrase 'compared to ATB0,+ and LAT2' is omitted.",
            }]},
            source,
            target,
        )

        self.assertEqual(errors, [])
        self.assertEqual(audit, [])
        self.assertTrue(judge._translation_omission_witness_present(
            "omission",
            "The phrase 'Supplementary Fig' is omitted.",
            "The result appears in Supplementary Fig.",
            "結果顯示於補充圖。",
        ))

    def test_translation_audit_uses_only_source_side_quoted_witnesses(self):
        source = "Strategy: `Ono` reports intramolecular boroxine formation [Ono]."
        target = "策略：【Ono】報導了分子內 boroxine 形成 [Ono]。"
        audit, errors = judge._validate_translation_audit(
            {"errors": [{
                "type": "omission",
                "severity": "material",
                "source_ids": ["S1"],
                "target_ids": ["T1"],
                "reason": (
                    "The source subject 'Ono' is omitted because the target starts "
                    "directly with '報導了...'."
                ),
            }]},
            source,
            target,
        )

        self.assertEqual(errors, [])
        self.assertEqual(audit, [])


if __name__ == "__main__":
    unittest.main()
