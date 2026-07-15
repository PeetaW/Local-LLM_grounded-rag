#!/usr/bin/env python3
import os
import sys
import json
import types
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

if "requests" not in sys.modules:
    requests_stub = types.ModuleType("requests")
    requests_stub.post = None
    sys.modules["requests"] = requests_stub

import config as cfg
from rag.comparison_json_validator import (
    attach_comparison_requirements,
    build_comparison_requirements,
    exact_isotope_terms,
)
from rag.knowledge_synthesizer import (
    _build_user_prompt,
    _append_isotope_cost_fact,
    _comparison_json_validation_errors,
    KnowledgeSynthesizer,
    _normalize_comparison_json,
    _system_prompt,
)


class TestKnowledgeSynthesizerPrompt(unittest.TestCase):
    def test_generate_captures_ollama_terminal_metadata(self):
        response = MagicMock()
        response.iter_lines.return_value = [
            json.dumps({"response": "answer", "done": False}).encode(),
            json.dumps({
                "response": "",
                "done": True,
                "done_reason": "length",
                "prompt_eval_count": 123,
                "eval_count": 8192,
            }).encode(),
        ]
        metadata = {}

        with patch("rag.knowledge_synthesizer.requests.post", return_value=response):
            output = KnowledgeSynthesizer()._generate("prompt", "system", metadata)

        self.assertEqual(output, "answer")
        self.assertEqual(metadata["done_reason"], "length")
        self.assertEqual(metadata["prompt_eval_count"], 123)
        self.assertEqual(metadata["eval_count"], 8192)
        self.assertEqual(metadata["requested_num_ctx"], cfg.STAGE3_NUM_CTX)

    def test_comparison_schema_is_ab_switch(self):
        old = cfg.STAGE3_COMPARISON_SCHEMA_ENABLED
        old_json = cfg.COMPARISON_JSON_ENABLED
        try:
            cfg.STAGE3_COMPARISON_SCHEMA_ENABLED = False
            prompt = _build_user_prompt("FORMATTED_CHUNKS", "Compare synthetic routes.")
            self.assertNotIn("[source_roles]", prompt)

            cfg.STAGE3_COMPARISON_SCHEMA_ENABLED = True
            cfg.COMPARISON_JSON_ENABLED = False
            prompt = _build_user_prompt(
                "FORMATTED_CHUNKS",
                "Compare synthetic routes for scalability, cost-effectiveness, and safety.",
            )
            self.assertIn("[source_roles]", prompt)
            self.assertIn("role_hint=review/comparison source", prompt)
            self.assertIn("are not paper evidence", prompt)
            self.assertIn("[dimension_evidence]", prompt)
            self.assertIn("- safety:", prompt)
            self.assertIn("scalability、cost-effectiveness、safety", prompt)
            self.assertIn("high cost/expense", prompt)
            self.assertIn("FORMATTED_CHUNKS", prompt)
        finally:
            cfg.STAGE3_COMPARISON_SCHEMA_ENABLED = old
            cfg.COMPARISON_JSON_ENABLED = old_json

    def test_schema_only_applies_to_comparison_queries(self):
        old = cfg.STAGE3_COMPARISON_SCHEMA_ENABLED
        old_json = cfg.COMPARISON_JSON_ENABLED
        try:
            cfg.STAGE3_COMPARISON_SCHEMA_ENABLED = True
            cfg.COMPARISON_JSON_ENABLED = True
            prompt = _build_user_prompt("FORMATTED_CHUNKS", "What is the reported IC50?")
            self.assertNotIn("[source_roles]", prompt)
            self.assertNotIn("comparison_json", prompt)
        finally:
            cfg.STAGE3_COMPARISON_SCHEMA_ENABLED = old
            cfg.COMPARISON_JSON_ENABLED = old_json

    def test_comparison_json_is_ab_switch(self):
        old_schema = cfg.STAGE3_COMPARISON_SCHEMA_ENABLED
        old_json = cfg.COMPARISON_JSON_ENABLED
        try:
            cfg.STAGE3_COMPARISON_SCHEMA_ENABLED = True
            cfg.COMPARISON_JSON_ENABLED = True
            prompt = _build_user_prompt(
                "FORMATTED_CHUNKS",
                "Compare synthetic routes for isotopic enrichment and scalability.",
            )
            self.assertIn("COMPARISON_JSON MODE", prompt)
            self.assertIn('"comparison_json"', prompt)
            self.assertIn('"isotopic_enrichment"', prompt)
            self.assertIn('"cost_effectiveness"', prompt)
            self.assertIn('"safety"', prompt)
            self.assertIn('"outcome"', prompt)
            self.assertIn('"evidence": [{"source": "", "claim": ""}]', prompt)
            self.assertIn('"central_tradeoff": {"claim": "", "sources": []}', prompt)
            self.assertIn("requested=true", prompt)
            self.assertIn("evidence_found=true", prompt)
            self.assertIn("not in direct_routes", prompt)
            self.assertIn("chymotrypsin-catalysed enzymatic hydrolysis", prompt)
            self.assertIn("high cost/expense", prompt)
            self.assertNotIn("[source_roles]", prompt)
        finally:
            cfg.STAGE3_COMPARISON_SCHEMA_ENABLED = old_schema
            cfg.COMPARISON_JSON_ENABLED = old_json

    def test_english_distillation_is_ab_switch(self):
        old = cfg.STAGE3_ENGLISH_DISTILLATION_ENABLED
        try:
            cfg.STAGE3_ENGLISH_DISTILLATION_ENABLED = False
            self.assertIn("使用繁體中文輸出", _system_prompt())

            cfg.STAGE3_ENGLISH_DISTILLATION_ENABLED = True
            prompt = _system_prompt()
            self.assertIn("Output in English", prompt)
            self.assertNotIn("使用繁體中文輸出", prompt)
        finally:
            cfg.STAGE3_ENGLISH_DISTILLATION_ENABLED = old

    def test_comparison_json_normalizer_accepts_fenced_json(self):
        raw = '```json\n{"comparison_json":{"dimensions":{"safety":{"present":true}}}}\n```'
        normalized = _normalize_comparison_json(raw)
        self.assertIn('"comparison_json"', normalized)
        self.assertIn('"direct_routes": []', normalized)
        self.assertIn('"isotopic_enrichment"', normalized)
        self.assertIn('"requested": false', normalized)
        self.assertIn('"evidence_found": true', normalized)
        self.assertIn('"safety"', normalized)
        self.assertNotIn("```", normalized)

    def test_comparison_json_normalizer_patches_query_dimensions_and_review_routes(self):
        raw = """
        {
          "comparison_json": {
            "source_roles": [{"source": "ReviewA", "role": "review/comparison source"}],
            "direct_routes": [{"source": "ReviewA", "route_phrase": "example"}],
            "dimensions": {
              "scalability": {"requested": false, "evidence_found": false, "text": "", "sources": []},
              "cost_effectiveness": {"requested": false, "evidence_found": false, "text": "", "sources": []}
            },
            "central_tradeoff": "does not explicitly provide scalability or cost-effectiveness"
          }
        }
        """
        normalized = _normalize_comparison_json(
            raw,
            "Compare routes for isotopic enrichment, scalability, and cost-effectiveness.",
        )
        data = __import__("json").loads(normalized)
        comparison = data["comparison_json"]
        self.assertEqual(comparison["direct_routes"], [])
        self.assertTrue(comparison["dimensions"]["scalability"]["requested"])
        self.assertFalse(comparison["dimensions"]["scalability"]["evidence_found"])
        self.assertTrue(comparison["dimensions"]["cost_effectiveness"]["requested"])
        self.assertTrue(comparison["dimensions"]["isotopic_enrichment"]["requested"])
        self.assertIsInstance(comparison["central_tradeoff"], dict)

    def test_comparison_json_validator_flags_repairable_errors(self):
        raw = """
        {
          "comparison_json": {
            "source_roles": [{"source": "ReviewA", "role": "review/comparison source"}],
            "direct_routes": [{"source": "ReviewA", "route_phrase": "example"}],
            "review_comparison_sources": [],
            "dimensions": {
              "scalability": {"requested": false, "evidence_found": false, "text": "", "sources": []}
            },
            "central_tradeoff": "does not explicitly provide scalability"
          }
        }
        """
        errors = _comparison_json_validation_errors(raw, "Compare routes for scalability.")
        self.assertTrue(any("Review/comparison source" in err for err in errors))
        self.assertTrue(any("central_tradeoff" in err for err in errors))

    def test_validator_accepts_semantic_purity_isotope_tradeoff(self):
        raw = json.dumps({
            "comparison_json": {
                "source_roles": [{"source": "RouteA", "role": "route"}],
                "direct_routes": [{
                    "source": "RouteA",
                    "route_phrase": "hybrid route",
                    "outcome": "Optically pure product with 100% optical purity",
                    "produces_target": True,
                    "evidence": "route evidence",
                }],
                "review_comparison_sources": [],
                "dimensions": {
                    "isotopic_enrichment": {
                        "requested": True,
                        "evidence_found": True,
                        "evidence": [{"source": "RouteA", "claim": "10B material is required."}],
                    }
                },
                "central_tradeoff": {
                    "claim": "High optical purity is achievable, but 10B material is costly.",
                    "sources": ["RouteA"],
                },
            }
        })
        errors = _comparison_json_validation_errors(raw, "Compare isotopic enrichment.")
        self.assertFalse(any("high-purity/isotopically enriched" in err for err in errors))

    def test_requirement_audit_flags_missing_review_dimension_support(self):
        query = "Compare routes focusing on isotopic enrichment, scalability, and cost-effectiveness."
        requirements = build_comparison_requirements(query, [
            {
                "source": "ReviewA",
                "text": (
                    "Source metadata (not paper evidence): role_hint=review/comparison source\n"
                    "Retrieved evidence snippets:\n"
                    "Boron-10 isotopically enriched routes face scale-up limits and high cost."
                ),
            },
            {
                "source": "RouteA",
                "text": "Retrieved evidence snippets:\nA hybrid route produces the target.",
            },
        ])
        payload = {
            "comparison_json": {
                "source_roles": [{"source": "RouteA", "role": "route"}],
                "direct_routes": [{
                    "source": "RouteA",
                    "route_phrase": "hybrid route",
                    "outcome": "target product",
                }],
                "review_comparison_sources": [],
                "dimensions": {
                    "isotopic_enrichment": {
                        "requested": True,
                        "evidence_found": True,
                        "evidence": [{"source": "RouteA", "claim": "10B-enriched target"}],
                    },
                    "scalability": {
                        "requested": True,
                        "evidence_found": True,
                        "evidence": [{"source": "RouteA", "claim": "The route can be scaled."}],
                    },
                    "cost_effectiveness": {
                        "requested": True,
                        "evidence_found": False,
                        "evidence": [],
                    },
                },
                "central_tradeoff": {
                    "claim": "10B enrichment must be balanced against scale and cost.",
                    "sources": ["RouteA"],
                },
            }
        }
        attach_comparison_requirements(payload, requirements)

        errors = _comparison_json_validation_errors(json.dumps(payload), query)

        self.assertTrue(any("must retain role=review/comparison source" in error for error in errors))
        self.assertTrue(any("Review/comparison source `ReviewA`" in error for error in errors))
        self.assertTrue(any("dimensions.cost_effectiveness.evidence" in error for error in errors))

    def test_exact_isotope_parser_ignores_figure_and_nmr_labels(self):
        text = (
            "BNCT results are shown in Fig. 5C. "
            "The 13C-NMR spectrum was recorded. "
            "Producing isotopically enriched 10B material remains difficult."
        )
        self.assertEqual(exact_isotope_terms(text), ["10B"])

    def test_requirement_audit_keeps_isotope_and_cost_in_one_claim(self):
        query = "Compare routes focusing on isotopic enrichment and cost-effectiveness."
        requirements = build_comparison_requirements(query, [{
            "source": "ReviewA",
            "text": (
                "Source metadata (not paper evidence): role_hint=review/comparison source\n"
                "Retrieved evidence snippets:\n"
                "[Snippet 1] BNCT requires 10B material.\n"
                "[Snippet 2] The high cost of 10B is expected to decline."
            ),
        }])
        self.assertEqual(requirements["relation_requirements"][0]["anchors"], ["10B"])
        payload = {
            "comparison_json": {
                "source_roles": [{"source": "ReviewA", "role": "review/comparison source"}],
                "direct_routes": [],
                "review_comparison_sources": [{"source": "ReviewA", "claim": "compares routes"}],
                "dimensions": {
                    "isotopic_enrichment": {
                        "requested": True,
                        "evidence_found": True,
                        "evidence": [{"source": "ReviewA", "claim": "10B-enriched material is required."}],
                    },
                    "cost_effectiveness": {
                        "requested": True,
                        "evidence_found": True,
                        "evidence": [{
                            "source": "ReviewA",
                            "claim": "The major cost comes from the isotope starting material.",
                        }],
                    },
                },
                "central_tradeoff": {
                    "claim": "10B enrichment must be balanced against cost.",
                    "sources": ["ReviewA"],
                },
            }
        }
        attach_comparison_requirements(payload, requirements)

        errors = _comparison_json_validation_errors(json.dumps(payload), query)
        self.assertTrue(any("related facts in separate dimensions" in error for error in errors))

        payload["comparison_json"]["dimensions"]["cost_effectiveness"]["evidence"][0]["claim"] = (
            "The major cost of 10B-enriched material comes from the isotope starting material."
        )
        errors = _comparison_json_validation_errors(json.dumps(payload), query)
        self.assertFalse(any("related facts in separate dimensions" in error for error in errors))

    def test_requirement_audit_does_not_force_cross_snippet_relation(self):
        requirements = build_comparison_requirements(
            "Compare routes focusing on isotopic enrichment and cost-effectiveness.",
            [{
                "source": "ReviewA",
                "text": (
                    "Source metadata (not paper evidence): role_hint=review/comparison source\n"
                    "Retrieved evidence snippets:\n"
                    "[Snippet 1] Producing isotopically enriched 10B material is challenging.\n"
                    "[Snippet 2] The major cost comes from the isotope starting material."
                ),
            }],
        )

        self.assertEqual(requirements["relation_requirements"], [])

    def test_validator_does_not_misread_chemical_derivative_as_background(self):
        raw = """
        {
          "comparison_json": {
            "source_roles": [{"source": "bbb0683", "role": "route"}],
            "direct_routes": [{
              "source": "bbb0683",
              "route_phrase": "enantioselective alkylation followed by chymotrypsin-catalysed enzymatic hydrolysis",
              "evidence": "alkylation with protected boronic acid derivative 2 followed by hydrolysis"
            }],
            "review_comparison_sources": [],
            "dimensions": {},
            "central_tradeoff": ""
          }
        }
        """
        errors = _comparison_json_validation_errors(raw, "Compare synthetic routes.")
        self.assertFalse(any("Derivative/formulation" in err for err in errors))

        background = raw.replace('"role": "route"', '"role": "background"')
        errors = _comparison_json_validation_errors(background, "Compare synthetic routes.")
        self.assertTrue(any("Derivative/formulation" in err for err in errors))

    def test_background_dimension_evidence_is_rejected_and_filtered(self):
        raw = """
        {
          "comparison_json": {
            "source_roles": [
              {"source": "ReviewA", "role": "review/comparison source"},
              {"source": "FormulationA", "role": "background"}
            ],
            "direct_routes": [],
            "review_comparison_sources": [{"source": "ReviewA", "claim": "compares safety"}],
            "dimensions": {
              "safety": {
                "requested": false,
                "evidence_found": true,
                "evidence": [
                  {"source": "FormulationA", "claim": "The formulation is safe."},
                  {"source": "ReviewA", "claim": "The review compares route safety."}
                ]
              }
            },
            "central_tradeoff": {"claim": "Route safety differs.", "sources": ["ReviewA"]}
          }
        }
        """
        errors = _comparison_json_validation_errors(raw, "Compare synthetic routes.")
        self.assertTrue(any("must not use background" in err for err in errors))

        normalized = __import__("json").loads(_normalize_comparison_json(raw))
        safety = normalized["comparison_json"]["dimensions"]["safety"]
        self.assertEqual(
            safety["evidence"],
            [{"source": "ReviewA", "claim": "The review compares route safety."}],
        )
        self.assertTrue(safety["evidence_found"])

    def test_validator_rechecks_requested_dimensions_with_review_source(self):
        raw = """
        {
          "comparison_json": {
            "source_roles": [{"source": "ReviewA", "role": "review/comparison source"}],
            "direct_routes": [],
            "review_comparison_sources": [
              {"source": "ReviewA", "claim": "Compares route efficiency and protecting-group burden."}
            ],
            "dimensions": {
              "scalability": {
                "requested": true,
                "evidence_found": false,
                "text": "The provided text does not explicitly provide quantitative scalability data.",
                "sources": []
              },
              "cost_effectiveness": {
                "requested": true,
                "evidence_found": false,
                "text": "The provided text does not contain reagent-price information.",
                "sources": []
              }
            },
            "central_tradeoff": "The review compares route efficiency qualitatively."
          }
        }
        """
        errors = _comparison_json_validation_errors(
            raw,
            "Compare routes for scalability and cost-effectiveness.",
        )
        self.assertTrue(any("dimensions.scalability" in err for err in errors))
        self.assertTrue(any("dimensions.cost_effectiveness" in err for err in errors))

    def test_validator_rejects_flattened_multi_source_evidence(self):
        raw = """
        {
          "comparison_json": {
            "source_roles": [],
            "direct_routes": [],
            "review_comparison_sources": [],
            "dimensions": {
              "scalability": {
                "requested": true,
                "evidence_found": true,
                "text": "PaperA claim. PaperB claim.",
                "sources": ["PaperA", "PaperB"]
              }
            },
            "central_tradeoff": {"claim": "Scalability differs across routes.", "sources": ["PaperA"]}
          }
        }
        """
        errors = _comparison_json_validation_errors(raw, "Compare routes for scalability.")
        self.assertTrue(any("source-bound atomic evidence" in err for err in errors))

    def test_synthesizer_repairs_invalid_comparison_json_once(self):
        old_enabled = cfg.COMPARISON_JSON_ENABLED
        old_validation = cfg.COMPARISON_JSON_VALIDATION_ENABLED
        old_retries = cfg.COMPARISON_JSON_REPAIR_RETRIES
        try:
            cfg.COMPARISON_JSON_ENABLED = True
            cfg.COMPARISON_JSON_VALIDATION_ENABLED = True
            cfg.COMPARISON_JSON_REPAIR_RETRIES = 1
            bad = """
            {"comparison_json":{"source_roles":[{"source":"ReviewA","role":"review/comparison source"}],
            "direct_routes":[{"source":"ReviewA"}],"review_comparison_sources":[],
            "dimensions":{"scalability":{"requested":false,"evidence_found":false,"text":"","sources":[]}},
            "central_tradeoff":"does not explicitly provide scalability"}}
            """
            good = """
            {"comparison_json":{"source_roles":[{"source":"RouteA","role":"route"}],
            "direct_routes":[{"source":"RouteA","route_phrase":"route","outcome":"target product","produces_target":true,"evidence":"e"}],
            "review_comparison_sources":[],
            "dimensions":{"scalability":{"requested":true,"evidence_found":true,
            "evidence":[{"source":"RouteA","claim":"scalable route evidence"}]}},
            "central_tradeoff":{"claim":"Scalability is compared qualitatively.","sources":["RouteA"]}}}
            """
            synth = KnowledgeSynthesizer()
            synth._generate = MagicMock(side_effect=[bad, good])
            statuses = []

            result = synth.synthesize(
                [{"text": "evidence about scalability", "source": "RouteA"}],
                query="Compare routes for scalability.",
                on_status=statuses.append,
            )

            self.assertEqual(synth._generate.call_count, 2)
            self.assertIn('"evidence_found": true', result)
            self.assertTrue(any("validator failed" in status for status in statuses))
            self.assertTrue(any("validation passed" in status for status in statuses))
        finally:
            cfg.COMPARISON_JSON_ENABLED = old_enabled
            cfg.COMPARISON_JSON_VALIDATION_ENABLED = old_validation
            cfg.COMPARISON_JSON_REPAIR_RETRIES = old_retries

    def test_synthesizer_repairs_lost_exact_isotope_once(self):
        old_enabled = cfg.COMPARISON_JSON_ENABLED
        old_validation = cfg.COMPARISON_JSON_VALIDATION_ENABLED
        old_retries = cfg.COMPARISON_JSON_REPAIR_RETRIES
        try:
            cfg.COMPARISON_JSON_ENABLED = True
            cfg.COMPARISON_JSON_VALIDATION_ENABLED = True
            cfg.COMPARISON_JSON_REPAIR_RETRIES = 1
            generic = {
                "comparison_json": {
                    "source_roles": [{"source": "ReviewA", "role": "review/comparison source"}],
                    "direct_routes": [],
                    "review_comparison_sources": [{"source": "ReviewA", "claim": "compares routes"}],
                    "dimensions": {
                        "isotopic_enrichment": {
                            "requested": True,
                            "evidence_found": True,
                            "evidence": [{
                                "source": "ReviewA",
                                "claim": "Producing isotopically enriched material is challenging.",
                            }],
                        }
                    },
                    "central_tradeoff": {
                        "claim": "Isotopic enrichment raises synthesis difficulty.",
                        "sources": ["ReviewA"],
                    },
                }
            }
            repaired = json.loads(json.dumps(generic))
            repaired["comparison_json"]["dimensions"]["isotopic_enrichment"]["evidence"][0]["claim"] = (
                "Producing high-purity, 10B-enriched material is challenging."
            )
            synth = KnowledgeSynthesizer()
            synth._generate = MagicMock(side_effect=[json.dumps(generic), json.dumps(repaired)])
            statuses = []

            result = synth.synthesize(
                [{
                    "text": (
                        "Source metadata (not paper evidence): role_hint=review/comparison source\n"
                        "Retrieved evidence snippets:\n"
                        "Boron-10 is required; producing high-purity, isotopically enriched material is challenging."
                    ),
                    "source": "ReviewA",
                }],
                query="Compare routes focusing on isotopic enrichment.",
                on_status=statuses.append,
            )
            payload = json.loads(result)
            comparison = payload["comparison_json"]

            self.assertEqual(synth._generate.call_count, 2)
            self.assertEqual(payload["comparison_requirements"]["exact_isotopes"], ["10B"])
            self.assertEqual(
                payload["comparison_requirements"]["dimension_sources"],
                {"isotopic_enrichment": ["ReviewA"]},
            )
            self.assertIn("10B-enriched", comparison["dimensions"]["isotopic_enrichment"]["text"])
            self.assertTrue(any("exact isotope" in status for status in statuses))
            self.assertTrue(any("validation passed" in status for status in statuses))
        finally:
            cfg.COMPARISON_JSON_ENABLED = old_enabled
            cfg.COMPARISON_JSON_VALIDATION_ENABLED = old_validation
            cfg.COMPARISON_JSON_REPAIR_RETRIES = old_retries

    def test_synthesizer_falls_back_when_output_empty(self):
        synth = KnowledgeSynthesizer()
        synth._generate = MagicMock(return_value="")
        statuses = []

        result = synth.synthesize(
            [{"text": "retrieved evidence", "source": "PaperA"}],
            query="Compare routes for scalability.",
            on_status=statuses.append,
        )

        self.assertIn("retrieved evidence", result)
        self.assertTrue(any("empty output" in status for status in statuses))

    def test_synthesizer_emits_debug_artifacts(self):
        synth = KnowledgeSynthesizer()
        synth._generate = MagicMock(return_value="[fact1] route comparison evidence (source: PaperA)")
        artifacts = {}

        result = synth.synthesize(
            [{"text": "retrieved evidence", "source": "PaperA"}],
            query="Compare routes for scalability.",
            on_status=[].append,
            on_artifact=artifacts.__setitem__,
        )

        self.assertIn("retrieved evidence", artifacts["stage3_prompt"])
        self.assertEqual(artifacts["stage3_generation_meta"], [{"attempt": "comparison_json"}])
        self.assertIn("route comparison evidence", artifacts["stage3_raw_output"])
        self.assertEqual(artifacts["stage3_knowledge_base"], result)

    def test_synthesizer_falls_back_when_comparison_json_is_invalid(self):
        old_enabled = cfg.COMPARISON_JSON_ENABLED
        old_validation = cfg.COMPARISON_JSON_VALIDATION_ENABLED
        old_retries = cfg.COMPARISON_JSON_REPAIR_RETRIES
        try:
            cfg.COMPARISON_JSON_ENABLED = True
            cfg.COMPARISON_JSON_VALIDATION_ENABLED = True
            cfg.COMPARISON_JSON_REPAIR_RETRIES = 1
            synth = KnowledgeSynthesizer()
            synth._generate = MagicMock(side_effect=[
                '{"comparison_json":{"dimensions":{"scalability":',
                "[fact1] route comparison evidence (source: PaperA)",
            ])
            statuses = []

            result = synth.synthesize(
                [{"text": "retrieved evidence", "source": "PaperA"}],
                query="Compare routes for scalability.",
                on_status=statuses.append,
            )

            self.assertEqual(synth._generate.call_count, 2)
            self.assertIn("route comparison evidence", result)
            self.assertTrue(any("invalid JSON" in status for status in statuses))
        finally:
            cfg.COMPARISON_JSON_ENABLED = old_enabled
            cfg.COMPARISON_JSON_VALIDATION_ENABLED = old_validation
            cfg.COMPARISON_JSON_REPAIR_RETRIES = old_retries

    def test_appends_complete_isotope_cost_fact(self):
        result = "[dimension_evidence]\n- cost-effectiveness: When preparing isotopically enriched compounds, the major"
        evidence = (
            "[Chunk 1] 來源：CMDC-20-e202500059\n"
            "The review highlights the high cost of isotopically enriched 10B. "
            "【CMDC-20-e202500059】 when preparing isotopically enriched compounds, "
            "the major cost typically comes from the isotope starting material."
        )
        fixed = _append_isotope_cost_fact(
            result,
            evidence,
            "Compare routes for isotopic enrichment, scalability, and cost-effectiveness.",
        )
        self.assertIn("high cost of isotopically enriched 10B", fixed)
        self.assertIn("Source: CMDC-20-e202500059", fixed)
        self.assertNotIn("the major\n-", fixed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
