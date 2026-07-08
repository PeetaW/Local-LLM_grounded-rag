#!/usr/bin/env python3
import os
import sys
import types
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

if "requests" not in sys.modules:
    requests_stub = types.ModuleType("requests")
    requests_stub.post = None
    sys.modules["requests"] = requests_stub

import config as cfg
from rag.knowledge_synthesizer import (
    _build_user_prompt,
    _comparison_json_validation_errors,
    KnowledgeSynthesizer,
    _normalize_comparison_json,
    _system_prompt,
)


class TestKnowledgeSynthesizerPrompt(unittest.TestCase):
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
            self.assertIn("requested=true", prompt)
            self.assertIn("evidence_found=true", prompt)
            self.assertIn("not in direct_routes", prompt)
            self.assertIn("chymotrypsin-catalysed enzymatic hydrolysis", prompt)
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
            "direct_routes":[{"source":"RouteA","route_phrase":"route","produces_target":true,"evidence":"e"}],
            "review_comparison_sources":[],
            "dimensions":{"scalability":{"requested":true,"evidence_found":true,"text":"scalable route evidence","sources":["RouteA"]}},
            "central_tradeoff":"Scalability is compared qualitatively."}}
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
