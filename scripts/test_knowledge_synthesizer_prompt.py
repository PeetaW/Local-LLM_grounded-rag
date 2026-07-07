#!/usr/bin/env python3
import os
import sys
import types
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

if "requests" not in sys.modules:
    requests_stub = types.ModuleType("requests")
    requests_stub.post = None
    sys.modules["requests"] = requests_stub

import config as cfg
from rag.knowledge_synthesizer import _build_user_prompt


class TestKnowledgeSynthesizerPrompt(unittest.TestCase):
    def test_comparison_schema_is_ab_switch(self):
        old = cfg.STAGE3_COMPARISON_SCHEMA_ENABLED
        try:
            cfg.STAGE3_COMPARISON_SCHEMA_ENABLED = False
            prompt = _build_user_prompt("FORMATTED_CHUNKS", "Compare synthetic routes.")
            self.assertNotIn("[source_roles]", prompt)

            cfg.STAGE3_COMPARISON_SCHEMA_ENABLED = True
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

    def test_schema_only_applies_to_comparison_queries(self):
        old = cfg.STAGE3_COMPARISON_SCHEMA_ENABLED
        try:
            cfg.STAGE3_COMPARISON_SCHEMA_ENABLED = True
            prompt = _build_user_prompt("FORMATTED_CHUNKS", "What is the reported IC50?")
            self.assertNotIn("[source_roles]", prompt)
        finally:
            cfg.STAGE3_COMPARISON_SCHEMA_ENABLED = old


if __name__ == "__main__":
    unittest.main(verbosity=2)
