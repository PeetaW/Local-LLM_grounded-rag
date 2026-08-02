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
    comparison_json_payload,
    exact_isotope_terms,
)
from rag.fact_contract import (
    bind_fact_list,
    build_evidence_catalog,
    build_fact_contract_requirements,
    complete_fact_contract,
    fact_contract_prompt,
    fact_contract_schema,
    validate_fact_contract,
)
from rag.knowledge_synthesizer import (
    _build_user_prompt,
    _append_isotope_cost_fact,
    _comparison_json_validation_errors,
    KnowledgeSynthesizer,
    _normalize_comparison_json,
    _split_protocol_condition_scope,
    _system_prompt,
)


class TestKnowledgeSynthesizerPrompt(unittest.TestCase):
    def setUp(self):
        self._structured_fact_contract = cfg.STRUCTURED_FACT_CONTRACT_ENABLED
        cfg.STRUCTURED_FACT_CONTRACT_ENABLED = False

    def tearDown(self):
        cfg.STRUCTURED_FACT_CONTRACT_ENABLED = self._structured_fact_contract

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

    def test_generate_passes_native_json_schema(self):
        response = MagicMock()
        response.iter_lines.return_value = [
            json.dumps({"response": '{"evidence_ids":["E1"]}', "done": True}).encode(),
        ]
        schema = {"type": "object", "properties": {"evidence_ids": {"type": "array"}}}

        with patch("rag.knowledge_synthesizer.requests.post", return_value=response) as post:
            KnowledgeSynthesizer()._generate("prompt", "system", format_schema=schema)

        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["format"], schema)
        self.assertEqual(payload["options"]["temperature"], 0.0)

    def test_fact_contract_restores_pdf_interrupted_clause(self):
        catalog = build_evidence_catalog([{
            "source": "PaperA",
            "text": (
                "[Snippet 1] BPA/mannitol drug product slowly degraded to phenylalanine, "
                "generating 0 1 2 mAU Fig. 2. Chromatographic trace. "
                "approximately 1% of phenylalanine at 40 C over 6 months. "
                "Both products were reported as impurities, and while both have low "
                "*Corresponding author."
            ),
        }])

        stitched = [item["text"] for item in catalog if "generating approximately 1%" in item["text"]]
        self.assertEqual(len(stitched), 1)
        self.assertIn("drug product slowly degraded", stitched[0])
        self.assertFalse(any("mAU" in item["text"] for item in catalog))
        self.assertFalse(any(item["text"].startswith("approximately") for item in catalog))
        self.assertFalse(any("Corresponding author" in item["text"] for item in catalog))

        figure_prefixed = bind_fact_list(
            "[Fact 1] (See Fig. 2.) BrPD and FBBA are detected at 256 nm and elute at "
            "17.3 and 23.7 min respectively. (Source: PaperA)",
            [{
                "id": "E1",
                "source": "PaperA",
                "text": (
                    "BrPD and FBBA are detected at 256 nm and elute at "
                    "17.3 and 23.7 min respectively."
                ),
            }],
        )
        self.assertEqual(len(figure_prefixed["facts"]), 1)

        unrelated = build_evidence_catalog([{
            "source": "PaperA",
            "text": (
                "[Snippet 1] Product A degraded to phenylalanine, generating an unresolved signal. "
                "A separate storage experiment was then performed. "
                "approximately 1% of phenylalanine was measured."
            ),
        }])
        self.assertFalse(any("generating approximately 1%" in item["text"] for item in unrelated))

    def test_fact_contract_rejects_cross_sentence_condition_scope(self):
        catalog = build_evidence_catalog([{
            "source": "PaperA",
            "text": (
                "[Snippet 1] BPA in 100 mM HCl was incubated at 55 C for 24 h. "
                "[Snippet 2] A BPA solution in 6 mM H2O2 was prepared immediately before analysis."
            ),
        }])
        selected = validate_fact_contract({"evidence_ids": ["E1", "E2"]}, catalog)
        self.assertEqual(len(selected["facts"]), 2)
        self.assertTrue(all(fact["coverage"] == 1.0 for fact in selected["facts"]))

        bound = bind_fact_list(
            "\n".join([
                "[Fact 1] BPA in 100 mM HCl was incubated at 55 C for 24 h. (Source: PaperA)",
                "[Fact 2] A BPA solution in 6 mM H2O2 was prepared immediately before analysis. (Source: PaperA)",
                "[Fact 3] BPA in 6 mM H2O2 was incubated at 55 C for 24 h. (Source: PaperA)",
            ]),
            catalog,
        )
        self.assertEqual(len(bound["facts"]), 2)
        self.assertEqual(len(bound["rejected"]), 1)

    def test_fact_contract_keeps_vs_comparison_values_together(self):
        catalog = build_evidence_catalog([{
            "source": "PaperA",
            "text": (
                "[Snippet 1] Combined treatment lowered the IC50 "
                "(34.2 ± 3.6 nM vs. 99.2 ± 11.0 nM)."
            ),
        }])

        self.assertEqual(len(catalog), 1)
        self.assertIn("vs. 99.2 ± 11.0 nM", catalog[0]["text"])
        bound = bind_fact_list(
            "[Fact 1] Combined treatment lowered the IC50 "
            "(34.2 ± 3.6 nM vs. 99.2 ± 11.0 nM). (Source: PaperA)",
            catalog,
        )
        self.assertEqual(len(bound["facts"]), 1)

    def test_fact_contract_keeps_inline_figure_references(self):
        catalog = build_evidence_catalog([{
            "source": "PaperA",
            "text": (
                "[Snippet 1] JPH203 binds within the substrate-binding pocket (Fig. 1b). "
                "The hydrophobic tail occupies a defined pocket (Fig. 1c). "
                "JPH203 resembles previously determined LAT1 inhibitors (Fig. 1b, c)13."
            ),
        }])
        bound = bind_fact_list(
            "\n".join([
                "[Fact 1] JPH203 binds within the substrate-binding pocket (Fig. 1b). (Source: PaperA)",
                "[Fact 2] The hydrophobic tail occupies a defined pocket (Fig. 1c). (Source: PaperA)",
                "[Fact 3] JPH203 resembles previously determined LAT1 inhibitors (Fig. 1b, c)13 (Source: PaperA)",
            ]),
            catalog,
        )

        self.assertEqual(len(bound["facts"]), 3)
        self.assertEqual(bound["rejected"], [])

    def test_fact_contract_completes_uncovered_planner_facet(self):
        catalog = build_evidence_catalog([{
            "source": "PaperA",
            "text": (
                "[Snippet 1] The boroxine forms a water-stable hydrogel. "
                "The boroxine selectively binds fluoride with stronger affinity than phenylboronic acid."
            ),
        }])
        contract = validate_fact_contract({"evidence_ids": ["E1"]}, catalog)
        completed = complete_fact_contract(
            contract,
            catalog,
            ["What role does fluoride binding play in hydrogel formation?"],
        )

        self.assertEqual(completed["supplemented_evidence_ids"], ["E2"])
        self.assertIn("selectively binds fluoride", completed["facts"][1]["claim"])

        prompt = fact_contract_prompt(
            catalog,
            "What is the reported structure?",
            focus_questions=["What role does fluoride binding play?"],
        )
        self.assertIn("Planned coverage facets", prompt)
        self.assertIn("fluoride binding", prompt)

    def test_fact_contract_completes_each_method_requirement(self):
        catalog = build_evidence_catalog([{
            "source": "PaperA",
            "text": (
                "[Snippet 1] The route produced L-BPA in 78% yield. "
                "Benzaldehyde and glycine methyl ester were reacted in THF at -78 C, "
                "affording intermediate 4 in 74% e.e. "
                "The intermediate was converted to its methyl ester with HCl and then "
                "hydrolyzed with chymotrypsin, yielding optically pure L-BPA."
            ),
        }])
        requirements = build_fact_contract_requirements(
            "Describe the key steps in the synthesis method.",
            ["Report exact reactants, conditions, optimized yield, and product outcome."],
        )
        contract = validate_fact_contract({"evidence_ids": ["E1"]}, catalog)
        completed = complete_fact_contract(contract, catalog, requirements)

        claims = " ".join(item["claim"] for item in completed["facts"])
        self.assertIn("Benzaldehyde and glycine methyl ester", claims)
        self.assertIn("hydrolyzed with chymotrypsin", claims)
        self.assertTrue(all(
            item["covered"]
            for item in completed["requirement_coverage"]
            if item["available_count"]
        ))

        schema = fact_contract_schema(catalog, requirements)
        self.assertIn("requirement_evidence", schema["properties"])
        self.assertEqual(
            schema["properties"]["requirement_evidence"]["required"],
            [item["id"] for item in requirements],
        )

    def test_retrieval_only_control_facet_does_not_expand_answer_contract(self):
        requirements = build_fact_contract_requirements(
            "Describe the key steps in the synthesis method.",
            ["Report exact conditions, outcomes, and control or comparison outcomes."],
        )

        kinds = {item["kind"] for item in requirements}
        self.assertNotIn("control", kinds)
        key_step_condition = next(
            item for item in requirements if item["kind"] == "method_conditions"
        )
        self.assertEqual(key_step_condition["minimum"], 1)
        explicit_condition = next(
            item
            for item in build_fact_contract_requirements(
                "Describe the synthesis method and its reaction conditions."
            )
            if item["kind"] == "method_conditions"
        )
        self.assertEqual(explicit_condition["minimum"], 2)

    def test_grouped_fact_contract_ignores_ungrouped_extra_ids(self):
        catalog = build_evidence_catalog([{
            "source": "PaperA",
            "text": (
                "[Snippet 1] Compound A was reacted with reagent B to yield product C. "
                "A general background sentence does not describe the requested method."
            ),
        }])
        requirements = build_fact_contract_requirements("Describe the synthesis method.")
        contract = validate_fact_contract({
            "evidence_ids": ["E1", "E2"],
            "requirement_evidence": {
                requirement["id"]: ["E1"] for requirement in requirements
            },
        }, catalog, requirements)

        self.assertEqual(
            [fact["evidence_id"] for fact in contract["facts"]],
            ["E1"],
        )

    def test_grouped_fact_contract_caps_each_requirement_to_its_minimum(self):
        catalog = build_evidence_catalog([{
            "source": "PaperA",
            "text": (
                "[Snippet 1] Compound A was reacted with reagent B to yield product C. "
                "Compound D was reacted with reagent E to yield product F."
            ),
        }])
        requirements = [{
            "id": "R1",
            "kind": "method_transform",
            "label": "Requested transformation",
            "minimum": 1,
        }]
        contract = validate_fact_contract({
            "evidence_ids": ["E1", "E2"],
            "requirement_evidence": {"R1": ["E1", "E2"]},
        }, catalog, requirements)

        self.assertEqual(
            [fact["evidence_id"] for fact in contract["facts"]],
            ["E1"],
        )
        schema = fact_contract_schema(catalog, requirements)
        self.assertEqual(
            schema["properties"]["requirement_evidence"]["properties"]["R1"]["maxItems"],
            1,
        )

    def test_grouped_fact_contract_rejects_requirement_mismatch(self):
        catalog = build_evidence_catalog([{
            "source": "PaperA",
            "text": (
                "[Snippet 1] The boroxine cross-links form the hydrogel. "
                "Excess free trimer de-crosslinks the network and collapses the hydrogel."
            ),
        }])
        requirements = [{
            "id": "R1",
            "kind": "network_disruption",
            "label": "Requested network disruption",
            "minimum": 1,
        }]
        contract = validate_fact_contract({
            "evidence_ids": [],
            "requirement_evidence": {"R1": ["E1"]},
        }, catalog, requirements)
        completed = complete_fact_contract(contract, catalog, requirements)

        self.assertEqual(contract["facts"], [])
        self.assertEqual(
            contract["rejected"][0]["reason"],
            "evidence does not satisfy requirement",
        )
        self.assertIn("collapses the hydrogel", completed["facts"][0]["claim"])

    def test_fact_contract_relation_requirements_keep_both_mechanism_witnesses(self):
        catalog = build_evidence_catalog([{
            "source": "PaperA",
            "text": (
                "[Snippet 1] The boroxine forms a water-stable hydrogel. "
                "Fluoride exchange breaks the dynamic boroxine cross-links and collapses the gel. "
                "Adding THF reconstructs the boroxine network and reforms the hydrogel."
            ),
        }])
        requirements = build_fact_contract_requirements(
            "How does fluoride binding control hydrogel collapse and reformation?"
        )
        contract = validate_fact_contract({"evidence_ids": ["E1"]}, catalog)
        completed = complete_fact_contract(contract, catalog, requirements)
        claims = " ".join(item["claim"] for item in completed["facts"])

        self.assertIn("collapses the gel", claims)
        self.assertIn("reforms the hydrogel", claims)

    def test_fact_contract_keeps_unit_clauses_and_rejects_catalog_fragments(self):
        catalog = build_evidence_catalog([{
            "source": "PaperA",
            "text": (
                "[Snippet 1] The reaction furnished 2.83 g. 10.0 mmol of product after "
                "30 min. and was cooled to room temperature. "
                "Supporting details are shown in Supplementary Fig. "
                "Department of Chemistry, Example University. "
                "7 ) We now describe an efficient synthesis illustrated in Scheme. "
                "The reaction was )( OH THF >C::B ~~~~ 0.1 N HCl. "
                "1.2 mmol) and enzyme were adjusted to pH 5.0. "
                "The rotation was [(X] 8.6 [lit.,41]. "
                "TM3 Received: 2 February 2024 Accepted: 31 May 2024 "
                "www.example.org remains fully folded."
            ),
        }])
        texts = [item["text"] for item in catalog]

        self.assertTrue(any("2.83 g. 10.0 mmol" in text and "30 min. and" in text for text in texts))
        self.assertFalse(any("Supplementary Fig" in text for text in texts))
        self.assertFalse(any("Department of Chemistry" in text for text in texts))
        self.assertFalse(any("illustrated in Scheme" in text for text in texts))
        self.assertFalse(any("~~~~" in text for text in texts))
        self.assertFalse(any("1.2 mmol)" in text for text in texts))
        self.assertFalse(any("[lit." in text for text in texts))
        self.assertFalse(any("Received:" in text for text in texts))

    def test_fact_contract_preserves_figs_stability_and_salvages_scheme_prefix(self):
        catalog = build_evidence_catalog([{
            "source": "PaperA",
            "text": (
                "[Snippet 1] The trimer remained stable during the 7-day study. "
                "The hydrogel remained stable at pH=2 and pH=10. "
                "It was stable at 2.4 < pH < 9, while the methyl analogue was stable "
                "at 2.4 < pH < 10 (Supplementary Figs. 13h and 19c, d). "
                "Enantioselective alkylation of lithiated 3 with bromide 2 in THF "
                "at -78 C was )( OH THF >C::B ~~~~ 0.1 N HCl. "
                "Hydrolysis of 6 with sodium hydroxide was stirred at room temperature for 36 h. "
                "Recrystallization gave L-BPA (120 mg, 8()l~·~) as a crystal."
            ),
        }])
        texts = [item["text"] for item in catalog]

        self.assertTrue(any("2.4 < pH < 10" in text and "Figs. 13h" in text for text in texts))
        self.assertTrue(any(
            text == "Enantioselective alkylation of lithiated 3 with bromide 2 in THF at -78 C."
            for text in texts
        ))
        self.assertFalse(any("8()l~·~" in text for text in texts))

        stability = next(
            item
            for item in build_fact_contract_requirements(
                "What water-stable structure was reported and how does it form a hydrogel?"
            )
            if item["kind"] == "stability_values"
        )
        selected = [
            item["id"]
            for item in catalog
            if "7-day study" in item["text"] or "pH=2 and pH=10" in item["text"]
        ]
        contract = validate_fact_contract({"evidence_ids": selected}, catalog)
        completed = complete_fact_contract(contract, catalog, [stability])
        claims = [item["claim"] for item in completed["facts"]]

        self.assertEqual(stability["minimum"], 2)
        self.assertTrue(any("2.4 < pH < 10" in claim for claim in claims))
        self.assertTrue(completed["requirement_coverage"][0]["covered"])

        condition = next(
            item
            for item in build_fact_contract_requirements(
                "What hybrid process is used for the synthesis, and what are its key steps?"
            )
            if item["kind"] == "method_conditions"
        )
        condition_contract = complete_fact_contract(
            validate_fact_contract({"evidence_ids": []}, catalog),
            catalog,
            [condition],
        )
        self.assertEqual(
            condition_contract["facts"][0]["claim"],
            "Enantioselective alkylation of lithiated 3 with bromide 2 in THF at -78 C",
        )

    def test_grouped_stability_contract_keeps_valid_lower_ranked_witness(self):
        catalog = build_evidence_catalog([{
            "source": "PaperA",
            "text": (
                "[Snippet 1] The trimer remained stable during the 7-day study. "
                "The hydrogel remained stable at pH=2 and pH=10. "
                "The boroxine was stable at 2.4 < pH < 9."
            ),
        }])
        requirement = {
            "id": "R1",
            "kind": "stability_values",
            "label": "Requested stability evidence",
            "minimum": 2,
        }
        hydrogel_id = next(
            item["id"] for item in catalog if "hydrogel remained stable" in item["text"]
        )
        contract = validate_fact_contract({
            "evidence_ids": [],
            "requirement_evidence": {"R1": [hydrogel_id]},
        }, catalog, [requirement])
        completed = complete_fact_contract(contract, catalog, [requirement])
        claims = " ".join(item["claim"] for item in completed["facts"])

        self.assertEqual(
            [item["evidence_id"] for item in contract["facts"]],
            [hydrogel_id],
        )
        self.assertIn("pH=2 and pH=10", claims)
        self.assertIn("7-day study", claims)
        self.assertIn("2.4 < pH < 9", claims)
        self.assertTrue(completed["requirement_coverage"][0]["covered"])

    def test_fact_contract_prefers_complete_fact_over_fragments(self):
        catalog = build_evidence_catalog([{
            "source": "PaperA",
            "text": (
                "[Snippet 1] BPA/mannitol drug product slowly degraded to phenylalanine, "
                "generating 0 1 2 mAU Fig. 2. Chromatographic trace. "
                "approximately 1% of phenylalanine at 40 C over 6 months."
            ),
        }])
        bound = bind_fact_list(
            "\n".join([
                "[Fact 1] BPA/mannitol drug product slowly degraded to phenylalanine. (Source: PaperA)",
                "[Fact 2] approximately 1% of phenylalanine at 40 C over 6 months. (Source: PaperA)",
                "[Fact 3] BPA/mannitol drug product slowly degraded to phenylalanine, generating approximately 1% of phenylalanine at 40 C over 6 months. (Source: PaperA)",
                "[Fact 4] Mechanistic pathway. approximately 1% of phenylalanine at 40 C over 6 months. (Source: PaperA)",
            ]),
            catalog,
        )

        self.assertEqual(len(bound["facts"]), 1)
        self.assertIn("drug product slowly degraded", bound["facts"][0]["claim"])
        self.assertEqual(len(bound["rejected"]), 1)

    def test_fact_contract_prefers_precursor_formation_for_structure_identity(self):
        catalog = build_evidence_catalog([{
            "source": "PaperA",
            "text": (
                "[Snippet 1] Exposure to water transforms the dimer into a stable trimer complex. "
                "[Snippet 2] Under ambient conditions, HO-PBA spontaneously dehydrates into a "
                "dimer with aggregation-induced enhanced emission."
            ),
        }])
        requirement = {
            "id": "R1",
            "kind": "structure_identity",
            "label": "How is the water-stable boroxine formed?",
            "minimum": 1,
        }
        contract = complete_fact_contract(
            validate_fact_contract({"evidence_ids": []}, catalog),
            catalog,
            [requirement],
        )

        self.assertIn("spontaneously dehydrates", contract["facts"][0]["claim"])

    def test_water_stable_contract_requires_precursor_and_water_conversion(self):
        query = (
            "What is the water-stable boroxine structure reported, and what role do "
            "the dynamic covalent bonds play in its fluoride binding and hydrogel formation?"
        )
        requirements = build_fact_contract_requirements(query)
        catalog = build_evidence_catalog([{
            "source": "PaperA",
            "text": (
                "[Snippet 1] Upon exposure to water, the dimer transforms into a boroxine. "
                "The dimer converts in water into a stable trimer complex. "
                "Under ambient conditions, HO-PBA spontaneously dehydrates into a dimer "
                "with dynamic covalent bonds and aggregation-induced enhanced emission."
            ),
        }])
        water_ids = [
            item["id"]
            for item in catalog
            if "water" in item["text"].lower()
        ]
        contract = complete_fact_contract(
            validate_fact_contract({"evidence_ids": water_ids}, catalog),
            catalog,
            requirements,
        )
        coverage = {item["kind"]: item for item in contract["requirement_coverage"]}

        self.assertEqual(len(requirements), 8)
        self.assertNotIn("relation", coverage)
        self.assertTrue(coverage["precursor_formation"]["covered"])
        self.assertTrue(coverage["water_conversion"]["covered"])
        self.assertTrue(any(
            "spontaneously dehydrates" in fact["claim"]
            for fact in contract["facts"]
        ))

    def test_synthesizer_structured_contract_is_ab_switch(self):
        cfg.STRUCTURED_FACT_CONTRACT_ENABLED = True
        synth = KnowledgeSynthesizer()
        synth._generate = MagicMock(return_value=json.dumps({
            "evidence_ids": ["E1", "E2"],
            "requirement_evidence": {"R1": ["E1", "E2"]},
        }))
        artifacts = {}

        result = synth.synthesize(
            [{
                "source": "PaperA",
                "text": (
                    "[Snippet 1] Preincubation alone gave an IC50 of 193 nM. "
                    "[Snippet 2] Combined treatment lowered the IC50 to 34.2 nM."
                ),
            }],
            query="How did preincubation change inhibitory potency?",
            on_status=[].append,
            on_artifact=artifacts.__setitem__,
        )

        self.assertIn("[Fact 1] Preincubation alone gave an IC50 of 193 nM", result)
        self.assertIn("[Fact 2] Combined treatment lowered the IC50 to 34.2 nM", result)
        self.assertIn("evidence_ids", synth._generate.call_args.kwargs["format_schema"]["properties"])
        self.assertIn("requirement_evidence", synth._generate.call_args.kwargs["format_schema"]["properties"])
        self.assertIn('"schema": "fact_contract_v1"', artifacts["stage3_fact_contract"])

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

    def test_partial_recovery_hint_is_added_to_fact_prompt(self):
        prompt = _build_user_prompt(
            "FORMATTED_CHUNKS",
            "What storage outcome was reported?",
            recovery_hint="missing the temperature-dependent degradation result",
        )
        self.assertIn("RECOVERY PASS", prompt)
        self.assertIn("missing the temperature-dependent degradation result", prompt)
        self.assertIn("Preserve experimental setup or storage conditions", prompt)

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

    def test_mechanism_comparison_schema_requires_source_bound_witnesses(self):
        prompt = _build_user_prompt(
            "FORMATTED_CHUNKS",
            "Compare LAT1-targeting strategies and explain how their mechanisms differ.",
        )
        self.assertIn('"supporting_mechanisms"', prompt)
        self.assertIn("fill supporting_mechanisms", prompt)

        payload = {
            "comparison_json": {
                "source_roles": [{"source": "StructureA", "role": "mechanism"}],
                "direct_routes": [],
                "review_comparison_sources": [],
                "dimensions": {},
                "central_tradeoff": "",
            }
        }
        errors = _comparison_json_validation_errors(
            json.dumps(payload),
            "How do the mechanisms differ?",
        )
        self.assertTrue(any("supporting_mechanisms" in error for error in errors))

        payload["comparison_json"]["supporting_mechanisms"] = [{
            "source": "StructureA",
            "claim": "JPH203 occupies the LAT1 substrate-binding pocket.",
            "evidence": "JPH203 binds within the traditional substrate-binding pocket.",
        }]
        errors = _comparison_json_validation_errors(
            json.dumps(payload),
            "How do the mechanisms differ?",
        )
        self.assertFalse(any("supporting_mechanisms" in error for error in errors))

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

    def test_protocol_condition_scope_guard_splits_separate_sample(self):
        old = cfg.FACT_RELATION_ATOMICITY_GUARD_ENABLED
        raw = (
            "[Fact 11] Forced degradation tests for BPA were performed using 100 mM NaOH, "
            "100 mM HCl or 5% FeCl3 incubated at 55 °C for 24 h, and a solution in 6 mM "
            "H2O2 (Source: PaperA)"
        )
        try:
            cfg.FACT_RELATION_ATOMICITY_GUARD_ENABLED = True
            fixed = _split_protocol_condition_scope(raw)
            self.assertIn("[Fact 1] Forced degradation tests for BPA were performed", fixed)
            self.assertIn("[Fact 2] Forced degradation tests for BPA also used", fixed)
            self.assertNotIn("24 h, and a solution", fixed)

            cfg.FACT_RELATION_ATOMICITY_GUARD_ENABLED = False
            self.assertEqual(_split_protocol_condition_scope(raw), raw)
        finally:
            cfg.FACT_RELATION_ATOMICITY_GUARD_ENABLED = old

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

    def test_comparison_json_parser_accepts_pdf_control_character(self):
        raw = '{"comparison_json":{"evidence":"20.3 \x01 0.8"}}'
        parsed = comparison_json_payload(raw)
        self.assertEqual(parsed["comparison_json"]["evidence"], "20.3 \x01 0.8")

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

    def test_comparison_normalizer_rejects_model_requested_drift_and_repairs_route_role(self):
        raw = json.dumps({
            "comparison_json": {
                "source_roles": [{
                    "source": "TherapyA",
                    "role": "mechanism",
                    "claim": "inhibits a transporter",
                    "evidence": "direct evidence",
                }],
                "direct_routes": [{
                    "source": "TherapyA",
                    "route_phrase": "transporter inhibition",
                    "outcome": "reduced uptake",
                }],
                "supporting_mechanisms": [{
                    "source": "TherapyA",
                    "claim": "The therapy blocks transporter-mediated uptake.",
                    "evidence": "The therapy reduced transporter-mediated uptake.",
                }],
                "dimensions": {
                    "isotopic_enrichment": {
                        "requested": True,
                        "evidence_found": True,
                        "evidence": [{"source": "TherapyA", "claim": "10B was measured."}],
                    },
                },
                "central_tradeoff": {"claim": "The mechanisms differ.", "sources": ["TherapyA"]},
            },
        })
        normalized = json.loads(_normalize_comparison_json(
            raw,
            "How do therapeutic strategies targeting LAT1 differ in mechanism?",
        ))
        comparison = normalized["comparison_json"]

        self.assertFalse(comparison["dimensions"]["isotopic_enrichment"]["requested"])
        self.assertEqual(comparison["source_roles"][0]["role"], "route")
        self.assertFalse(_comparison_json_validation_errors(
            json.dumps(normalized),
            "How do therapeutic strategies targeting LAT1 differ in mechanism?",
        ))

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

    def test_mechanism_requirement_restores_named_source_interaction(self):
        query = "How do therapeutic strategies targeting LAT1 differ in mechanism?"
        chunks = [{
            "source": "StructureA",
            "text": (
                "Retrieved evidence snippets:\n"
                "[Snippet 1] JPH203 binds within the traditional substrate-binding pocket. "
                "The chloride atom of JPH203 forms a halogen bond with Tyr259."
            ),
        }]
        requirements = build_comparison_requirements(query, chunks)
        payload = {"comparison_json": {
            "source_roles": [{
                "source": "StructureA",
                "role": "mechanism",
                "claim": "JPH203 binds the traditional substrate-binding pocket.",
                "evidence": (
                    "The α-amino group and α-carboxyl group of the head... "
                    "the chloride atom of JPH203 forms a halogen bond with Tyr259."
                ),
            }],
            "direct_routes": [],
            "review_comparison_sources": [],
            "supporting_mechanisms": [{
                "source": "StructureA",
                "claim": "JPH203 causes a conformational shift in TM1.",
                "evidence": "JPH203 causes a conformational shift in TM1.",
            }],
            "dimensions": {},
            "central_tradeoff": {"claim": "Mechanisms differ.", "sources": ["StructureA"]},
        }, "comparison_requirements": requirements}
        raw = json.dumps(payload)

        self.assertTrue(any(
            "preserve the source relation" in error
            for error in _comparison_json_validation_errors(raw, query)
        ))

        normalized = json.loads(_normalize_comparison_json(
            raw,
            query,
            requirements=requirements,
        ))
        mechanisms = normalized["comparison_json"]["supporting_mechanisms"]

        self.assertEqual(
            requirements["mechanism_requirements"][0]["anchors"],
            ["halogen bond", "Tyr259"],
        )
        self.assertIn("halogen bond with Tyr259", mechanisms[0]["claim"])
        self.assertFalse(_comparison_json_validation_errors(
            json.dumps(normalized),
            query,
        ))

    def test_strategy_requirements_keep_qualifiers_and_reclassify_uptake_only_source(self):
        query = "How do therapeutic strategies targeting LAT1 differ in mechanism?"
        requirements = build_comparison_requirements(query, [{
            "source": "InhibitorA",
            "text": (
                "Retrieved evidence snippets:\n"
                "JPH203 competitively inhibits LAT1-mediated amino-acid transport. "
                "At minimally toxic concentrations, JPH203 sensitized cancer cells to radiation."
            ),
        }, {
            "source": "PeptideA",
            "text": (
                "Retrieved evidence snippets:\n"
                "For example, prior self-assembling peptides conjugated with a "
                "CAIX-targeting motif inhibited cancer growth through multivalent interactions. "
                "We designed a self-assembling peptide conjugated to the L-phenylalanine "
                "targeting motif as a LAT1 ligand. The peptide suppressed LAT1-mediated "
                "transport and thereby inhibited cancer-cell proliferation."
            ),
        }, {
            "source": "StructureA",
            "text": (
                "Retrieved evidence snippets:\n"
                "In summary, the LAT1 structure provides a structural basis for rational "
                "inhibitor design."
            ),
        }, {
            "source": "DeliveryA",
            "text": "Retrieved evidence snippets:\nBPA uptake into tumor cells is mediated by LAT1.",
        }])
        claims = " ".join(item["claim"] for item in requirements["strategy_requirements"])
        for term in (
            "competitively",
            "minimally toxic",
            "L-phenylalanine",
            "proliferation",
            "structural basis",
        ):
            self.assertIn(term, claims)
        self.assertNotIn("CAIX", claims)

        payload = {"comparison_json": {
            "source_roles": [
                {"source": "InhibitorA", "role": "route"},
                {"source": "PeptideA", "role": "route"},
                {"source": "StructureA", "role": "mechanism"},
                {"source": "DeliveryA", "role": "route"},
            ],
            "direct_routes": [{
                "source": "InhibitorA",
                "route_phrase": "competitive JPH203 inhibition of LAT1",
                "outcome": "blocked LAT1-mediated transport",
            }, {
                "source": "PeptideA",
                "route_phrase": "self-assembling peptide suppression of LAT1",
                "outcome": "inhibited cancer-cell proliferation",
            }, {
                "source": "DeliveryA",
                "route_phrase": "LAT1-mediated BPA uptake",
                "outcome": "delivered BPA into tumor cells",
            }],
            "supporting_mechanisms": [{
                "source": "StructureA",
                "claim": "The LAT1 structure provides a structural basis for inhibitor design.",
                "evidence": "The LAT1 structure provides a structural basis for inhibitor design.",
            }],
            "review_comparison_sources": [],
            "dimensions": {},
            "central_tradeoff": {
                "claim": "The strategies act on LAT1 through distinct mechanisms.",
                "sources": ["InhibitorA", "PeptideA", "StructureA"],
            },
        }, "comparison_requirements": requirements}
        normalized = json.loads(_normalize_comparison_json(
            json.dumps(payload),
            query,
            requirements=requirements,
        ))
        comparison = normalized["comparison_json"]

        self.assertEqual(
            next(item["role"] for item in comparison["source_roles"] if item["source"] == "DeliveryA"),
            "background",
        )
        self.assertNotIn("DeliveryA", {item["source"] for item in comparison["direct_routes"]})
        self.assertFalse(_comparison_json_validation_errors(json.dumps(normalized), query))

    def test_mechanism_requirement_precedes_dense_anchor_paraphrase(self):
        query = "How do therapeutic strategies targeting LAT1 differ in mechanism?"
        chunks = [{
            "source": "StructureA",
            "text": (
                "Retrieved evidence snippets:\n"
                "[Snippet 1] JPH203 binds within the traditional substrate-binding pocket. "
                "The chloride atom of JPH203 forms a halogen bond with Tyr259."
            ),
        }]
        requirements = build_comparison_requirements(query, chunks)
        dense_claim = (
            "JPH203 uses a halogen bond with Tyr259 and several hydrophobic contacts."
        )
        payload = {"comparison_json": {
            "source_roles": [{
                "source": "StructureA",
                "role": "mechanism",
                "claim": "JPH203 binds the traditional substrate-binding pocket.",
                "evidence": "JPH203 binds within the traditional substrate-binding pocket.",
            }],
            "direct_routes": [],
            "review_comparison_sources": [],
            "supporting_mechanisms": [{
                "source": "StructureA",
                "claim": dense_claim,
                "evidence": dense_claim,
            }],
            "dimensions": {},
            "central_tradeoff": {"claim": "Mechanisms differ.", "sources": ["StructureA"]},
        }, "comparison_requirements": requirements}

        normalized = json.loads(_normalize_comparison_json(
            json.dumps(payload),
            query,
            requirements=requirements,
        ))
        mechanisms = normalized["comparison_json"]["supporting_mechanisms"]
        required_claim = requirements["mechanism_requirements"][0]["claim"].rstrip(".")

        self.assertEqual(mechanisms[0]["claim"], required_claim)
        self.assertEqual(mechanisms[1]["claim"], dense_claim)

    def test_validator_accepts_source_close_generic_isotope_claim(self):
        query = "Compare routes focusing on isotopic enrichment and cost-effectiveness."
        requirements = build_comparison_requirements(query, [{
            "source": "ReviewA",
            "text": (
                "Source metadata (not paper evidence): role_hint=review/comparison source\n"
                "Retrieved evidence snippets:\n"
                "[Snippet 1] Producing high-purity, isotopically enriched material is difficult.\n"
                "[Snippet 2] BNCT requires delivery of 10B to malignant cells.\n"
                "[Snippet 3] The major cost comes from the isotope starting material."
            ),
        }])
        payload = {
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
                            "claim": "Producing high-purity, isotopically enriched material is difficult.",
                        }],
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
                    "claim": "High-purity isotopic enrichment must be balanced against cost.",
                    "sources": ["ReviewA"],
                },
            }
        }
        attach_comparison_requirements(payload, requirements)

        errors = _comparison_json_validation_errors(json.dumps(payload), query)

        self.assertFalse(any("exact isotope identifier" in error for error in errors))

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

    def test_comparison_normalizer_drops_non_review_source_from_review_section(self):
        raw = json.dumps({"comparison_json": {
            "source_roles": [{"source": "StructureA", "role": "mechanism"}],
            "direct_routes": [],
            "review_comparison_sources": [{
                "source": "StructureA",
                "claim": "A direct structural result was reported.",
            }],
            "dimensions": {},
            "central_tradeoff": {"claim": "", "sources": []},
        }})

        normalized = json.loads(_normalize_comparison_json(raw))

        self.assertEqual(
            normalized["comparison_json"]["review_comparison_sources"],
            [],
        )

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

    def test_synthesizer_does_not_inject_exact_isotope_into_generic_claim(self):
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
            synth = KnowledgeSynthesizer()
            synth._generate = MagicMock(return_value=json.dumps(generic))
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

            self.assertEqual(synth._generate.call_count, 1)
            self.assertEqual(payload["comparison_requirements"]["exact_isotopes"], ["10B"])
            self.assertEqual(
                payload["comparison_requirements"]["dimension_sources"],
                {"isotopic_enrichment": ["ReviewA"]},
            )
            self.assertEqual(
                comparison["dimensions"]["isotopic_enrichment"]["text"],
                "Producing isotopically enriched material is challenging.",
            )
            self.assertFalse(any("exact isotope" in status for status in statuses))
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

    def test_synthesizer_replaces_chunk_labels_with_paper_sources(self):
        synth = KnowledgeSynthesizer()
        synth._generate = MagicMock(
            return_value="[Fact 1] Storage remained stable. (Source: [Chunk 1], [Chunk 2])"
        )

        result = synth.synthesize(
            [
                {"text": "first storage result", "source": "PaperA"},
                {"text": "second storage result", "source": "PaperA"},
            ],
            query="What storage stability was reported?",
            on_status=[].append,
        )

        self.assertNotIn("[Chunk", result)
        self.assertIn("Source: PaperA", result)
        self.assertNotIn("PaperA, PaperA", result)

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
