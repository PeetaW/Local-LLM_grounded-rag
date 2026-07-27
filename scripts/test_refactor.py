#!/usr/bin/env python3
"""
Unit tests for query engine refactor modules (QE-11 regression).

Tests pure functions directly; mocks external services (Ollama, LLM, file I/O)
so the suite can run without a running server.

Run:
  python scripts/test_refactor.py
  python -m pytest scripts/test_refactor.py -v
"""
import sys
import os
import json
import unittest
from unittest.mock import MagicMock, patch, mock_open

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))

# ── Windows UTF-8 fix ────────────────────────────────────────────────────────
# Debug print statements in the rag modules use emoji (🔬 ⏳ 📋 🔗 …).
# Python on Windows defaults stdout to cp936/cp1252; reconfigure to UTF-8 so
# those prints don't raise UnicodeEncodeError during tests.
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

# ── requests stub ───────────────────────────────────────────────────────────
# `requests` may not be installed in the bare CPython used to run the tests.
# Stub it before any rag module is imported so @patch("requests.post") works
# and the `except requests.exceptions.Timeout:` clauses remain catchable.
if "requests" not in sys.modules:
    import types as _types
    _requests_stub = _types.ModuleType("requests")
    class _Timeout(OSError):
        pass
    class _ReqExceptions:
        Timeout = _Timeout
        ConnectionError = ConnectionError
    _requests_stub.exceptions = _ReqExceptions()
    _requests_stub.post = MagicMock()
    sys.modules["requests"] = _requests_stub

# ── Pre-import stubs for heavy / network-bound dependencies ──────────────────
# These must be in sys.modules BEFORE any rag module is imported, because
# query_pipeline.py creates module-level instances (KnowledgeSynthesizer, etc.)
_STUBS = [
    "rag.llm_client",
    "rag.metadata_manager",
    "rag.citation_grounding",
    "rag.comparison_json_validator",
    "rag.knowledge_synthesizer",
    "rag.answer_verifier",
    "llama_index.core",
    "llama_index.core.response_synthesizers",
]
for _mod in _STUBS:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# KnowledgeSynthesizer / AnswerVerifier must be constructable (called at import)
sys.modules["rag.knowledge_synthesizer"].KnowledgeSynthesizer = MagicMock(
    return_value=MagicMock()
)
sys.modules["rag.knowledge_synthesizer"]._comparison_json_validation_errors = MagicMock(
    return_value=[]
)
sys.modules["rag.comparison_json_validator"].comparison_json_validation_errors = MagicMock(
    return_value=[]
)
sys.modules["rag.comparison_json_validator"].exact_isotope_terms = (
    lambda text, require_context=True: ["10B"] if "10b" in str(text).lower() else []
)
sys.modules["rag.answer_verifier"].AnswerVerifier = MagicMock(
    return_value=MagicMock()
)

# ── Now safe to import rag modules ───────────────────────────────────────────
from rag.query_types import PipelineContext, SubqueryTask, SubqueryResult
from rag.query_embedding_guard import (
    _clean_for_embed, _test_embed, _embed_with_retry, prepare_query_text,
)
from rag.query_planning import (
    detect_target_paper, _keyword_prefilter,
    select_relevant_papers, plan_sub_questions,
)
from rag.query_retrieval import (
    is_empty_result, extract_paper_name,
    build_subquery_tasks, run_subqueries_parallel, _nodes_to_evidence_block,
    _clip_evidence_snippet, _query_aware_window,
)
from rag.query_grounding_flow import (
    _extract_direct_citation_section,
    _partition_results_by_section,
    _cited_sources_in_sentence,
    _fetch_grounding_chunks,
    run_grounding_check,
    split_into_sentences,
)
from rag.query_prompts import build_synthesis_prompt, build_fallback_prompt
from rag.query_translation import translate_to_traditional_chinese
from rag.answerability import assess_answerability
import rag.query_pipeline as pipeline_module
import metrics as eval_metrics
import run_eval as eval_run
import config as cfg


# ══════════════════════════════════════════════════════════════════════════════
# query_types
# ══════════════════════════════════════════════════════════════════════════════
class TestQueryTypes(unittest.TestCase):
    def test_pipeline_context_defaults(self):
        ctx = PipelineContext(question="test?")
        self.assertEqual(ctx.question, "test?")
        self.assertEqual(ctx.memory_context, "")
        self.assertFalse(ctx.rag_found_anything)
        self.assertEqual(ctx.nli_report, "")

    def test_mutable_defaults_are_independent(self):
        a = PipelineContext(question="a")
        b = PipelineContext(question="b")
        a.sub_answers.append("x")
        self.assertEqual(b.sub_answers, [])

    def test_subquery_task_fields(self):
        eng = MagicMock()
        task = SubqueryTask(idx=0, label="【PaperA】", engine=eng, sub_q="what is X?")
        self.assertEqual(task.idx, 0)
        self.assertEqual(task.label, "【PaperA】")
        self.assertEqual(task.sub_q, "what is X?")


# ══════════════════════════════════════════════════════════════════════════════
# query_embedding_guard — _clean_for_embed (pure)
# ══════════════════════════════════════════════════════════════════════════════
class TestCleanForEmbed(unittest.TestCase):
    def test_removes_null_control_char(self):
        result = _clean_for_embed("hello\x00world")
        self.assertNotIn("\x00", result)
        self.assertIn("hello", result)
        self.assertIn("world", result)

    def test_replaces_lt_with_number(self):
        result = _clean_for_embed("size <10 nm")
        self.assertIn("less than", result)
        self.assertNotIn("<10", result)

    def test_replaces_gt_with_number(self):
        result = _clean_for_embed("temperature >50 celsius")
        self.assertIn("greater than", result)
        self.assertNotIn(">50", result)

    def test_removes_long_parentheses(self):
        result = _clean_for_embed("word (this is a very long parenthetical note here) end")
        self.assertNotIn("this is a very long", result)
        self.assertIn("word", result)
        self.assertIn("end", result)

    def test_preserves_clean_text_unchanged(self):
        text = "The synthesis used Fe3O4 nanoparticles."
        self.assertEqual(_clean_for_embed(text), text)

    def test_converts_full_width_parens(self):
        result = _clean_for_embed("value（10）units")
        self.assertIn("(10)", result)


# ══════════════════════════════════════════════════════════════════════════════
# query_embedding_guard — _test_embed (mock requests.post)
# ══════════════════════════════════════════════════════════════════════════════
class TestTestEmbed(unittest.TestCase):
    @patch("requests.post")
    def test_ok_response(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        mock_post.return_value = mock_resp
        self.assertEqual(_test_embed("test text"), "ok")

    @patch("requests.post")
    def test_nan_in_embedding_list_returns_nan(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"embedding": [float("nan"), 0.2, 0.3]}
        mock_post.return_value = mock_resp
        self.assertEqual(_test_embed("test text"), "nan")

    @patch("requests.post")
    def test_http_500_with_nan_message_returns_nan(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "error: NaN encountered in embeddings"
        mock_post.return_value = mock_resp
        self.assertEqual(_test_embed("test text"), "nan")

    @patch("requests.post")
    def test_timeout_returns_timeout(self, mock_post):
        # Use the Timeout class from our stub (or real requests if installed)
        import requests as _r
        mock_post.side_effect = _r.exceptions.Timeout("connection timed out")
        self.assertEqual(_test_embed("test text"), "timeout")

    @patch("requests.post")
    def test_empty_embedding_returns_error(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"embedding": []}
        mock_post.return_value = mock_resp
        self.assertEqual(_test_embed("test text"), "error")


# ══════════════════════════════════════════════════════════════════════════════
# query_embedding_guard — _embed_with_retry / prepare_query_text
# ══════════════════════════════════════════════════════════════════════════════
class TestEmbedWithRetry(unittest.TestCase):
    @patch("rag.query_embedding_guard._test_embed", return_value="ok")
    def test_ok_returns_true(self, _):
        self.assertTrue(_embed_with_retry("text"))

    @patch("rag.query_embedding_guard._test_embed", return_value="nan")
    def test_nan_returns_false_without_retrying(self, mock_test):
        result = _embed_with_retry("text", max_retries=5)
        self.assertFalse(result)
        mock_test.assert_called_once()  # no further retries on NaN

    @patch("time.sleep")
    @patch("rag.query_embedding_guard._test_embed", side_effect=["timeout", "timeout", "ok"])
    def test_retries_on_timeout_then_succeeds(self, mock_test, _sleep):
        result = _embed_with_retry("text", max_retries=5)
        self.assertTrue(result)
        self.assertEqual(mock_test.call_count, 3)

    @patch("time.sleep")
    @patch("rag.query_embedding_guard._test_embed", return_value="timeout")
    def test_exhausted_retries_returns_false(self, mock_test, _sleep):
        result = _embed_with_retry("text", max_retries=3)
        self.assertFalse(result)
        self.assertEqual(mock_test.call_count, 3)


class TestPrepareQueryText(unittest.TestCase):
    @patch("rag.query_embedding_guard._embed_with_retry", return_value=True)
    def test_clean_text_passes_through(self, _):
        text = "simple clean query without special chars"
        result = prepare_query_text(text)
        self.assertEqual(result, text)

    @patch("rag.query_embedding_guard._embed_with_retry", return_value=False)
    def test_truncates_when_embed_persistently_fails(self, _):
        text = "A" * 300
        result = prepare_query_text(text)
        self.assertLessEqual(len(result), len(text))

    @patch("rag.query_embedding_guard._embed_with_retry", side_effect=[False, False, True])
    def test_truncated_clean_text_returned_when_both_fail(self, mock_retry):
        # cleaned fails (call 1), original fails (call 2),
        # first truncation of cleaned succeeds (call 3) → result has no \x00
        text = "query\x00with control char longer than thirty chars"
        result = prepare_query_text(text)
        self.assertNotIn("\x00", result)
        self.assertEqual(mock_retry.call_count, 3)


# ══════════════════════════════════════════════════════════════════════════════
# query_planning — detect_target_paper (pure)
# ══════════════════════════════════════════════════════════════════════════════
class TestDetectTargetPaper(unittest.TestCase):
    _PAPERS = [
        "1-s2.0-S1234567890-main",
        "1-s2.0-S9876543210-main",
        "unrelated-paper-2021",
    ]

    def test_detects_paper_id_substring(self):
        result = detect_target_paper("What does S1234567890 say about iron?", self._PAPERS)
        self.assertEqual(result, "1-s2.0-S1234567890-main")

    def test_returns_none_when_no_match(self):
        result = detect_target_paper("Tell me about nanotechnology in general", self._PAPERS)
        self.assertIsNone(result)

    def test_selects_best_scoring_paper(self):
        result = detect_target_paper("S1234567890 main findings", self._PAPERS)
        self.assertEqual(result, "1-s2.0-S1234567890-main")

    def test_empty_paper_list_returns_none(self):
        self.assertIsNone(detect_target_paper("some question", []))

    def test_minimum_score_threshold(self):
        # Single short segment "ab" (len <= 3) should not match
        papers = ["ab-cd"]
        result = detect_target_paper("ab question", papers)
        self.assertIsNone(result)


# ══════════════════════════════════════════════════════════════════════════════
# query_planning — _keyword_prefilter (mock load_metadata)
# ══════════════════════════════════════════════════════════════════════════════
class TestKeywordPrefilter(unittest.TestCase):
    _META = {
        "nanoparticle-synthesis-2020": {
            "keywords": ["nano", "iron", "synthesis"],
            "main_topic": "nanoparticle synthesis",
            "short_desc": "Synthesis of iron nanoparticles",
        },
        "polymer-degradation-2019": {
            "keywords": ["polymer", "degradation"],
            "main_topic": "polymer science",
            "short_desc": "Polymer degradation study",
        },
    }

    def test_filters_to_matching_paper(self):
        with patch("rag.metadata_manager.load_metadata", return_value=self._META):
            result = _keyword_prefilter("iron nanoparticle synthesis", list(self._META.keys()))
        self.assertIn("nanoparticle-synthesis-2020", result)
        self.assertNotIn("polymer-degradation-2019", result)

    def test_falls_back_to_all_on_no_match(self):
        papers = list(self._META.keys())
        with patch("rag.metadata_manager.load_metadata", return_value=self._META):
            result = _keyword_prefilter("xyz completely unrelated topic", papers)
        self.assertEqual(sorted(result), sorted(papers))

    def test_no_question_terms_returns_all(self):
        papers = list(self._META.keys())
        with patch("rag.metadata_manager.load_metadata", return_value=self._META):
            result = _keyword_prefilter("!!! ???", papers)
        self.assertEqual(sorted(result), sorted(papers))


# ══════════════════════════════════════════════════════════════════════════════
# query_planning — select_relevant_papers (mock LLM + metadata)
# ══════════════════════════════════════════════════════════════════════════════
class TestSelectRelevantPapers(unittest.TestCase):
    _META = {
        "paper_a": {"short_desc": "desc A", "keywords": ["kw1"]},
        "paper_b": {"short_desc": "desc B", "keywords": ["kw2"]},
    }

    def _llm_resp(self, text):
        r = MagicMock()
        r.text = text
        return r

    def test_returns_valid_selected_papers(self):
        papers = list(self._META.keys())
        with patch("rag.metadata_manager.load_metadata", return_value=self._META):
            with patch("rag.llm_client.planning_llm") as mock_llm:
                mock_llm.complete.return_value = self._llm_resp('["paper_a"]')
                result = select_relevant_papers("question", papers)
        self.assertEqual(result, ["paper_a"])

    def test_fallback_on_json_parse_error(self):
        papers = list(self._META.keys())
        with patch("rag.metadata_manager.load_metadata", return_value=self._META):
            with patch("rag.llm_client.planning_llm") as mock_llm:
                mock_llm.complete.return_value = self._llm_resp("not valid json at all")
                result = select_relevant_papers("question", papers)
        self.assertEqual(sorted(result), sorted(papers))

    def test_fallback_when_selected_names_not_in_list(self):
        papers = list(self._META.keys())
        with patch("rag.metadata_manager.load_metadata", return_value=self._META):
            with patch("rag.llm_client.planning_llm") as mock_llm:
                mock_llm.complete.return_value = self._llm_resp('["nonexistent_paper"]')
                result = select_relevant_papers("question", papers)
        self.assertEqual(sorted(result), sorted(papers))

    def test_strips_think_tags_from_response(self):
        papers = list(self._META.keys())
        raw = '<think>some reasoning</think>\n["paper_b"]'
        with patch("rag.metadata_manager.load_metadata", return_value=self._META):
            with patch("rag.llm_client.planning_llm") as mock_llm:
                mock_llm.complete.return_value = self._llm_resp(raw)
                result = select_relevant_papers("question", papers)
        self.assertEqual(result, ["paper_b"])

    def test_comparison_selection_drops_redundant_supplement(self):
        parent = "LAT1 ChemComm 2026"
        supplement = "LAT1 ChemComm 2026SI"
        other = "s41421-024-00697-6"
        papers = [parent, supplement, other]
        meta = {
            paper: {"short_desc": "LAT1 transporter evidence", "keywords": ["LAT1"]}
            for paper in papers
        }
        captured = {}

        def complete(prompt):
            captured["prompt"] = prompt
            return self._llm_resp(json.dumps([parent, other]))

        with (
            patch("rag.metadata_manager.load_metadata", return_value=meta),
            patch("rag.llm_client.planning_llm") as mock_llm,
        ):
            mock_llm.complete.side_effect = complete
            result = select_relevant_papers(
                "Compare LAT1 with other transporters across the literature.",
                papers,
            )

        self.assertEqual(result, [parent, other])
        self.assertNotIn(f"- {supplement}：", captured["prompt"])


class TestPlanSubQuestions(unittest.TestCase):
    def _llm_resp(self, text):
        r = MagicMock()
        r.text = text
        return r

    def test_comparison_prompt_preserves_original_dimensions(self):
        papers = ["paper_a"]
        meta = {"paper_a": {"short_desc": "desc", "main_topic": "topic"}}
        captured = {}

        def complete(prompt):
            captured["prompt"] = prompt
            return self._llm_resp('[{"paper": "paper_a", "sub_q": "Q?"}]')

        with patch("rag.metadata_manager.load_metadata", return_value=meta):
            with patch("rag.llm_client.planning_llm") as mock_llm:
                mock_llm.model = "planner"
                mock_llm.complete.side_effect = complete
                result = plan_sub_questions(
                    "Compare synthetic routes focusing on isotopic enrichment, scalability, and cost-effectiveness.",
                    papers,
                )

        self.assertEqual(result[0]["paper"], "paper_a")
        prompt = captured["prompt"]
        self.assertIn("跨論文比較", prompt)
        self.assertIn("reports, reviews, or compares high-level synthetic approaches", prompt)
        self.assertIn("isotopic enrichment, scalability, cost-effectiveness, safety", prompt)
        self.assertIn("不要詢問 exhaustive procedural details", prompt)
        self.assertIn("每個面向至少要有一個聚焦子問題", prompt)
        self.assertIn("額外建立一個子問題驗證該前提", prompt)

    def test_dedupes_tasks_and_drops_all_when_every_paper_is_covered(self):
        papers = ["paper_a", "paper_b"]
        raw = json.dumps([
            {"paper": "paper_a", "sub_q": "Question A?"},
            {"paper": "paper_a", "sub_q": "Question A?"},
            {"paper": "paper_b", "sub_q": "Question B?"},
            {"paper": "ALL", "sub_q": "Compare A and B?"},
        ])
        meta = {paper: {"short_desc": "desc", "main_topic": "topic"} for paper in papers}

        with patch("rag.metadata_manager.load_metadata", return_value=meta):
            with patch("rag.llm_client.planning_llm") as mock_llm:
                mock_llm.model = "planner"
                mock_llm.complete.return_value = self._llm_resp(raw)
                result = plan_sub_questions("Compare the papers.", papers)

        self.assertEqual([sq["paper"] for sq in result], papers)

    def test_method_protocol_gets_deterministic_retrieval_facet(self):
        papers = ["paper_a"]
        meta = {"paper_a": {"short_desc": "desc", "main_topic": "topic"}}
        raw = json.dumps([{
            "paper": "paper_a",
            "sub_q": "What is the solvent-free N-Boc protection protocol?",
        }])

        with (
            patch.object(cfg, "METHOD_RETRIEVAL_FACET_GUARD_ENABLED", True),
            patch("rag.metadata_manager.load_metadata", return_value=meta),
            patch("rag.llm_client.planning_llm") as mock_llm,
        ):
            mock_llm.model = "planner"
            mock_llm.complete.return_value = self._llm_resp(raw)
            result = plan_sub_questions(
                "What is the solvent-free N-Boc protection protocol and its reaction conditions?",
                papers,
            )

        self.assertEqual(len(result), 2)
        self.assertEqual(result[1]["paper"], "paper_a")
        self.assertIn("optimized yield", result[1]["sub_q"])
        self.assertIn("control or comparison outcomes", result[1]["sub_q"])

    def test_therapeutic_effect_gets_quantitative_outcome_facet(self):
        papers = ["paper_a"]
        meta = {"paper_a": {"short_desc": "desc", "main_topic": "topic"}}
        raw = json.dumps([{
            "paper": "paper_a",
            "sub_q": "What therapeutic effect was observed?",
        }])

        with (
            patch.object(cfg, "OUTCOME_RETRIEVAL_FACET_GUARD_ENABLED", True),
            patch("rag.metadata_manager.load_metadata", return_value=meta),
            patch("rag.llm_client.planning_llm") as mock_llm,
        ):
            mock_llm.model = "planner"
            mock_llm.complete.return_value = self._llm_resp(raw)
            result = plan_sub_questions(
                "What therapeutic effect and supporting data were reported?",
                papers,
            )

        self.assertEqual(len(result), 2)
        self.assertIn("survival time", result[1]["sub_q"])
        self.assertIn("treatment and control groups", result[1]["sub_q"])

    def test_water_stable_query_gets_quantitative_stability_facet(self):
        papers = ["paper_a"]
        meta = {"paper_a": {"short_desc": "desc", "main_topic": "topic"}}
        raw = json.dumps([{
            "paper": "paper_a",
            "sub_q": "How do dynamic bonds support fluoride binding and hydrogel formation?",
        }])

        with (
            patch.object(cfg, "OUTCOME_RETRIEVAL_FACET_GUARD_ENABLED", True),
            patch("rag.metadata_manager.load_metadata", return_value=meta),
            patch("rag.llm_client.planning_llm") as mock_llm,
        ):
            mock_llm.model = "planner"
            mock_llm.complete.return_value = self._llm_resp(raw)
            result = plan_sub_questions(
                "What is the water-stable boroxine structure and how does it form a hydrogel?",
                papers,
            )

        self.assertEqual(len(result), 2)
        self.assertIn("study duration", result[1]["sub_q"])
        self.assertIn("exact pH ranges", result[1]["sub_q"])


# ══════════════════════════════════════════════════════════════════════════════
# query_retrieval — is_empty_result (pure)
# ══════════════════════════════════════════════════════════════════════════════
class TestIsEmptyResult(unittest.TestCase):
    def test_short_text_is_empty(self):
        self.assertTrue(is_empty_result("short"))

    def test_exactly_30_chars_is_not_empty(self):
        self.assertFalse(is_empty_result("x" * 30))

    def test_no_result_zh_pattern(self):
        self.assertTrue(is_empty_result("此論文未涉及任何跟這個主題相關的內容，因此無法提供答案"))

    def test_no_result_en_pattern(self):
        self.assertTrue(is_empty_result(
            "The context does not contain information about this topic at all."
        ))

    def test_substantive_content_is_not_empty(self):
        self.assertFalse(is_empty_result(
            "The synthesis involved mixing Fe3O4 nanoparticles with EDTA "
            "in a 3:1 molar ratio at 80°C for 4 hours under nitrogen atmosphere."
        ))


# ══════════════════════════════════════════════════════════════════════════════
# query_retrieval — extract_paper_name (pure)
# ══════════════════════════════════════════════════════════════════════════════
class TestExtractPaperName(unittest.TestCase):
    def test_extracts_first_bracket(self):
        self.assertEqual(
            extract_paper_name("【MyPaper2021】Some content about results.", "fb"),
            "MyPaper2021",
        )

    def test_returns_first_when_multiple_brackets(self):
        self.assertEqual(
            extract_paper_name("【First】 and 【Second】", "fb"),
            "First",
        )

    def test_fallback_when_no_bracket(self):
        self.assertEqual(extract_paper_name("no brackets at all", "fallback"), "fallback")


# ══════════════════════════════════════════════════════════════════════════════
# query_retrieval — build_subquery_tasks
# ══════════════════════════════════════════════════════════════════════════════
class TestBuildSubqueryTasks(unittest.TestCase):
    def setUp(self):
        self.eng_a = MagicMock()
        self.eng_b = MagicMock()
        self.engines = {"paper_a": self.eng_a, "paper_b": self.eng_b}

    def test_all_creates_one_task_per_engine(self):
        sub_q = [{"paper": "ALL", "sub_q": "What is X?"}]
        valid, prefilled = build_subquery_tasks(sub_q, self.engines, self.engines)
        self.assertEqual(len(valid), 2)
        self.assertEqual(len(prefilled), 0)

    def test_all_only_expands_to_papers_without_specific_task(self):
        sub_q = [
            {"paper": "paper_a", "sub_q": "Specific A?"},
            {"paper": "ALL", "sub_q": "Compare all?"},
        ]
        valid, _ = build_subquery_tasks(sub_q, self.engines, self.engines)
        self.assertEqual([(task[1], task[3]) for task in valid], [
            ("【paper_a】", "Specific A?"),
            ("【paper_b】", "Compare all?"),
        ])

    def test_specific_paper_found_creates_one_task(self):
        sub_q = [{"paper": "paper_a", "sub_q": "Q?"}]
        valid, prefilled = build_subquery_tasks(sub_q, self.engines, self.engines)
        self.assertEqual(len(valid), 1)
        self.assertEqual(valid[0][1], "【paper_a】")
        self.assertEqual(valid[0][3], "Q?")

    def test_missing_paper_goes_to_prefilled(self):
        sub_q = [{"paper": "missing_paper", "sub_q": "Q?"}]
        valid, prefilled = build_subquery_tasks(sub_q, {}, {})
        self.assertEqual(len(valid), 0)
        self.assertIn(0, prefilled)
        self.assertIn("找不到對應論文", prefilled[0][1])

    def test_partial_name_match_finds_engine(self):
        # "paper" is a substring of "paper_a"
        sub_q = [{"paper": "paper", "sub_q": "Q?"}]
        valid, prefilled = build_subquery_tasks(sub_q, self.engines, self.engines)
        self.assertEqual(len(valid), 1)

    def test_sequential_idx_across_multiple_sub_questions(self):
        sub_q = [
            {"paper": "ALL", "sub_q": "Q1?"},
            {"paper": "paper_a", "sub_q": "Q2?"},
        ]
        valid, _ = build_subquery_tasks(sub_q, self.engines, self.engines)
        indices = [t[0] for t in valid]
        self.assertEqual(indices, sorted(indices))


# ══════════════════════════════════════════════════════════════════════════════
# query_retrieval — run_subqueries_parallel
# ══════════════════════════════════════════════════════════════════════════════
class TestRunSubqueriesParallel(unittest.TestCase):
    @patch("rag.query_retrieval._generate_from_nodes", return_value="Generated answer")
    @patch("rag.query_retrieval._retrieve_nodes", return_value=["node1"])
    @patch("rag.query_retrieval.prepare_query_text", return_value="clean query")
    def test_returns_results_for_all_tasks(self, *_):
        eng = MagicMock()
        valid_tasks = [(0, "【PaperA】", eng, "Q1?"), (1, "【PaperB】", eng, "Q2?")]
        results = run_subqueries_parallel(valid_tasks, {})
        self.assertEqual(len(results), 2)
        labels = [r[0] for r in results]
        self.assertIn("【PaperA】", labels)
        self.assertIn("【PaperB】", labels)

    @patch("rag.query_retrieval._generate_from_nodes", return_value="Answer")
    @patch("rag.query_retrieval._retrieve_nodes", return_value=[])
    @patch("rag.query_retrieval.prepare_query_text", return_value="q")
    def test_prefilled_entries_appear_in_correct_order(self, *_):
        valid_tasks = [(1, "【PaperA】", MagicMock(), "Q?")]
        prefilled = {0: ("【PaperB】", "找不到對應論文")}
        results = run_subqueries_parallel(valid_tasks, prefilled)
        self.assertEqual(len(results), 2)
        # idx 0 (prefilled) must come first
        self.assertEqual(results[0][0], "【PaperB】")
        self.assertEqual(results[1][0], "【PaperA】")

    @patch("rag.query_retrieval._generate_from_nodes", return_value="Answer")
    @patch("rag.query_retrieval._retrieve_nodes", return_value=[])
    @patch("rag.query_retrieval.prepare_query_text", return_value="q")
    def test_emits_retrieval_timing_status(self, *_):
        statuses = []
        valid_tasks = [(0, "【PaperA】", MagicMock(), "Q?")]
        run_subqueries_parallel(valid_tasks, {}, on_status=statuses.append)
        self.assertTrue(any("[retrieval-timing]" in s for s in statuses))

    @patch("rag.query_retrieval._generate_from_nodes", return_value="Generated answer")
    @patch("rag.query_retrieval._retrieve_nodes", return_value=["raw evidence"])
    @patch("rag.query_retrieval.prepare_query_text", return_value="clean query")
    def test_evidence_mode_skips_subanswer_llm(self, _, __, mock_generate):
        old = cfg.STAGE2_LLM_SUBANSWERS_ENABLED
        try:
            cfg.STAGE2_LLM_SUBANSWERS_ENABLED = False
            with patch("rag.metadata_manager.load_metadata", return_value={
                "PaperA": {"title": "A Comprehensive Review of Routes", "short_desc": ""}
            }):
                results = run_subqueries_parallel([(0, "【PaperA】", MagicMock(), "Q?")], {})
            self.assertIn("Retrieved evidence snippets", results[0][1])
            self.assertIn("role_hint=review/comparison source", results[0][1])
            self.assertIn("not paper evidence", results[0][1])
            mock_generate.assert_not_called()
        finally:
            cfg.STAGE2_LLM_SUBANSWERS_ENABLED = old

    def test_comparison_evidence_block_uses_more_snippets(self):
        old_base = getattr(cfg, "STAGE2_EVIDENCE_SNIPPETS_PER_TASK", 2)
        old_compare = getattr(cfg, "COMPARISON_EVIDENCE_SNIPPETS_PER_TASK", 4)
        old_query_aware = getattr(cfg, "STAGE2_QUERY_AWARE_EVIDENCE_ENABLED", False)
        try:
            cfg.STAGE2_EVIDENCE_SNIPPETS_PER_TASK = 2
            cfg.COMPARISON_EVIDENCE_SNIPPETS_PER_TASK = 4
            cfg.STAGE2_QUERY_AWARE_EVIDENCE_ENABLED = False
            nodes = [f"snippet {i}" for i in range(1, 6)]

            normal = _nodes_to_evidence_block(nodes, "What are key steps?", "【PaperA】")
            self.assertIn("[Snippet 2]", normal)
            self.assertNotIn("[Snippet 3]", normal)

            comparison = _nodes_to_evidence_block(nodes, "Compare routes for scalability.", "【PaperA】")
            self.assertIn("[Snippet 4]", comparison)
            self.assertNotIn("[Snippet 5]", comparison)
        finally:
            cfg.STAGE2_EVIDENCE_SNIPPETS_PER_TASK = old_base
            cfg.COMPARISON_EVIDENCE_SNIPPETS_PER_TASK = old_compare
            cfg.STAGE2_QUERY_AWARE_EVIDENCE_ENABLED = old_query_aware

    def test_query_aware_evidence_selects_relevant_tail_node(self):
        old = getattr(cfg, "STAGE2_QUERY_AWARE_EVIDENCE_ENABLED", False)
        try:
            cfg.STAGE2_QUERY_AWARE_EVIDENCE_ENABLED = True
            nodes = [f"Generic introductory discussion {i}." for i in range(4)]
            nodes.append(
                "[摘要：Generated context must not be cited.]\n\n"
                + "Unrelated background sentence. " * 80
                + "Raw BPA powder shows no detectable degradation after storage at 55 C for 6 months."
            )
            block = _nodes_to_evidence_block(
                nodes,
                "What degradation is reported under the storage conditions?",
                "【PaperA】",
            )
            self.assertIn("storage at 55 C for 6 months", block)
            self.assertNotIn("[摘要：", block)
            self.assertIn("[Snippet 2]", block)
            self.assertNotIn("[Snippet 3]", block)
        finally:
            cfg.STAGE2_QUERY_AWARE_EVIDENCE_ENABLED = old

    def test_query_aware_evidence_prioritizes_detected_impurities(self):
        old = getattr(cfg, "STAGE2_QUERY_AWARE_EVIDENCE_ENABLED", False)
        try:
            cfg.STAGE2_QUERY_AWARE_EVIDENCE_ENABLED = True
            nodes = [
                "The HPLC method used a C18 column and a standard mobile phase.",
                "The compound was stored in sealed containers for routine testing.",
                (
                    "BrPD and FBBA were detected as degradation products and eluted before BPA. "
                    "BDPA was also detectable at the reported concentration."
                ),
            ]
            block = _nodes_to_evidence_block(
                nodes,
                "Which impurities and degradation products were detected by HPLC during storage?",
                "【PaperA】",
            )
            self.assertIn("BrPD and FBBA were detected", block)
            self.assertIn("[Snippet 2]", block)
            self.assertNotIn("[Snippet 3]", block)
        finally:
            cfg.STAGE2_QUERY_AWARE_EVIDENCE_ENABLED = old

    def test_query_aware_value_window_carries_preceding_numeric_result(self):
        text = (
            "Introductory background without reported values. " * 15
            + "Under solvent-free conditions, 0.2 equivalents of catalyst for 60 min "
            "gave an optimized yield of 95%. "
            + "Encouraged by this outcome, the optimized reaction conditions, catalyst loading, "
            "and reaction time were applied across a broad substrate scope. "
            + "Additional substrate descriptions followed. " * 15
        )
        window = _query_aware_window(
            text,
            "What optimized reaction conditions, catalyst loading, reaction time, and yield were reported?",
            240,
        )

        self.assertIn("optimized yield of 95%", window)
        self.assertIn("optimized reaction conditions", window)

    def test_query_aware_evidence_uses_a_complementary_second_snippet(self):
        old_query_aware = cfg.STAGE2_QUERY_AWARE_EVIDENCE_ENABLED
        old_diverse = cfg.STAGE2_DIVERSE_EVIDENCE_ENABLED
        try:
            cfg.STAGE2_QUERY_AWARE_EVIDENCE_ENABLED = True
            nodes = [
                "BPA degradation products and impurities were detected during storage. DUPLICATE_A",
                "BPA degradation products and impurities were detected during storage. DUPLICATE_B",
                (
                    "The lyophilized BPA-mannitol drug product showed slow temperature-dependent "
                    "degradation to phenylalanine, reaching 1% at 40 C over 6 months. COMPLEMENT"
                ),
            ]
            question = "Which impurities and degradation products form during storage?"

            cfg.STAGE2_DIVERSE_EVIDENCE_ENABLED = False
            control = _nodes_to_evidence_block(nodes, question)
            self.assertNotIn("COMPLEMENT", control)

            cfg.STAGE2_DIVERSE_EVIDENCE_ENABLED = True
            diverse = _nodes_to_evidence_block(nodes, question)
            self.assertIn("COMPLEMENT", diverse)
            self.assertEqual(diverse.count("[Snippet "), 2)
        finally:
            cfg.STAGE2_QUERY_AWARE_EVIDENCE_ENABLED = old_query_aware
            cfg.STAGE2_DIVERSE_EVIDENCE_ENABLED = old_diverse

    def test_evidence_snippet_override_is_used_for_partial_recovery(self):
        old = cfg.STAGE2_QUERY_AWARE_EVIDENCE_ENABLED
        try:
            cfg.STAGE2_QUERY_AWARE_EVIDENCE_ENABLED = False
            block = _nodes_to_evidence_block(
                ["first evidence", "second evidence", "third evidence"],
                "What was reported?",
                snippet_count_override=3,
            )
            self.assertIn("[Snippet 3]", block)
        finally:
            cfg.STAGE2_QUERY_AWARE_EVIDENCE_ENABLED = old

    def test_partial_recovery_can_add_one_adjacent_witness(self):
        related = MagicMock()
        related.node_id = "next-node"

        anchor = MagicMock()
        anchor.node_id = "anchor-node"
        anchor.relationships = {"3": related}
        anchor.metadata = {}
        anchor.get_content.return_value = "BPA storage setup was evaluated."
        class AdjacentNode:
            node_id = "next-node"
            metadata = {}

            @staticmethod
            def get_content():
                return "BPA degraded to phenylalanine, reaching approximately 1% at 40 C over 6 months."

        adjacent = AdjacentNode()
        wrapped = MagicMock()
        wrapped.node = anchor

        docstore = MagicMock()
        docstore.get_node.return_value = adjacent
        index = MagicMock()
        index.docstore = docstore
        child = MagicMock()
        child._index = index
        engine = MagicMock()
        engine.retriever._index = None
        engine.retriever.index = None
        engine.retriever._retrievers = [child]

        block = _nodes_to_evidence_block(
            [wrapped],
            "Under which storage conditions does BPA degradation occur?",
            snippet_count_override=1,
            engine=engine,
            include_adjacent=True,
        )

        self.assertIn("approximately 1% at 40 C over 6 months", block)
        self.assertIn("[Snippet 2]", block)

    def test_comparison_snippet_clips_around_dimension_terms(self):
        text = (
            "opening procedural background " * 80
            + "The review highlights limitations regarding scalability, cost-effectiveness, "
            + "and safety, especially considering the high cost of isotopically enriched 10B. "
            + "trailing text " * 80
        )
        clipped = _clip_evidence_snippet(
            text,
            "Compare routes for isotopic enrichment, scalability, and cost-effectiveness.",
            limit=260,
        )
        self.assertIn("cost-effectiveness", clipped)
        self.assertIn("high cost of isotopically enriched 10B", clipped)

    def test_comparison_snippet_finishes_nearby_sentence(self):
        text = (
            "introductory material " * 8
            + "The high cost is more than 1000 fold that of normal boric acid. "
            + "A separate sentence must stay outside the clip. "
            + "trailing material " * 20
        )
        clipped = _clip_evidence_snippet(text, "Compare route cost-effectiveness.", limit=90)
        self.assertTrue(clipped.endswith("normal boric acid."))

    def test_comparison_snippet_uses_raw_safety_evidence_not_summary(self):
        text = (
            "[摘要：NaIO4-based synthesis poses contamination and safety risks.]\n\n"
            + "opening material " * 35
            + "The oral LD50 values for NaIO4 are reported in rats. "
            + "Considering this toxicity potential, late-stage deprotection raises concerns. "
            + "There is a substantial risk of contamination of the final L-BPA with NaIO4. "
            + "The use of any oxidant on scale is also inherently a process safety risk. "
            + "trailing material " * 35
        )
        clipped = _clip_evidence_snippet(
            text,
            "Compare routes for isotopic enrichment, scalability, cost-effectiveness, and safety.",
            limit=420,
        )
        self.assertNotIn("[摘要：", clipped)
        self.assertIn("toxicity potential", clipped)
        self.assertIn("risk of contamination", clipped)
        self.assertIn("process safety risk", clipped)

    def test_atomic_comparison_snippets_keep_top_two_and_dimension_overview(self):
        old = cfg.COMPARISON_JSON_DIRECT_RENDER_ENABLED
        try:
            cfg.COMPARISON_JSON_DIRECT_RENDER_ENABLED = True

            def nws(text, source_type="pdf_text"):
                node = MagicMock()
                node.metadata = {"source_type": source_type}
                node.get_content.return_value = text
                return MagicMock(node=node)

            nodes = [
                nws("top route evidence"),
                nws("top isotope context evidence"),
                nws("generic isotopically enriched compounds have a major cost"),
                nws("The cost of 10B is high compared with normal boric acid"),
                nws("overview: isotopically enriched material, scalability, cost-effectiveness, and safety"),
                nws("image safety noise", "image_description"),
            ]
            block = _nodes_to_evidence_block(
                nodes,
                "Compare routes for isotopic enrichment, scalability, and cost-effectiveness.",
                "【ReviewA】",
            )

            self.assertIn("top route evidence", block)
            self.assertIn("top isotope context evidence", block)
            self.assertIn("The cost of 10B is high", block)
            self.assertIn("overview: isotopically enriched material", block)
            self.assertNotIn("generic isotopically enriched compounds", block)
            self.assertNotIn("image safety noise", block)
        finally:
            cfg.COMPARISON_JSON_DIRECT_RENDER_ENABLED = old

    @patch("rag.query_retrieval._generate_from_nodes", return_value="Generated answer")
    @patch("rag.query_retrieval._retrieve_nodes", return_value=["raw evidence"])
    @patch("rag.query_retrieval.prepare_query_text", return_value="clean query")
    def test_subanswer_mode_uses_llm(self, _, __, mock_generate):
        old = cfg.STAGE2_LLM_SUBANSWERS_ENABLED
        try:
            cfg.STAGE2_LLM_SUBANSWERS_ENABLED = True
            results = run_subqueries_parallel([(0, "【PaperA】", MagicMock(), "Q?")], {})
            self.assertEqual(results[0][1], "Generated answer")
            mock_generate.assert_called_once()
        finally:
            cfg.STAGE2_LLM_SUBANSWERS_ENABLED = old


# ══════════════════════════════════════════════════════════════════════════════
# eval.metrics — retrieval timing parse
# ══════════════════════════════════════════════════════════════════════════════
class TestEvalMetrics(unittest.TestCase):
    def test_full_eval_set_uses_structured_english_contracts(self):
        with open(eval_run.EVAL_SET, "r", encoding="utf-8") as handle:
            questions = json.load(handle)["questions"]
        self.assertEqual(len(questions), 12)
        self.assertTrue(all(question.get("reference_facts") for question in questions))
        self.assertFalse(any(
            any("\u4e00" <= char <= "\u9fff" for char in question["reference_answer"])
            for question in questions
        ))

    def test_parse_retrieval_timing(self):
        lat = eval_metrics.parse_stage_latencies([
            "[retrieval] 完成 rag_found=True elapsed_ms=100",
            "[retrieval-timing] tasks=3 prefilled=1 phase_a_ms=20 phase_b_ms=80",
        ])
        self.assertEqual(lat["retrieval"], 100)
        self.assertEqual(lat["retrieval_tasks"], 3)
        self.assertEqual(lat["retrieval_prefilled"], 1)
        self.assertEqual(lat["retrieval_phase_a"], 20)
        self.assertEqual(lat["retrieval_phase_b"], 80)

    def test_summarize_exposes_missing_correctness_scores(self):
        summary = eval_metrics.summarize([
            {"correctness": 0.75},
            {"correctness": None},
        ])
        self.assertEqual(summary["avg_correctness"], 0.75)
        self.assertEqual(summary["n_correctness_scored"], 1)
        self.assertEqual(summary["n_correctness_na"], 1)

    def test_summarize_reports_translation_fidelity_separately(self):
        summary = eval_metrics.summarize([
            {"correctness": 1.0, "translation_fidelity": 0.5},
            {"correctness": 0.75, "translation_fidelity": None},
        ])
        self.assertEqual(summary["avg_translation_fidelity"], 0.5)
        self.assertEqual(summary["n_translation_scored"], 1)
        self.assertEqual(summary["n_translation_na"], 1)

    def test_summarize_separates_candidate_and_stage2_recall(self):
        summary = eval_metrics.summarize([
            {"retrieval_span_recall": 0.5, "retriever_candidate_recall": 1.0, "stage2_evidence_recall": 0.5},
            {"retrieval_span_recall": 1.0, "retriever_candidate_recall": 1.0, "stage2_evidence_recall": 1.0},
        ])
        self.assertEqual(summary["avg_retriever_candidate_recall"], 1.0)
        self.assertEqual(summary["avg_stage2_evidence_recall"], 0.75)


class TestSplitIntoSentences(unittest.TestCase):
    def test_skips_heading_and_splits_english_prose(self):
        claims = split_into_sentences(
            "## Direct Paper Evidence\n"
            "JPH203 binds within the LAT1 substrate-binding pocket. "
            "Its chloride atom forms a halogen bond with Tyr259."
        )
        self.assertEqual(len(claims), 2)
        self.assertFalse(any("Direct Paper Evidence" in claim for claim in claims))

    def test_skips_bold_markdown_headings(self):
        claims = split_into_sentences(
            "**Impurities and Degradation Products Identified by HPLC**\n\n"
            "Raw BPA powder remained stable for 12 months."
        )
        self.assertEqual(claims, ["Raw BPA powder remained stable for 12 months."])

    def test_keeps_numbered_english_proposition(self):
        claims = split_into_sentences(
            "7 ) We now describe an efficient hybrid synthesis illustrated in Scheme."
        )

        self.assertEqual(
            claims,
            ["7 ) We now describe an efficient hybrid synthesis illustrated in Scheme."],
        )

    def test_keeps_substantive_tradeoff_sentence(self):
        claims = split_into_sentences(
            "Comparison scaffold:\n"
            "- Route A produces optically pure material [PaperA].\n"
            "Central trade-off: Higher isotope purity increases precursor cost [ReviewA]."
        )
        self.assertTrue(any(claim.startswith("Central trade-off:") for claim in claims))
        self.assertFalse(any(claim == "Comparison scaffold:" for claim in claims))


# ══════════════════════════════════════════════════════════════════════════════
# query_grounding_flow — section extraction (pure regex)
# ══════════════════════════════════════════════════════════════════════════════
class TestExtractDirectCitationSection(unittest.TestCase):
    def test_extracts_zh_direct_section(self):
        text = (
            "## 【論文直接依據】\nDirect content here.\n\n"
            "## 【跨文獻推論】\nInference here."
        )
        result = _extract_direct_citation_section(text)
        self.assertIn("Direct content here", result)
        self.assertNotIn("Inference here", result)

    def test_extracts_en_direct_evidence_section(self):
        text = (
            "## [Direct Paper Evidence]\nEN direct content.\n\n"
            "## [Cross-Literature Inference]\nother content."
        )
        result = _extract_direct_citation_section(text)
        self.assertIn("EN direct content", result)
        self.assertNotIn("other content", result)

    def test_returns_empty_when_section_absent(self):
        text = "## Some Random Section\nContent without a direct-citation header."
        result = _extract_direct_citation_section(text)
        self.assertEqual(result, "")


class TestCitationAwareGroundingRetrieval(unittest.TestCase):
    def test_extracts_only_known_citation_sources(self):
        sentence = "Claim [PaperA, PaperB]. [Fact 1]"
        self.assertEqual(
            _cited_sources_in_sentence(sentence, ("PaperA", "PaperB", "PaperC")),
            ("PaperA", "PaperB"),
        )

    @patch("rag.query_embedding_guard.prepare_query_text", side_effect=lambda text: text)
    def test_retrieves_with_claims_citing_each_paper(self, _):
        old = cfg.GROUNDING_CITATION_AWARE_ENABLED
        try:
            cfg.GROUNDING_CITATION_AWARE_ENABLED = True
            nws = MagicMock()
            nws.node.node_id = "node-12345678"
            nws.node.get_content.return_value = "[摘要：Generated summary.]\n\nRaw supporting evidence."
            nws.score = 0.9
            engine = MagicMock()
            engine.retriever.retrieve.return_value = [nws]

            chunks = _fetch_grounding_chunks(
                "Original question",
                {"PaperA": engine},
                ["Specific safety claim [PaperA]."],
            )

            query = engine.retriever.retrieve.call_args.args[0]
            self.assertIn("Original question", query)
            self.assertIn("Specific safety claim", query)
            self.assertNotIn("[PaperA]", query)
            self.assertEqual(chunks[0]["source"], "PaperA")
            self.assertEqual(chunks[0]["text"], "Raw supporting evidence.")
        finally:
            cfg.GROUNDING_CITATION_AWARE_ENABLED = old


# ══════════════════════════════════════════════════════════════════════════════
# query_grounding_flow — _partition_results_by_section
# ══════════════════════════════════════════════════════════════════════════════
class TestPartitionResultsBySection(unittest.TestCase):
    def _split(self, text):
        return [s.strip() for s in text.split("\n") if s.strip()]

    def test_separates_direct_and_inference(self):
        text = (
            "## 【論文直接依據】\nFact one.\n\n"
            "## 【跨文獻推論】\nInference one.\n"
        )
        citation_results = [
            {"sentence": "Fact one.", "supported": True, "confidence": 0.9},
            {"sentence": "Inference one.", "supported": False, "confidence": 0.3},
        ]
        with patch("rag.query_grounding_flow.split_into_sentences", side_effect=self._split):
            result = _partition_results_by_section(citation_results, text)

        self.assertIn("direct", result)
        self.assertIn("inference", result)
        self.assertIn("Fact one.", [r["sentence"] for r in result["direct"]])
        self.assertIn("Inference one.", [r["sentence"] for r in result["inference"]])

    def test_absent_sections_not_in_output(self):
        text = "## 【論文直接依據】\nOnly direct content.\n"
        citation_results = [
            {"sentence": "Only direct content.", "supported": True, "confidence": 0.95}
        ]
        with patch("rag.query_grounding_flow.split_into_sentences", side_effect=self._split):
            result = _partition_results_by_section(citation_results, text)

        self.assertIn("direct", result)
        self.assertNotIn("inference", result)
        self.assertNotIn("speculation", result)


# ══════════════════════════════════════════════════════════════════════════════
# query_prompts — pure string builders
# ══════════════════════════════════════════════════════════════════════════════
class TestBuildSynthesisPrompt(unittest.TestCase):
    def test_reasoning_en_has_three_english_tiers(self):
        prompt = build_synthesis_prompt("kb", "q", "", "reasoning", "en")
        self.assertIn("## [Direct Paper Evidence]", prompt)
        self.assertIn("## [Cross-Literature Inference]", prompt)
        self.assertIn("## [Knowledge Extension and Speculation]", prompt)

    def test_reasoning_zh_has_three_chinese_tiers(self):
        prompt = build_synthesis_prompt("kb", "q", "", "reasoning", "zh")
        self.assertIn("## 【論文直接依據】", prompt)
        self.assertIn("## 【跨文獻推論】", prompt)
        self.assertIn("## 【知識延伸與推測】", prompt)

    def test_strict_en_citation_only(self):
        prompt = build_synthesis_prompt("kb", "q", "", "strict", "en")
        self.assertNotIn("## [Cross-Literature Inference]", prompt)
        self.assertIn("Only use the content from the above data", prompt)

    def test_relation_atomicity_is_shared_ab_switch(self):
        old = cfg.FACT_RELATION_ATOMICITY_GUARD_ENABLED
        try:
            cfg.FACT_RELATION_ATOMICITY_GUARD_ENABLED = True
            for mode in ("reasoning", "strict"):
                prompt = build_synthesis_prompt("kb", "q", "", mode, "en")
                self.assertIn("FACT RELATION FIDELITY", prompt)
                self.assertIn("Keep observed results or stability conclusions separate", prompt)

            cfg.FACT_RELATION_ATOMICITY_GUARD_ENABLED = False
            prompt = build_synthesis_prompt("kb", "q", "", "strict", "en")
            self.assertNotIn("FACT RELATION FIDELITY", prompt)
        finally:
            cfg.FACT_RELATION_ATOMICITY_GUARD_ENABLED = old

    def test_strict_zh_citation_only(self):
        prompt = build_synthesis_prompt("kb", "q", "", "strict", "zh")
        self.assertNotIn("## 【跨文獻推論】", prompt)
        self.assertIn("論文名稱", prompt)

    def test_knowledge_base_injected(self):
        kb = "UNIQUE_KB_CONTENT_XYZ_12345"
        prompt = build_synthesis_prompt(kb, "q", "", "strict", "en")
        self.assertIn(kb, prompt)

    def test_memory_section_injected(self):
        mem = "UNIQUE_MEMORY_ABC_67890"
        prompt = build_synthesis_prompt("kb", "q", mem, "reasoning", "en")
        self.assertIn(mem, prompt)

    def test_question_injected(self):
        q = "UNIQUE_QUESTION_TEXT_FOR_TEST"
        prompt = build_synthesis_prompt("kb", q, "", "strict", "zh")
        self.assertIn(q, prompt)

    def test_query_facet_focus_is_ab_switch(self):
        old = cfg.QUERY_FACET_FOCUS_ENABLED
        try:
            cfg.QUERY_FACET_FOCUS_ENABLED = False
            prompt = build_synthesis_prompt("kb", "Report stability and impurities.", "", "strict", "en")
            self.assertNotIn("QUERY FOCUS", prompt)

            cfg.QUERY_FACET_FOCUS_ENABLED = True
            prompt = build_synthesis_prompt("kb", "Report stability and impurities.", "", "strict", "en")
            self.assertIn("QUERY FOCUS", prompt)
            self.assertIn("every distinct item or facet", prompt)
            self.assertIn("Once all requested facets are covered, stop", prompt)
        finally:
            cfg.QUERY_FACET_FOCUS_ENABLED = old

    def test_comparison_tradeoff_guard_is_ab_switch(self):
        old = cfg.COMPARISON_TRADEOFF_GUARD_ENABLED
        old_scaffold = cfg.COMPARISON_QUERY_SCAFFOLD_ENABLED
        try:
            cfg.COMPARISON_QUERY_SCAFFOLD_ENABLED = False
            cfg.COMPARISON_TRADEOFF_GUARD_ENABLED = False
            prompt = build_synthesis_prompt("kb", "Compare A and B", "", "reasoning", "en")
            self.assertNotIn("Central trade-off", prompt)

            cfg.COMPARISON_TRADEOFF_GUARD_ENABLED = True
            prompt = build_synthesis_prompt("kb", "Compare A and B", "", "reasoning", "en")
            self.assertIn("Central trade-off", prompt)
            self.assertIn("high-purity", prompt)
            self.assertIn("corpus-level", prompt)
            self.assertIn("route map", prompt)
            self.assertIn("directly synthesizes the target compound", prompt)
            self.assertIn("Do not list derivative", prompt)
            self.assertIn("enantioselective alkylation followed by enzymatic hydrolysis", prompt)
        finally:
            cfg.COMPARISON_TRADEOFF_GUARD_ENABLED = old
            cfg.COMPARISON_QUERY_SCAFFOLD_ENABLED = old_scaffold

    def test_comparison_query_scaffold_is_ab_switch(self):
        old = cfg.COMPARISON_QUERY_SCAFFOLD_ENABLED
        try:
            cfg.COMPARISON_QUERY_SCAFFOLD_ENABLED = False
            prompt = build_synthesis_prompt("kb", "Compare A and B", "", "reasoning", "en")
            self.assertNotIn("Comparison scaffold", prompt)

            cfg.COMPARISON_QUERY_SCAFFOLD_ENABLED = True
            prompt = build_synthesis_prompt("kb", "Compare A and B", "", "reasoning", "en")
            self.assertIn("Comparison scaffold", prompt)
            self.assertIn("one short bullet per relevant source role", prompt)
            self.assertIn("Do not use a Markdown table", prompt)
            self.assertIn("review/comparison source", prompt)
            self.assertIn("route bullets must directly synthesize the target compound", prompt)
            self.assertIn("chymotrypsin-catalysed enzymatic hydrolysis", prompt)
            self.assertIn("must use the full phrase exactly", prompt)
            self.assertIn("defer \"chymotrypsin\" to a later evidence bullet", prompt)
            self.assertIn("scalability, cost-effectiveness, and safety", prompt)
            self.assertIn("compares multiple approaches", prompt)
            self.assertIn("not background", prompt)
            self.assertIn("exclusion", prompt)
            self.assertIn("do not include background rows", prompt)
            self.assertIn("only route map", prompt)
            self.assertIn("exhaustive historical route variants", prompt)
            self.assertIn("reference-level qualitative summaries", prompt)
            self.assertIn("exact temperatures", prompt)
            self.assertIn("cost multipliers", prompt)
            self.assertIn("patent names", prompt)
            self.assertIn("reagent/catalyst/oxidant names", prompt)
            self.assertIn("broad route identifiers", prompt)
            self.assertIn("high cost/expense", prompt)
            self.assertIn("exactly two short evidence bullets", prompt)
            self.assertIn("followed by one \"Central trade-off:\" sentence", prompt)
            self.assertIn("must preserve the named comparison dimensions", prompt)
            self.assertIn("must literally include \"scalability, cost-effectiveness, and safety\"", prompt)
            self.assertIn("Do not add separate sections or expand beyond the scaffold", prompt)
            self.assertIn("Do not add separate sections", prompt)
            self.assertIn("do not include derivative/formulation/solubility", prompt)
            self.assertIn("do not leave its source implicit", prompt)
        finally:
            cfg.COMPARISON_QUERY_SCAFFOLD_ENABLED = old

    def test_tradeoff_guard_skips_route_map_when_scaffold_enabled(self):
        old_tradeoff = cfg.COMPARISON_TRADEOFF_GUARD_ENABLED
        old_scaffold = cfg.COMPARISON_QUERY_SCAFFOLD_ENABLED
        try:
            cfg.COMPARISON_TRADEOFF_GUARD_ENABLED = True
            cfg.COMPARISON_QUERY_SCAFFOLD_ENABLED = True
            prompt = build_synthesis_prompt("kb", "Compare routes", "", "strict", "en")
            self.assertIn("Central trade-off", prompt)
            self.assertIn("Comparison scaffold", prompt)
            self.assertNotIn("first include a compact route map", prompt)
        finally:
            cfg.COMPARISON_TRADEOFF_GUARD_ENABLED = old_tradeoff
            cfg.COMPARISON_QUERY_SCAFFOLD_ENABLED = old_scaffold

    def test_comparison_json_guides_stage4_when_present(self):
        old = cfg.COMPARISON_JSON_ENABLED
        try:
            cfg.COMPARISON_JSON_ENABLED = False
            kb = '{"comparison_json": {"dimensions": {"safety": {"requested": true, "evidence_found": true}}}}'
            prompt = build_synthesis_prompt(kb, "Compare routes", "", "strict", "en")
            self.assertNotIn("COMPARISON JSON", prompt)

            cfg.COMPARISON_JSON_ENABLED = True
            prompt = build_synthesis_prompt(kb, "Compare routes", "", "strict", "en")
            self.assertIn("COMPARISON JSON", prompt)
            self.assertIn("authoritative outline", prompt)
            self.assertIn("Use only `direct_routes` as routes", prompt)
            self.assertIn("cover every dimension whose `requested` value is true", prompt)
            self.assertIn("For requested dimensions with `evidence_found=true`", prompt)
            self.assertIn("For requested dimensions with `evidence_found=false`", prompt)
            self.assertIn("isotopic enrichment", prompt)
            self.assertIn("scalability, cost-effectiveness, and safety", prompt)
        finally:
            cfg.COMPARISON_JSON_ENABLED = old

    def test_method_key_step_guard_is_ab_switch(self):
        old = cfg.METHOD_KEY_STEP_GUARD_ENABLED
        try:
            cfg.METHOD_KEY_STEP_GUARD_ENABLED = False
            prompt = build_synthesis_prompt("kb", "What are the key steps?", "", "strict", "en")
            self.assertNotIn("METHOD KEY STEPS", prompt)

            cfg.METHOD_KEY_STEP_GUARD_ENABLED = True
            prompt = build_synthesis_prompt("kb", "What are the key steps?", "", "strict", "en")
            self.assertIn("METHOD KEY STEPS", prompt)
            self.assertIn("process-defining level", prompt)
            self.assertIn("Do not promote starting-material preparation/protection", prompt)
            self.assertIn("Do not write exclusion notes", prompt)
            self.assertIn("do not add a separate note about omitted precursor/preparation steps", prompt)
            self.assertIn("acidic hydrolysis/deprotection of the auxiliary", prompt)
            self.assertIn("chymotrypsin-catalysed enzymatic hydrolysis", prompt)
            self.assertIn("Schoellkopf-type auxiliary", prompt)
            self.assertIn("4-bromomethylbenzeneboronate", prompt)
            self.assertIn("~74% e.e.", prompt)
            self.assertIn("up to 100% e.e.", prompt)
            self.assertIn("do not describe 72% yield or 86% e.e. as the core process performance", prompt)
            self.assertIn("short atomic sentences", prompt)
            self.assertIn("Do not add post-step rationale/caveat sentences", prompt)
            self.assertIn("not as a key step", prompt)
        finally:
            cfg.METHOD_KEY_STEP_GUARD_ENABLED = old


class TestBuildFallbackPrompt(unittest.TestCase):
    def test_contains_question(self):
        q = "UNIQUE_FALLBACK_QUESTION_XYZ"
        prompt = build_fallback_prompt(q, "")
        self.assertIn(q, prompt)

    def test_contains_memory_section(self):
        mem = "UNIQUE_MEMORY_FALLBACK_CONTENT"
        prompt = build_fallback_prompt("q", mem)
        self.assertIn(mem, prompt)


# ══════════════════════════════════════════════════════════════════════════════
# query_translation (mock requests.post)
# ══════════════════════════════════════════════════════════════════════════════
class TestTranslateToTraditionalChinese(unittest.TestCase):
    @patch("requests.post")
    def test_returns_translated_text_on_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {"response": "繁體中文翻譯結果"}
        mock_post.return_value = mock_resp
        result = translate_to_traditional_chinese("English text to translate")
        self.assertEqual(result, "繁體中文翻譯結果")

    @patch("requests.post")
    def test_term_fidelity_preserves_route_defining_phrase(self, mock_post):
        old = cfg.TERM_FIDELITY_GUARD_ENABLED
        try:
            cfg.TERM_FIDELITY_GUARD_ENABLED = True
            mock_resp = MagicMock()
            mock_resp.ok = True
            mock_resp.json.return_value = {"response": "繁體中文翻譯結果"}
            mock_post.return_value = mock_resp
            translate_to_traditional_chinese("x")
            prompt = mock_post.call_args.kwargs["json"]["prompt"]
            self.assertIn("chymotrypsin-catalysed enzymatic hydrolysis", prompt)
            self.assertIn("Preserve route-defining phrases", prompt)
        finally:
            cfg.TERM_FIDELITY_GUARD_ENABLED = old

    @patch("requests.post")
    def test_fold_change_rule_is_unambiguous_ab_switch(self, mock_post):
        old = cfg.TRANSLATION_FOLD_CHANGE_GUARD_ENABLED
        try:
            mock_resp = MagicMock()
            mock_resp.ok = True
            mock_resp.json.return_value = {"response": "繁體中文翻譯結果"}
            mock_post.return_value = mock_resp

            cfg.TRANSLATION_FOLD_CHANGE_GUARD_ENABLED = True
            translate_to_traditional_chinese("A three-fold IC50 decrease was observed.")
            prompt = mock_post.call_args.kwargs["json"]["prompt"]
            self.assertIn("降至原來約三分之一", prompt)
            self.assertIn("never as the ambiguous '降低 N 倍'", prompt)

            cfg.TRANSLATION_FOLD_CHANGE_GUARD_ENABLED = False
            translate_to_traditional_chinese("A three-fold IC50 decrease was observed.")
            prompt = mock_post.call_args.kwargs["json"]["prompt"]
            self.assertNotIn("降至原來約三分之一", prompt)
        finally:
            cfg.TRANSLATION_FOLD_CHANGE_GUARD_ENABLED = old

    @patch("requests.post")
    def test_returns_original_on_connection_error(self, mock_post):
        mock_post.side_effect = Exception("connection refused")
        original = "English fallback text"
        result = translate_to_traditional_chinese(original)
        self.assertEqual(result, original)

    @patch("requests.post")
    def test_returns_original_when_response_empty(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {"response": ""}
        mock_post.return_value = mock_resp
        original = "original unchanged text"
        result = translate_to_traditional_chinese(original)
        self.assertEqual(result, original)


# ══════════════════════════════════════════════════════════════════════════════
# answerability (mock requests.post)
# ══════════════════════════════════════════════════════════════════════════════
class TestAnswerability(unittest.TestCase):
    @patch("rag.answerability.requests.post")
    def test_preserves_full_recovery_reason_and_requires_each_value_arm(self, mock_post):
        reason = "The facts omit the requested value for the preincubation-only comparison arm. " * 4
        response = MagicMock()
        response.json.return_value = {"response": f"VERDICT: PARTIAL\nREASON: {reason}"}
        mock_post.return_value = response

        result = assess_answerability("Give the reported values.", "[Fact 1] One value.")

        self.assertEqual(result, {"verdict": "PARTIAL", "reason": reason.strip()})
        system = mock_post.call_args.kwargs["json"]["system"]
        self.assertIn("each requested comparison arm", system)

    @patch("rag.answerability.requests.post")
    def test_deterministic_guard_downgrades_unquantified_comparison_arm(self, mock_post):
        response = MagicMock()
        response.json.return_value = {
            "response": "VERDICT: ANSWERABLE\nREASON: the combined value is present"
        }
        mock_post.return_value = response
        knowledge_base = (
            "[Fact 1] The combined IC50 was 34.2 nM, greatly lower than that of "
            "preincubation alone (Source: 1-s2.0-S1347861320300633-main)"
        )

        with patch.object(cfg, "PARTIAL_RECOVERY_DETERMINISTIC_GUARDS_ENABLED", True):
            result = assess_answerability("Give the reported values.", knowledge_base)

        self.assertEqual(result["verdict"], "PARTIAL")
        self.assertIn("preincubation alone", result["reason"])

    @patch("rag.answerability.requests.post")
    def test_deterministic_guard_accepts_semantic_standalone_value(self, mock_post):
        response = MagicMock()
        response.json.return_value = {
            "response": "VERDICT: ANSWERABLE\nREASON: all requested values are present"
        }
        mock_post.return_value = response
        knowledge_base = (
            "[Fact 1] The combined IC50 was lower than that of preincubation alone. "
            "(Source: PaperA)\n"
            "[Fact 2] The IC50 for the preincubation inhibitory effects was 193 ± 50 nM. "
            "(Source: PaperA)"
        )

        with patch.object(cfg, "PARTIAL_RECOVERY_DETERMINISTIC_GUARDS_ENABLED", True):
            result = assess_answerability("Give the reported values.", knowledge_base)

        self.assertEqual(result["verdict"], "ANSWERABLE")

    def test_literal_recovery_facts_keep_query_relevant_measurement(self):
        evidence = [(
            "【PaperA】",
            "Retrieved evidence snippets:\n"
            "[Snippet 1] All three synthetic impurities were detectable at concentrations of "
            "0.5 \x03g/ml (or 0.1% of the BPA nominal working concentration).\n"
            "[Snippet 2] Raw BPA remained stable at 55 C for 6 months and 40 C for 12 months.\n"
            "[Snippet 3] Sample preparation and degradation assays: BPA-mannitol at pH 8 "
            "was incubated in the dark at 4, 25, and 40 C for several months.\n"
            "[Snippet 4] BPA-mannitol formed about 1% phenylalanine at 40 C over 6 months.",
        )]

        facts = pipeline_module._literal_recovery_facts(
            evidence,
            "Which BPA impurities form, and under which storage conditions?",
        )

        self.assertIn("0.5 µg/ml", facts)
        self.assertIn("0.1%", facts)
        self.assertIn("pH 8", facts)
        self.assertIn("incubated in the dark", facts)
        self.assertIn("1% phenylalanine", facts)
        self.assertIn("Source: PaperA", facts)

    def test_literal_recovery_keeps_result_relation_and_ignores_footnote_digits(self):
        evidence = [
            (
                "【PaperA】",
                "Retrieved evidence snippets:\n"
                "[Snippet 1] Pioneering reports prompted a new mechanistic hypothesis.17\n"
                "[Snippet 2] Preincubation inhibitory effects of JPH203 on LAT1 function. "
                "Based on the results, the IC50 value was determined as 193 ± 50 nM.\n"
                "[Snippet 3] These results indicate that JPH203 exerts preincubation inhibitory "
                "effects in a concentration- and time-dependent manner.",
            ),
            (
                "【PaperB】",
                "Retrieved evidence snippets:\n"
                "[Snippet 1] PCL has less acidic degradation products than polylactide.",
            ),
        ]

        facts = pipeline_module._literal_recovery_facts(
            evidence,
            "How does preincubation change inhibition? Give the reported values.",
        )

        self.assertIn("193 ± 50 nM", facts)
        self.assertNotIn("LAT1 function. Based on the results", facts)
        self.assertIn("concentration- and time-dependent", facts)
        self.assertNotIn("Pioneering reports", facts)
        self.assertNotIn("PCL", facts)
        contract = pipeline_module.bind_fact_list(
            facts,
            pipeline_module.build_evidence_catalog([
                {"source": "PaperA", "text": evidence[0][1]},
            ]),
        )
        self.assertTrue(any("193 ± 50 nM" in fact["claim"] for fact in contract["facts"]))

    def test_literal_completeness_appends_only_omitted_direct_facts(self):
        literal_facts = (
            "[Fact 1] Raw BPA shows no detectable degradation at 55 C for 6 months. "
            "(Source: PaperA)\n\n"
            "[Fact 2] All three impurities are detectable at 0.5 micrograms per millilitre, "
            "equivalent to 0.1%. (Source: PaperA)\n\n"
            "[Fact 3] BPA degrades to tyrosine under alkaline and oxidative conditions extremely "
            "rapidly. (Source: PaperA)"
        )
        answer = "Raw BPA shows no detectable degradation at 55 C for 6 months [Source: PaperA]."

        fixed = pipeline_module._append_missing_literal_facts(answer, literal_facts)

        self.assertEqual(fixed.count("Raw BPA shows no detectable degradation"), 1)
        self.assertIn("0.5 micrograms per millilitre", fixed)
        self.assertIn("alkaline and oxidative conditions", fixed)

    def test_literal_recovery_removes_chart_axis_without_losing_results(self):
        evidence = [(
            "【PaperA】",
            "Retrieved evidence snippets:\n"
            "[Snippet 1] BPA degradation to tyrosine was observed under alkali and oxidative "
            "conditions and occurred extremely rapidly. BPA/mannitol lyophilised drug product "
            "showed a slow, temperature dependent degradation to phenylalanine, generating "
            "-0.004 -0.002 0 0.002 0.004 0.006 0.008 0.01 5 10 15 20 25 mAU. "
            "Mechanistic pathway showing degradation to phenylalanine. approximately 1% of "
            "phenylalanine at 40 C over 6 months.",
        )]

        facts = pipeline_module._literal_recovery_facts(
            evidence,
            "Which degradation products form, and under which storage conditions?",
        )

        self.assertIn("alkali and oxidative conditions", facts)
        self.assertIn("slow, temperature dependent degradation", facts)
        self.assertIn("approximately 1%", facts)
        self.assertNotIn("-0.004", facts)


# ══════════════════════════════════════════════════════════════════════════════
# query_pipeline — integration (all external calls mocked)
# ══════════════════════════════════════════════════════════════════════════════
def _setup_cfg(cfg_mock):
    cfg_mock.REVIEW_MODE = False
    cfg_mock.SYNTHESIS_ENABLED = False
    cfg_mock.VERIFY_ENABLED = False
    cfg_mock.CITATION_GROUNDING_ENABLED = False
    cfg_mock.ANSWERABILITY_GATE_ENABLED = False
    cfg_mock.PARTIAL_ANSWER_RECOVERY_ENABLED = False
    cfg_mock.PARTIAL_RECOVERY_DETERMINISTIC_GUARDS_ENABLED = False
    cfg_mock.PARTIAL_RECOVERY_EVIDENCE_SNIPPETS_PER_TASK = 4
    cfg_mock.STAGE2_LLM_SUBANSWERS_ENABLED = False
    cfg_mock.STAGE4_ANSWER_VALIDATION_ENABLED = False
    cfg_mock.STAGE4_ANSWER_REWRITE_RETRIES = 1
    cfg_mock.FACT_RELATION_ATOMICITY_GUARD_ENABLED = False
    cfg_mock.STRUCTURED_FACT_CONTRACT_ENABLED = False
    cfg_mock.COMPARISON_JSON_DIRECT_RENDER_ENABLED = False
    cfg_mock.METHOD_FACT_LIST_DIRECT_RENDER_ENABLED = False
    cfg_mock.EN_DRAFT_PIPELINE = False
    cfg_mock.FINAL_TRANSLATION_ENABLED = True
    cfg_mock.REASONING_MODE = "strict"
    cfg_mock.SUBQUERY_MAX_WORKERS = 1


class TestStage4RelationAtomicity(unittest.TestCase):
    def test_splits_stability_result_from_forced_degradation_protocol(self):
        old = cfg.FACT_RELATION_ATOMICITY_GUARD_ENABLED
        try:
            cfg.FACT_RELATION_ATOMICITY_GUARD_ENABLED = True
            answer = (
                "BPA is stable in acidic and FeCl3 solutions [Paper A], including forced "
                "degradation tests performed using 100 mM HCl at 55 C for 24 h [Paper A]."
            )
            fixed = pipeline_module._separate_stability_protocol_clause(answer)
            self.assertIn("solutions [Paper A]. Forced degradation tests were performed", fixed)

            legitimate = (
                "The study used several analyses, including forced degradation tests "
                "performed at 55 C."
            )
            self.assertEqual(
                pipeline_module._separate_stability_protocol_clause(legitimate),
                legitimate,
            )
        finally:
            cfg.FACT_RELATION_ATOMICITY_GUARD_ENABLED = old


class TestExecuteStructuredQuery(unittest.TestCase):
    def test_fact_contract_renderer_rejects_cross_sentence_scope(self):
        answer, claims, audit = pipeline_module._render_validated_fact_contract(
            "\n".join([
                "[Fact 1] BPA in 100 mM HCl was incubated at 55 C for 24 h. (Source: PaperA)",
                "[Fact 2] A BPA solution in 6 mM H2O2 was prepared immediately before analysis. (Source: PaperA)",
                "[Fact 3] BPA in 6 mM H2O2 was incubated at 55 C for 24 h. (Source: PaperA)",
            ]),
            [
                "【PaperA】\n[Snippet 1] BPA in 100 mM HCl was incubated at 55 C for 24 h. "
                "[Snippet 2] A BPA solution in 6 mM H2O2 was prepared immediately before analysis."
            ],
        )

        self.assertIn("100 mM HCl", answer)
        self.assertIn("prepared immediately before analysis", answer)
        self.assertNotIn("H2O2 was incubated at 55 C", answer)
        self.assertEqual(len(claims), 2)
        self.assertEqual(len(audit["rejected"]), 1)

    def test_partial_recovery_accepts_only_answerable_expanded_facts(self):
        artifacts = {}
        recovered_kb = (
            "[Fact 1] Partial fact. (Source: PaperA)\n\n"
            "[Fact 2] Complete storage outcome. (Source: PaperA)"
        )
        with (
            patch.object(cfg, "PARTIAL_ANSWER_RECOVERY_ENABLED", True),
            patch.object(cfg, "SYNTHESIS_ENABLED", True),
            patch.object(cfg, "STAGE2_LLM_SUBANSWERS_ENABLED", False),
            patch.object(cfg, "PARTIAL_RECOVERY_EVIDENCE_SNIPPETS_PER_TASK", 4),
            patch.object(
                pipeline_module,
                "run_subqueries_parallel",
                return_value=[("【PaperA】", "expanded evidence")],
            ) as mock_run,
            patch.object(
                pipeline_module._synthesizer,
                "synthesize",
                return_value=recovered_kb,
            ) as mock_synthesize,
            patch(
                "rag.answerability.assess_answerability",
                return_value={"verdict": "ANSWERABLE", "reason": "complete"},
            ),
        ):
            result = pipeline_module._attempt_partial_recovery(
                "What storage outcome was reported?",
                [(0, "【PaperA】", MagicMock(), "storage outcome")],
                {},
                ["【PaperA】\ninitial evidence"],
                "[Fact 1] Partial fact. (Source: PaperA)",
                {"verdict": "PARTIAL", "reason": "missing storage outcome"},
                on_status=[].append,
                on_artifact=artifacts.__setitem__,
            )

        self.assertTrue(result["attempted"])
        self.assertTrue(result["accepted"])
        self.assertEqual(result["knowledge_base"].count("Partial fact"), 1)
        self.assertIn("Complete storage outcome", result["knowledge_base"])
        self.assertEqual(mock_run.call_args.kwargs["evidence_snippets_per_task"], 4)
        self.assertEqual(mock_synthesize.call_args.kwargs["recovery_hint"], "missing storage outcome")
        self.assertIn("stage2_recovery_evidence", artifacts)
        self.assertIn("Complete storage outcome", artifacts["stage3_recovery_knowledge_base"])
        self.assertIn("partial_recovery_assessment", artifacts)

    def test_partial_recovery_keeps_original_when_retry_is_still_partial(self):
        original_kb = "[Fact 1] Partial fact. (Source: PaperA)"
        with (
            patch.object(cfg, "PARTIAL_ANSWER_RECOVERY_ENABLED", True),
            patch.object(cfg, "SYNTHESIS_ENABLED", True),
            patch.object(cfg, "STAGE2_LLM_SUBANSWERS_ENABLED", False),
            patch.object(
                pipeline_module,
                "run_subqueries_parallel",
                return_value=[("【PaperA】", "expanded evidence")],
            ),
            patch.object(
                pipeline_module._synthesizer,
                "synthesize",
                return_value="[Fact 1] Still partial. (Source: PaperA)",
            ),
            patch(
                "rag.answerability.assess_answerability",
                return_value={"verdict": "PARTIAL", "reason": "still incomplete"},
            ),
        ):
            result = pipeline_module._attempt_partial_recovery(
                "What storage outcome was reported?",
                [(0, "【PaperA】", MagicMock(), "storage outcome")],
                {},
                ["【PaperA】\ninitial evidence"],
                original_kb,
                {"verdict": "PARTIAL", "reason": "missing storage outcome"},
                on_status=[].append,
            )

        self.assertTrue(result["attempted"])
        self.assertFalse(result["accepted"])
        self.assertEqual(result["knowledge_base"], original_kb)

    def test_method_fact_renderer_keeps_core_condition_and_excludes_precursors(self):
        kb = """
[Fact 1] Optically pure L-BPA was synthesized by a hybrid process. (Source: bbb0683)
[Fact 2] Lithiated auxiliary was reacted with bromide to yield an adduct in a 74% e.e. (Source: bbb0683)
[Fact 3] Treatment with hydrochloric acid gave L-BPA methyl ester. (Source: bbb0683)
[Fact 4] L-BPA methyl ester was hydrolyzed with chymotrypsin to furnish optically pure L-BPA. (Source: bbb0683)
[Fact 5] The starting material was prepared from commercially available 4-bromotoluene. (Source: bbb0683)
[Fact 6] The dihydroxyboryl group was protected as a cyclic borinate in a 79% yield. (Source: bbb0683)
[Fact 7] Enantioselective alkylation was conducted in THF at -78°C. (Source: bbb0683)
"""
        answer, claims, audit = pipeline_module._render_method_fact_list(
            kb,
            "What hybrid process is used for the synthesis, and what are its key steps?",
            [{"paper": "bbb0683", "sub_q": "Describe reagents and experimental conditions."}],
        )

        self.assertIn("THF at -78°C", answer)
        self.assertIn("chymotrypsin", answer)
        self.assertNotIn("commercially available", answer)
        self.assertNotIn("79%", answer)
        self.assertIn("conditions", audit["requirements"])
        self.assertEqual(audit["missing_requirements"], [])
        self.assertTrue(all(claim.startswith("- ") and "[bbb0683]" in claim for claim in claims))

    def test_method_fact_renderer_ignores_retrieval_only_condition_facet(self):
        answer, claims, audit = pipeline_module._render_method_fact_list(
            "[Fact 1] Compound A was synthesized by a method that gave product B. (Source: PaperA)",
            "What method is used to synthesize Compound A?",
            [{"paper": "PaperA", "sub_q": "Report the experimental conditions and temperature."}],
        )

        self.assertIn("Compound A was synthesized", answer)
        self.assertTrue(claims)
        self.assertEqual(audit["missing_requirements"], [])

    def test_stage4_validator_flags_bad_comparison_answer(self):
        old = cfg.STAGE4_ANSWER_VALIDATION_ENABLED
        try:
            cfg.STAGE4_ANSWER_VALIDATION_ENABLED = True
            kb = """
            {"comparison_json":{
              "direct_routes":[{"source":"bbb0683","route_phrase":"enantioselective alkylation followed by chymotrypsin-catalysed enzymatic hydrolysis"}],
              "review_comparison_sources":[{"source":"CMDC-20-e202500059"}],
              "dimensions":{
                "isotopic_enrichment":{"requested":true,"evidence_found":true},
                "scalability":{"requested":true,"evidence_found":true},
                "cost_effectiveness":{"requested":true,"evidence_found":true}
              }
            }}
            """
            issues = pipeline_module._stage4_answer_validation_issues(
                "No relevant query results or paper data were provided.",
                kb,
                "Compare routes for isotopic enrichment, scalability, and cost-effectiveness.",
            )
            self.assertIn("False no-data answer", issues)
            self.assertIn("Missing direct route phrase", issues)
            self.assertIn("Missing review/comparison source", issues)
        finally:
            cfg.STAGE4_ANSWER_VALIDATION_ENABLED = old

    def test_stage4_validator_flags_dense_background_comparison(self):
        old = cfg.STAGE4_ANSWER_VALIDATION_ENABLED
        try:
            cfg.STAGE4_ANSWER_VALIDATION_ENABLED = True
            kb = """
            {"comparison_json":{
              "source_roles":[
                {"source":"bbb0683","role":"route"},
                {"source":"CMDC-20-e202500059","role":"review/comparison source"},
                {"source":"water-soluble-BPA-derivatives","role":"background"}
              ],
              "direct_routes":[{"source":"bbb0683","route_phrase":"enantioselective alkylation followed by chymotrypsin-catalysed enzymatic hydrolysis"}],
              "review_comparison_sources":[{"source":"CMDC-20-e202500059"}],
              "dimensions":{
                "isotopic_enrichment":{"requested":true,"evidence_found":true},
                "scalability":{"requested":true,"evidence_found":true},
                "cost_effectiveness":{"requested":true,"evidence_found":true}
              }
            }}
            """
            answer = (
                "Comparison scaffold: bbb0683 uses enantioselective alkylation followed by "
                "chymotrypsin-catalysed enzymatic hydrolysis [bbb0683]. "
                "The hybrid process uses a long route description and CMDC compares protecting groups "
                "for scalability and cost-effectiveness [CMDC-20-e202500059], while 10B evidence appears "
                "in water-soluble-BPA-derivatives [water-soluble-BPA-derivatives] with many details that make "
                "this sentence deliberately long enough to be a dense multi-source claim rather than an atomic "
                "source-backed comparison sentence. Central trade-off: optical purity versus scalability and cost."
            )
            issues = pipeline_module._stage4_answer_validation_issues(
                answer,
                kb,
                "Compare routes for isotopic enrichment, scalability, and cost-effectiveness.",
            )
            self.assertIn("Background source cited", issues)
            self.assertIn("Over-dense multi-source sentence", issues)
            self.assertIn("Missing high-purity framing", issues)
        finally:
            cfg.STAGE4_ANSWER_VALIDATION_ENABLED = old

    def test_stage4_empty_fallback_formats_comparison_json(self):
        kb = """
        {"comparison_json":{
          "source_roles":[
            {"source":"bbb0683","role":"route"},
            {"source":"CMDC-20-e202500059","role":"review/comparison source"},
            {"source":"FormulationA","role":"background"}
          ],
          "direct_routes":[{"source":"bbb0683","route_phrase":"enantioselective alkylation followed by chymotrypsin-catalysed enzymatic hydrolysis","outcome":"optically pure L-BPA at high e.e."}],
          "review_comparison_sources":[{"source":"CMDC-20-e202500059","claim":"L-BPA synthesis has been approached through multiple routes"}],
          "dimensions":{
            "isotopic_enrichment":{"requested":true,"evidence_found":true,"evidence":[{"source":"CMDC-20-e202500059","claim":"10B-enriched material is required."}]},
            "scalability":{"requested":true,"evidence_found":true,"evidence":[
              {"source":"CMDC-20-e202500059","claim":"The use of any oxidant on scale is inherently a process safety risk."},
              {"source":"CMDC-20-e202500059","claim":"Gram-scale deprotection can leave ester residue."},
              {"source":"bbb0683","claim":"The hybrid route uses few reaction steps."}
            ]},
            "safety":{"requested":false,"evidence_found":true,"evidence":[
              {"source":"CMDC-20-e202500059","claim":"The review compares route safety."},
              {"source":"FormulationA","claim":"The formulation is safe."}
            ]}
          },
          "central_tradeoff":{"claim":"Optically pure, 10B-enriched material must be balanced against scalability and cost-effectiveness.","sources":["CMDC-20-e202500059"]}
        }}
        """
        answer = pipeline_module._stage4_empty_answer_fallback(kb, atomic_only=True)
        self.assertIn("Comparison scaffold", answer)
        self.assertIn("bbb0683", answer)
        self.assertIn("chymotrypsin-catalysed enzymatic hydrolysis", answer)
        self.assertIn("optically pure L-BPA at high e.e", answer)
        self.assertIn("CMDC-20-e202500059", answer)
        self.assertIn(
            "reports that L-BPA synthesis has been approached through multiple routes",
            answer,
        )
        self.assertIn("oxidant on scale", answer)
        self.assertIn("Gram-scale deprotection", answer)
        self.assertIn("The hybrid route uses few reaction steps", answer)
        self.assertIn("The review compares route safety", answer)
        self.assertNotIn("FormulationA", answer)
        self.assertNotIn("[CMDC-20-e202500059, bbb0683]", answer)
        self.assertIn("Central trade-off", answer)
        self.assertIn("High-purity/isotopic enrichment", answer)
        self.assertNotIn("must be balanced against", answer)
        self.assertTrue(all(line.count("[") <= 1 for line in answer.splitlines()))

        old = cfg.STAGE4_ANSWER_VALIDATION_ENABLED
        try:
            cfg.STAGE4_ANSWER_VALIDATION_ENABLED = True
            self.assertEqual(
                pipeline_module._stage4_answer_validation_issues(
                    answer,
                    kb,
                    "Compare routes for isotopic enrichment and scalability.",
                ),
                "",
            )
        finally:
            cfg.STAGE4_ANSWER_VALIDATION_ENABLED = old

    def test_stage4_renderer_uses_strategy_and_mechanism_for_non_synthesis_comparison(self):
        kb = json.dumps({"comparison_json": {
            "target_compound": "LAT1",
            "source_roles": [
                {"source": "InhibitorA", "role": "route"},
                {
                    "source": "StructureA",
                    "role": "mechanism",
                    "claim": "JPH203 binds within the traditional LAT1 substrate-binding pocket",
                    "evidence": (
                        "The α-amino group and α-carboxyl group of the head... "
                        "the chloride atom of JPH203 forms a halogen bond with Tyr259"
                    ),
                },
                {
                    "source": "KineticsA",
                    "role": "mechanism",
                    "claim": "JPH203 exerts co-incubation and preincubation inhibitory effects",
                    "evidence": (
                        "The preincubation effect enhances the co-incubation inhibitory effects"
                    ),
                },
                {
                    "source": "TheoryA",
                    "role": "mechanism",
                    "claim": "Possible mechanisms may involve transient membrane localization",
                    "evidence": "Possible mechanisms may involve transient membrane localization",
                },
            ],
            "direct_routes": [{
                "source": "InhibitorA",
                "route_phrase": "competitive JPH203 inhibition",
                "outcome": "blocked LAT1-mediated amino-acid transport",
            }],
            "supporting_mechanisms": [{
                "source": "StructureA",
                "claim": "JPH203 forms a halogen bond with Tyr259",
                "evidence": "JPH203 forms a halogen bond with Tyr259",
            }, {
                "source": "StructureA",
                "claim": "JPH203 causes a 4.34 degree shift in TM1",
                "evidence": "JPH203 causes a 4.34 degree shift in TM1",
            }, {
                "source": "KineticsA",
                "claim": (
                    "JPH203 inhibition involves both co-incubation and preincubation effects"
                ),
                "evidence": "JPH203 inhibition involves co-incubation and preincubation effects",
            }, {
                "source": "TheoryA",
                "claim": "Hypothesized mechanisms may involve transient membrane localization",
                "evidence": "Hypothesized mechanisms may involve transient membrane localization",
            }],
            "review_comparison_sources": [],
            "dimensions": {},
            "central_tradeoff": {"claim": "The mechanisms differ.", "sources": ["InhibitorA"]},
        }})
        answer = pipeline_module._stage4_empty_answer_fallback(
            kb,
            atomic_only=True,
            question="How do therapeutic strategies targeting LAT1 differ in mechanism?",
        )
        self.assertIn("- Strategy:", answer)
        self.assertIn("- Mechanism:", answer)
        self.assertIn("traditional LAT1 substrate-binding pocket", answer)
        self.assertIn("halogen bond with Tyr259", answer)
        self.assertNotIn("of the head", answer)
        self.assertNotIn("4.34 degree shift", answer)
        self.assertNotIn("TheoryA", answer)
        self.assertEqual(
            sum(line.startswith("- Mechanism: `StructureA`") for line in answer.splitlines()),
            2,
        )
        self.assertEqual(
            sum(line.startswith("- Mechanism: `KineticsA`") for line in answer.splitlines()),
            1,
        )
        self.assertNotIn("synthesis of LAT1", answer)

        old = cfg.STAGE4_ANSWER_VALIDATION_ENABLED
        try:
            cfg.STAGE4_ANSWER_VALIDATION_ENABLED = True
            self.assertEqual(
                pipeline_module._stage4_answer_validation_issues(
                    answer,
                    kb,
                    "How do therapeutic strategies targeting LAT1 differ in mechanism?",
                ),
                "",
            )
        finally:
            cfg.STAGE4_ANSWER_VALIDATION_ENABLED = old

    def test_stage4_renderer_splits_semicolon_dimension_claims(self):
        kb = json.dumps({"comparison_json": {
            "source_roles": [{"source": "ReviewA", "role": "review/comparison source"}],
            "direct_routes": [],
            "review_comparison_sources": [{"source": "ReviewA", "dimensions": ["cost-effectiveness"]}],
            "dimensions": {
                "cost_effectiveness": {
                    "requested": True,
                    "evidence_found": True,
                    "evidence": [{
                        "source": "ReviewA",
                        "claim": (
                            "The major cost comes from isotope starting material; "
                            "10B costs over 1000 times normal boric acid."
                        ),
                    }],
                },
            },
            "central_tradeoff": {"claim": "10B is expensive.", "sources": ["ReviewA"]},
        }})
        answer = pipeline_module._stage4_empty_answer_fallback(
            kb,
            atomic_only=True,
            question="Compare cost-effectiveness.",
        )

        self.assertIn("isotope starting material [ReviewA].", answer)
        self.assertIn("10B costs over 1000 times normal boric acid [ReviewA].", answer)

    def test_stage4_renderer_uses_atomic_review_evidence(self):
        kb = json.dumps({"comparison_json": {
            "source_roles": [{"source": "ReviewA", "role": "review/comparison source"}],
            "direct_routes": [],
            "review_comparison_sources": [{
                "source": "ReviewA",
                "claim": "Isotope, cost, and safety evidence form one combined limitation.",
                "evidence": (
                    "L-BPA synthesis has been approached through multiple routes, reflecting "
                    "the challenge of producing enriched material... isotope feedstock is costly"
                ),
            }],
            "dimensions": {},
            "central_tradeoff": {"claim": "Route constraints differ.", "sources": ["ReviewA"]},
        }})
        answer = pipeline_module._stage4_empty_answer_fallback(
            kb,
            atomic_only=True,
            question="Compare L-BPA synthesis routes.",
        )

        self.assertIn("approached through multiple routes", answer)
        self.assertNotIn("one combined limitation", answer)
        self.assertNotIn("feedstock is costly", answer)

    def test_stage4_renderer_does_not_render_snippet_locator_as_evidence(self):
        kb = json.dumps({"comparison_json": {
            "source_roles": [{"source": "ReviewA", "role": "review/comparison source"}],
            "direct_routes": [],
            "review_comparison_sources": [{
                "source": "ReviewA",
                "claim": "The review compares multiple synthetic approaches.",
                "evidence": "Snippet 2, 3, 4",
            }],
            "dimensions": {},
            "central_tradeoff": {"claim": "Route constraints differ.", "sources": ["ReviewA"]},
        }})
        answer = pipeline_module._stage4_empty_answer_fallback(
            kb,
            atomic_only=True,
            question="Compare synthetic routes.",
        )

        self.assertIn("compares multiple synthetic approaches", answer)
        self.assertNotIn("reports that Snippet 2, 3, 4", answer)

    def test_stage4_direct_render_is_concise_for_high_level_question(self):
        kb = """
        {"comparison_json":{
          "target_compound":"4-borono-L-phenylalanine (L-BPA)",
          "source_roles":[
            {"source":"bbb0683","role":"route"},
            {"source":"CMDC-20-e202500059","role":"review/comparison source"}
          ],
          "direct_routes":[{
            "source":"bbb0683",
            "route_phrase":"enantioselective alkylation followed by chymotrypsin-catalysed enzymatic hydrolysis",
            "outcome":"74% e.e. for adduct 4; optically pure L-BPA (100% optical purity) with 79% yield"
          }],
          "review_comparison_sources":[{
            "source":"CMDC-20-e202500059",
            "claim":"L-BPA synthesis has been approached through multiple routes.",
            "dimensions":["isotopic enrichment","scalability","cost-effectiveness","safety"]
          }],
          "dimensions":{
            "isotopic_enrichment":{"requested":true,"evidence_found":true,"evidence":[
              {"source":"CMDC-20-e202500059","claim":"Producing high-purity, isotopically enriched material is a primary challenge."}
            ]},
            "scalability":{"requested":true,"evidence_found":true,"evidence":[
              {"source":"CMDC-20-e202500059","claim":"The use of any oxidant on scale is inherently a process safety risk."},
              {"source":"bbb0683","claim":"The hybrid method has an advantage in ease of workup and few reaction steps."}
            ]},
            "cost_effectiveness":{"requested":true,"evidence_found":true,"evidence":[
              {"source":"CMDC-20-e202500059","claim":"The major cost comes from the isotope starting material."}
            ]},
            "safety":{"requested":false,"evidence_found":true,"evidence":[
              {"source":"CMDC-20-e202500059","claim":"NaIO4 toxicity includes specific LD50 values."}
            ]}
          },
          "central_tradeoff":{"claim":"High purity must be balanced with scale and cost.","sources":["CMDC-20-e202500059"]}
        }}
        """
        answer = pipeline_module._stage4_empty_answer_fallback(
            kb,
            atomic_only=True,
            question="Compare routes focusing on isotopic enrichment, scalability, and cost-effectiveness.",
        )
        self.assertIn("yielding optically pure L-BPA at high e.e.", answer)
        self.assertIn(
            "has been approached through multiple routes [CMDC-20-e202500059].",
            answer,
        )
        self.assertIn(
            "Review dimensions: The review highlights limitations of each method regarding "
            "scalability, cost-effectiveness, and safety [CMDC-20-e202500059].",
            answer,
        )
        self.assertIn(
            "Central trade-off (high-purity/isotopic enrichment versus scalability and "
            "cost-effectiveness):",
            answer,
        )
        self.assertNotIn("multiple routes and compares", answer)
        self.assertIn("ease of workup and few reaction steps", answer)
        self.assertIn("The major cost comes from the isotope starting material", answer)
        self.assertNotIn("74%", answer)
        self.assertNotIn("79%", answer)
        self.assertNotIn("oxidant on scale", answer)
        self.assertNotIn("LD50", answer)

    @patch("rag.query_pipeline.translate_to_traditional_chinese")
    @patch("rag.query_pipeline.run_grounding_check")
    @patch("rag.query_pipeline.run_subqueries_parallel")
    @patch("rag.query_pipeline.build_subquery_tasks")
    @patch("rag.query_pipeline.plan_sub_questions")
    @patch("rag.query_pipeline.detect_target_paper")
    @patch("rag.query_pipeline.cfg")
    @patch("rag.query_pipeline.Settings")
    def test_atomic_comparison_skips_stage4_llm(
        self, mock_settings, mock_cfg,
        mock_detect, mock_plan, mock_build, mock_run,
        mock_grounding, mock_translate,
    ):
        _setup_cfg(mock_cfg)
        mock_cfg.COMPARISON_JSON_DIRECT_RENDER_ENABLED = True
        mock_cfg.CITATION_GROUNDING_ENABLED = True
        mock_cfg.EN_DRAFT_PIPELINE = True
        mock_cfg.FINAL_TRANSLATION_ENABLED = False
        mock_detect.return_value = "paper_a"
        mock_plan.return_value = [{"paper": "paper_a", "sub_q": "Q?"}]
        mock_build.return_value = ([], {})
        kb = """
        {"comparison_json":{
          "direct_routes":[{"source":"RouteA","route_phrase":"route A","outcome":"high-purity product at high e.e."}],
          "review_comparison_sources":[{"source":"ReviewA","claim":"compares route scalability and cost"}],
          "dimensions":{
            "isotopic_enrichment":{"requested":true,"evidence_found":true,
              "evidence":[{"source":"ReviewA","claim":"High-purity isotopically enriched material is required."}]},
            "scalability":{"requested":true,"evidence_found":true,
              "evidence":[{"source":"ReviewA","claim":"Route A is practical at scale."}]},
            "cost_effectiveness":{"requested":true,"evidence_found":true,
              "evidence":[{"source":"ReviewA","claim":"Enriched precursor material is expensive."}]}
          },
          "central_tradeoff":{"claim":"High purity must be balanced with scalability and cost-effectiveness.","sources":["ReviewA"]}
        }}
        """
        mock_run.return_value = [("【paper_a】", kb)]
        mock_grounding.side_effect = lambda full_text, *args, **kwargs: (full_text, "")

        result = pipeline_module.execute_structured_query(
            "Compare routes for isotopic enrichment, scalability, and cost-effectiveness.",
            {"paper_a": MagicMock()},
        )

        self.assertIn("high-purity product at high e.e", result)
        self.assertIn("Route A is practical at scale", result)
        pipeline_module._comparison_json_validation_errors.assert_called()
        mock_settings.llm.stream_complete.assert_not_called()
        claims = mock_grounding.call_args.kwargs["grounding_claims"]
        self.assertEqual(claims, split_into_sentences(result))
        mock_translate.assert_not_called()

    @patch("rag.query_pipeline.translate_to_traditional_chinese")
    @patch("rag.query_pipeline.run_grounding_check")
    @patch("rag.query_pipeline.run_subqueries_parallel")
    @patch("rag.query_pipeline.build_subquery_tasks")
    @patch("rag.query_pipeline.plan_sub_questions")
    @patch("rag.query_pipeline.detect_target_paper")
    @patch("rag.query_pipeline.cfg")
    @patch("rag.query_pipeline.Settings")
    def test_method_fact_renderer_skips_stage4_and_supplies_grounding_claims(
        self, mock_settings, mock_cfg,
        mock_detect, mock_plan, mock_build, mock_run,
        mock_grounding, mock_translate,
    ):
        _setup_cfg(mock_cfg)
        mock_cfg.SYNTHESIS_ENABLED = True
        mock_cfg.CITATION_GROUNDING_ENABLED = True
        mock_cfg.METHOD_FACT_LIST_DIRECT_RENDER_ENABLED = True
        mock_cfg.EN_DRAFT_PIPELINE = True
        mock_cfg.FINAL_TRANSLATION_ENABLED = False
        mock_detect.return_value = "bbb0683"
        mock_plan.return_value = [{
            "paper": "bbb0683",
            "sub_q": "Describe the key steps, reagents, and experimental conditions.",
        }]
        mock_build.return_value = ([], {})
        mock_run.return_value = [(
            "【bbb0683】",
            "Retrieved paper evidence contains the complete hybrid synthesis procedure.",
        )]
        kb = """
[Fact 1] Optically pure L-BPA was synthesized by a hybrid process. (Source: bbb0683)
[Fact 2] Alkylation yielded an adduct in a 74% e.e. (Source: bbb0683)
[Fact 3] Hydrochloric acid treatment gave L-BPA methyl ester. (Source: bbb0683)
[Fact 4] Chymotrypsin hydrolysis furnished optically pure L-BPA. (Source: bbb0683)
[Fact 5] The alkylation was conducted in THF at -78°C. (Source: bbb0683)
"""
        mock_grounding.side_effect = lambda full_text, *args, **kwargs: (full_text, "")

        with patch.object(pipeline_module._synthesizer, "synthesize", return_value=kb):
            result = pipeline_module.execute_structured_query(
                "What hybrid process is used for the synthesis, and what are its key steps?",
                {"bbb0683": MagicMock()},
            )

        self.assertIn("THF at -78°C", result)
        mock_settings.llm.stream_complete.assert_not_called()
        claims = mock_grounding.call_args.kwargs["grounding_claims"]
        self.assertTrue(any("THF at -78°C" in claim for claim in claims))
        mock_translate.assert_not_called()

    @patch("rag.query_pipeline.translate_to_traditional_chinese")
    @patch("rag.query_pipeline.run_grounding_check")
    @patch("rag.query_pipeline.run_subqueries_parallel")
    @patch("rag.query_pipeline.build_subquery_tasks")
    @patch("rag.query_pipeline.plan_sub_questions")
    @patch("rag.query_pipeline.detect_target_paper")
    @patch("rag.query_pipeline.cfg")
    @patch("rag.query_pipeline.Settings")
    def test_fact_contract_ab_branch_skips_stage4_and_verifier(
        self, mock_settings, mock_cfg,
        mock_detect, mock_plan, mock_build, mock_run,
        mock_grounding, mock_translate,
    ):
        _setup_cfg(mock_cfg)
        mock_cfg.SYNTHESIS_ENABLED = True
        mock_cfg.STRUCTURED_FACT_CONTRACT_ENABLED = True
        mock_cfg.EN_DRAFT_PIPELINE = True
        mock_cfg.FINAL_TRANSLATION_ENABLED = False
        mock_cfg.VERIFY_ENABLED = True
        mock_cfg.CITATION_GROUNDING_ENABLED = True
        mock_detect.return_value = "PaperA"
        mock_plan.return_value = [{"paper": "PaperA", "sub_q": "Report storage outcomes."}]
        mock_build.return_value = ([], {})
        mock_run.return_value = [(
            "【PaperA】",
            "Retrieved evidence snippets:\n"
            "[Snippet 1] Raw BPA remained stable at 55 C for 6 months.\n"
            "[Snippet 2] Drug product formed phenylalanine at 40 C over 6 months.",
        )]
        kb = (
            "[Fact 1] Raw BPA remained stable at 55 C for 6 months. (Source: PaperA)\n\n"
            "[Fact 2] Drug product formed phenylalanine at 40 C over 6 months. (Source: PaperA)"
        )
        artifacts = {}
        mock_grounding.side_effect = lambda full_text, *args, **kwargs: (full_text, "")

        with (
            patch.object(pipeline_module._synthesizer, "synthesize", return_value=kb),
            patch.object(pipeline_module, "_rewrite_stage4_if_needed") as rewrite,
            patch.object(pipeline_module._verifier, "verify_and_correct") as verify,
        ):
            result = pipeline_module.execute_structured_query(
                "What storage outcomes were reported?",
                {"PaperA": MagicMock()},
                on_artifact=artifacts.__setitem__,
            )

        self.assertIn("Raw BPA remained stable", result)
        self.assertIn("Drug product formed phenylalanine", result)
        mock_settings.llm.stream_complete.assert_not_called()
        rewrite.assert_not_called()
        verify.assert_not_called()
        self.assertIn('"schema": "fact_contract_v1"', artifacts["stage4_fact_contract"])
        self.assertEqual(len(mock_grounding.call_args.kwargs["grounding_claims"]), 2)
        mock_translate.assert_not_called()

    def test_stage4_appends_missing_isotope_cost_fact(self):
        answer = "Central trade-off: purity and isotopic enrichment versus scalability and cost-effectiveness."
        kb = (
            "[dimension_evidence]\n"
            "- cost-effectiveness: The review highlights the high cost of isotopically enriched 10B "
            "(Source: CMDC-20-e202500059)."
        )
        fixed = pipeline_module._append_missing_isotope_cost_answer(
            answer,
            kb,
            "Compare routes for isotopic enrichment, scalability, and cost-effectiveness.",
        )
        self.assertIn("high cost of isotopically enriched 10B", fixed)
        self.assertIn("CMDC-20-e202500059", fixed)

    @patch("rag.query_pipeline.translate_to_traditional_chinese")
    @patch("rag.query_pipeline.run_grounding_check")
    @patch("rag.query_pipeline.run_subqueries_parallel")
    @patch("rag.query_pipeline.build_subquery_tasks")
    @patch("rag.query_pipeline.plan_sub_questions")
    @patch("rag.query_pipeline.detect_target_paper")
    @patch("rag.query_pipeline.cfg")
    @patch("rag.query_pipeline.Settings")
    def test_rag_found_returns_llm_output(
        self, mock_settings, mock_cfg,
        mock_detect, mock_plan, mock_build, mock_run, *_
    ):
        _setup_cfg(mock_cfg)
        mock_detect.return_value = "paper_a"
        mock_plan.return_value = [{"paper": "paper_a", "sub_q": "Q?"}]
        mock_build.return_value = ([], {})
        mock_run.return_value = [(
            "【paper_a】",
            "The synthesis used Fe3O4 at 80°C for 4 hours with EDTA reagent solution.",
        )]
        chunk = MagicMock()
        chunk.delta = "Final synthesized answer text."
        mock_settings.llm.stream_complete.return_value = [chunk]

        result = pipeline_module.execute_structured_query(
            "What is the synthesis?", {"paper_a": MagicMock()}
        )
        self.assertIn("Final synthesized answer text.", result)

    @patch("rag.query_pipeline.translate_to_traditional_chinese")
    @patch("rag.query_pipeline.run_grounding_check")
    @patch("rag.query_pipeline.run_subqueries_parallel")
    @patch("rag.query_pipeline.build_subquery_tasks")
    @patch("rag.query_pipeline.plan_sub_questions")
    @patch("rag.query_pipeline.detect_target_paper")
    @patch("rag.query_pipeline.cfg")
    @patch("rag.query_pipeline.Settings")
    def test_empty_stage4_uses_stage3_facts(
        self, mock_settings, mock_cfg,
        mock_detect, mock_plan, mock_build, mock_run, *_
    ):
        _setup_cfg(mock_cfg)
        mock_detect.return_value = "paper_a"
        mock_plan.return_value = [{"paper": "paper_a", "sub_q": "Q?"}]
        mock_build.return_value = ([], {})
        mock_run.return_value = [(
            "【paper_a】",
            "retrieved evidence from PaperA",
        )]
        chunk = MagicMock()
        chunk.delta = ""
        mock_settings.llm.stream_complete.return_value = [chunk]

        result = pipeline_module.execute_structured_query(
            "What is the synthesis?", {"paper_a": MagicMock()}
        )
        self.assertIn("retrieved evidence from PaperA", result)

    @patch("rag.query_pipeline.translate_to_traditional_chinese")
    @patch("rag.query_pipeline.run_grounding_check")
    @patch("rag.query_pipeline.run_subqueries_parallel")
    @patch("rag.query_pipeline.build_subquery_tasks")
    @patch("rag.query_pipeline.plan_sub_questions")
    @patch("rag.query_pipeline.detect_target_paper")
    @patch("rag.query_pipeline.cfg")
    @patch("rag.query_pipeline.Settings")
    def test_stage4_timeout_uses_stage3_facts(
        self, mock_settings, mock_cfg,
        mock_detect, mock_plan, mock_build, mock_run, *_
    ):
        _setup_cfg(mock_cfg)
        mock_detect.return_value = "paper_a"
        mock_plan.return_value = [{"paper": "paper_a", "sub_q": "Q?"}]
        mock_build.return_value = ([], {})
        mock_run.return_value = [("【paper_a】", "retrieved evidence from PaperA")]
        mock_settings.llm.stream_complete.side_effect = TimeoutError("stage4 stalled")

        result = pipeline_module.execute_structured_query(
            "What was reported?", {"paper_a": MagicMock()}
        )

        self.assertIn("retrieved evidence from PaperA", result)

    @patch("rag.query_pipeline.translate_to_traditional_chinese")
    @patch("rag.query_pipeline.run_grounding_check")
    @patch("rag.query_pipeline.run_subqueries_parallel")
    @patch("rag.query_pipeline.build_subquery_tasks")
    @patch("rag.query_pipeline.plan_sub_questions")
    @patch("rag.query_pipeline.select_relevant_papers")
    @patch("rag.query_pipeline._keyword_prefilter")
    @patch("rag.query_pipeline.detect_target_paper")
    @patch("rag.query_pipeline.cfg")
    @patch("rag.query_pipeline.Settings")
    def test_rag_not_found_includes_fallback_notice(
        self, mock_settings, mock_cfg,
        mock_detect, mock_prefilter, mock_select,
        mock_plan, mock_build, mock_run, *_
    ):
        _setup_cfg(mock_cfg)
        mock_detect.return_value = None
        mock_prefilter.return_value = ["paper_a"]
        mock_select.return_value = ["paper_a"]
        mock_plan.return_value = [{"paper": "paper_a", "sub_q": "Q?"}]
        mock_build.return_value = ([], {})
        mock_run.return_value = [("【paper_a】", "此論文未涉及相關議題，無法提供答案。")]

        chunk = MagicMock()
        chunk.delta = "Model knowledge answer."
        mock_settings.llm.stream_complete.return_value = [chunk]

        result = pipeline_module.execute_structured_query(
            "question?", {"paper_a": MagicMock()}
        )
        self.assertIn("資料來源說明", result)
        self.assertIn("Model knowledge answer.", result)

    @patch("rag.query_pipeline.translate_to_traditional_chinese")
    @patch("rag.query_pipeline.run_grounding_check")
    @patch("rag.query_pipeline.run_subqueries_parallel")
    @patch("rag.query_pipeline.build_subquery_tasks")
    @patch("rag.query_pipeline.plan_sub_questions")
    @patch("rag.query_pipeline.detect_target_paper")
    @patch("rag.query_pipeline.cfg")
    @patch("rag.query_pipeline.Settings")
    def test_on_artifact_captures_pre_translation_answer(
        self, mock_settings, mock_cfg,
        mock_detect, mock_plan, mock_build, mock_run,
        mock_grounding, mock_translate,
    ):
        _setup_cfg(mock_cfg)
        mock_cfg.EN_DRAFT_PIPELINE = True
        mock_detect.return_value = "paper_a"
        mock_plan.return_value = [{"paper": "paper_a", "sub_q": "Q?"}]
        mock_build.return_value = ([], {})
        mock_run.return_value = [(
            "【paper_a】",
            "The synthesis used Fe3O4 at 80°C for 4 hours with EDTA reagent solution.",
        )]
        chunk = MagicMock()
        chunk.delta = "English draft answer."
        mock_settings.llm.stream_complete.return_value = [chunk]
        mock_translate.return_value = "繁中最終答案"

        artifacts = {}
        result = pipeline_module.execute_structured_query(
            "question?", {"paper_a": MagicMock()}, on_artifact=artifacts.__setitem__,
        )

        self.assertIn("Fe3O4", artifacts["stage2_evidence"])
        self.assertIn("Fe3O4", artifacts["knowledge_base"])
        self.assertIn("Fe3O4", artifacts["stage4_prompt"])
        self.assertEqual(artifacts["stage4_draft"], "English draft answer.")
        self.assertEqual(artifacts["stage4_validated"], "English draft answer.")
        self.assertEqual(artifacts["stage5_verified"], "English draft answer.")
        self.assertEqual(artifacts["stage6_grounded_answer"], "English draft answer.")
        self.assertEqual(artifacts["answer_for_judge"], "English draft answer.")
        self.assertEqual(artifacts["stage7_translated_answer"], "繁中最終答案")
        self.assertEqual(artifacts["planning_detected_paper"], "paper_a")
        self.assertEqual(artifacts["planning_selected_papers"], ["paper_a"])
        self.assertEqual(result, "繁中最終答案")

    @patch("rag.query_pipeline.translate_to_traditional_chinese")
    @patch("rag.query_pipeline.run_grounding_check")
    @patch("rag.query_pipeline.run_subqueries_parallel")
    @patch("rag.query_pipeline.build_subquery_tasks")
    @patch("rag.query_pipeline.plan_sub_questions")
    @patch("rag.query_pipeline.detect_target_paper")
    @patch("rag.query_pipeline.cfg")
    @patch("rag.query_pipeline.Settings")
    def test_deterministic_initial_evidence_keeps_literal_measurement(
        self, mock_settings, mock_cfg,
        mock_detect, mock_plan, mock_build, mock_run,
        mock_grounding, mock_translate,
    ):
        _setup_cfg(mock_cfg)
        mock_cfg.SYNTHESIS_ENABLED = True
        mock_cfg.PARTIAL_RECOVERY_DETERMINISTIC_GUARDS_ENABLED = True
        mock_detect.return_value = "paper_a"
        mock_plan.return_value = [{"paper": "paper_a", "sub_q": "storage conditions"}]
        mock_build.return_value = ([], {})
        mock_run.return_value = [(
            "【paper_a】",
            "Retrieved evidence snippets:\n"
            "[Snippet 1] All three synthetic impurities were detectable at concentrations of "
            "0.5 µg/ml (or 0.1% of the BPA nominal working concentration).",
        )]
        chunk = MagicMock()
        chunk.delta = "Final answer."
        mock_settings.llm.stream_complete.return_value = [chunk]
        artifacts = {}

        with patch.object(
            pipeline_module._synthesizer,
            "synthesize",
            return_value="[Fact 1] BPA has synthetic impurities. (Source: paper_a)",
        ):
            pipeline_module.execute_structured_query(
                "Which BPA impurities are detected and under which storage conditions?",
                {"paper_a": MagicMock()},
                on_artifact=artifacts.__setitem__,
            )

        self.assertTrue(mock_run.call_args.kwargs["include_adjacent_evidence"])
        self.assertEqual(mock_run.call_args.kwargs["evidence_snippets_per_task"], 4)
        self.assertIn("0.5 µg/ml", artifacts["stage3_literal_facts"])
        self.assertIn("0.1%", artifacts["stage3_knowledge_base"])

    @patch("rag.query_pipeline.translate_to_traditional_chinese")
    @patch("rag.query_pipeline.run_grounding_check")
    @patch("rag.query_pipeline.run_subqueries_parallel")
    @patch("rag.query_pipeline.build_subquery_tasks")
    @patch("rag.query_pipeline.plan_sub_questions")
    @patch("rag.query_pipeline.detect_target_paper")
    @patch("rag.query_pipeline.cfg")
    @patch("rag.query_pipeline.Settings")
    def test_final_translation_can_be_disabled_for_english_draft(
        self, mock_settings, mock_cfg,
        mock_detect, mock_plan, mock_build, mock_run,
        mock_grounding, mock_translate,
    ):
        _setup_cfg(mock_cfg)
        mock_cfg.EN_DRAFT_PIPELINE = True
        mock_cfg.FINAL_TRANSLATION_ENABLED = False
        mock_detect.return_value = "paper_a"
        mock_plan.return_value = [{"paper": "paper_a", "sub_q": "Q?"}]
        mock_build.return_value = ([], {})
        mock_run.return_value = [(
            "【paper_a】",
            "This paper reports direct evidence for the answer in retrieved chunks.",
        )]
        chunk = MagicMock()
        chunk.delta = "English draft answer."
        mock_settings.llm.stream_complete.return_value = [chunk]

        result = pipeline_module.execute_structured_query(
            "question?", {"paper_a": MagicMock()}
        )

        self.assertEqual(result, "English draft answer.")
        mock_translate.assert_not_called()

    @patch("rag.query_pipeline.translate_to_traditional_chinese")
    @patch("rag.query_pipeline.run_grounding_check")
    @patch("rag.query_pipeline.run_subqueries_parallel")
    @patch("rag.query_pipeline.build_subquery_tasks")
    @patch("rag.query_pipeline.plan_sub_questions")
    @patch("rag.query_pipeline.detect_target_paper")
    @patch("rag.query_pipeline.cfg")
    @patch("rag.query_pipeline.Settings")
    def test_stage4_validator_rewrites_bad_comparison_answer(
        self, mock_settings, mock_cfg,
        mock_detect, mock_plan, mock_build, mock_run,
        mock_grounding, mock_translate,
    ):
        _setup_cfg(mock_cfg)
        mock_cfg.STAGE4_ANSWER_VALIDATION_ENABLED = True
        mock_detect.return_value = "paper_a"
        mock_plan.return_value = [{"paper": "paper_a", "sub_q": "Q?"}]
        mock_build.return_value = ([], {})
        kb = """
        {"comparison_json":{
          "direct_routes":[{"source":"bbb0683","route_phrase":"enantioselective alkylation followed by chymotrypsin-catalysed enzymatic hydrolysis"}],
          "review_comparison_sources":[{"source":"CMDC-20-e202500059"}],
          "dimensions":{
            "isotopic_enrichment":{"requested":true,"evidence_found":true},
            "scalability":{"requested":true,"evidence_found":true},
            "cost_effectiveness":{"requested":true,"evidence_found":true}
          }
        }}
        """
        mock_run.return_value = [("【paper_a】", kb)]
        chunk = MagicMock()
        chunk.delta = "No relevant query results or paper data were provided."
        mock_settings.llm.stream_complete.return_value = [chunk]

        with patch.object(pipeline_module, "_verifier") as mock_verifier:
            mock_verifier.correct.return_value = "Corrected comparison answer."
            result = pipeline_module.execute_structured_query(
                "Compare routes for isotopic enrichment, scalability, and cost-effectiveness.",
                {"paper_a": MagicMock()},
            )

        self.assertEqual(result, "Corrected comparison answer.")
        self.assertTrue(mock_verifier.correct.called)


class TestRunEvalCorrectnessCandidate(unittest.TestCase):
    def test_prefers_pre_translation_answer_for_judge(self):
        candidate, source = eval_run._correctness_candidate(
            "繁中最終答案", {"answer_for_judge": "English draft answer"}
        )
        self.assertEqual(candidate, "English draft answer")
        self.assertEqual(source, "answer_for_judge")

    def test_prefers_canonical_draft_even_for_chinese_reference(self):
        candidate, source = eval_run._correctness_candidate(
            "繁中最終答案", {"answer_for_judge": "English draft answer"}, "中文標準答案"
        )
        self.assertEqual(candidate, "English draft answer")
        self.assertEqual(source, "answer_for_judge")

    def test_status_includes_correctness_and_translation(self):
        base = {
            "answer": "ok",
            "paper_selection_recall": 1.0,
            "retrieval_span_recall": 1.0,
            "grounding_score": 1.0,
        }
        self.assertEqual(eval_run._q_status({**base, "correctness": 0.25}), "❌")
        self.assertEqual(eval_run._q_status({**base, "correctness": 0.5}), "⚠️")
        self.assertEqual(
            eval_run._q_status({
                **base,
                "correctness": 1.0,
                "translation_fidelity": 0.25,
            }),
            "❌",
        )

    def test_writes_debug_artifact_files(self):
        old_enabled = cfg.EVAL_DEBUG_ARTIFACTS_ENABLED
        try:
            cfg.EVAL_DEBUG_ARTIFACTS_ENABLED = True
            row = {
                "answer_for_judge": "English draft answer",
                "translated_answer": "繁中最終答案",
                "answer": "Final answer",
                "correctness_detail": {"reason": "ok"},
                "translation_detail": {"reason": "faithful"},
            }
            m = mock_open()
            with patch.object(eval_run, "RESULTS_DIR", "results"), \
                    patch.object(eval_run.os, "makedirs") as makedirs, \
                    patch("builtins.open", m):
                out_dir = eval_run._write_debug_artifacts(
                    "label with spaces",
                    "Q08",
                    "Question text?",
                    {
                        "stage2_evidence": "Evidence block",
                        "stage2_recovery_evidence": "Expanded evidence block",
                        "stage3_generation_meta": [{"done_reason": "stop"}],
                        "stage3_fact_contract": "Stage 3 contract",
                        "stage3_recovery_prompt": "Recovery prompt",
                        "stage3_recovery_fact_contract": "Recovery contract",
                        "stage3_recovery_literal_facts": "Literal facts",
                        "partial_recovery_assessment": "{\"verdict\":\"ANSWERABLE\"}",
                        "stage4_fact_contract": "Stage 4 contract",
                        "stage4_grounding_claims": "Grounding claims",
                        "stage4_draft": "Draft",
                    },
                    ["[planning] ok"],
                    row,
                )

            self.assertEqual(out_dir, os.path.join("results", "label_with_spaces"))
            makedirs.assert_called_once_with(out_dir, exist_ok=True)
            opened_paths = [args[0] for args, _ in m.call_args_list]
            self.assertIn(os.path.join(out_dir, "Q08_stage2_evidence.txt"), opened_paths)
            self.assertIn(os.path.join(out_dir, "Q08_stage3_generation_meta.txt"), opened_paths)
            self.assertIn(os.path.join(out_dir, "Q08_stage3_fact_contract.txt"), opened_paths)
            self.assertIn(os.path.join(out_dir, "Q08_stage2_recovery_evidence.txt"), opened_paths)
            self.assertIn(os.path.join(out_dir, "Q08_stage3_recovery_prompt.txt"), opened_paths)
            self.assertIn(os.path.join(out_dir, "Q08_stage3_recovery_fact_contract.txt"), opened_paths)
            self.assertIn(os.path.join(out_dir, "Q08_stage3_recovery_literal_facts.txt"), opened_paths)
            self.assertIn(os.path.join(out_dir, "Q08_partial_recovery_assessment.txt"), opened_paths)
            self.assertIn(os.path.join(out_dir, "Q08_stage7_translated_answer.txt"), opened_paths)
            self.assertIn(os.path.join(out_dir, "Q08_translation_judge.json"), opened_paths)
            self.assertIn(os.path.join(out_dir, "Q08_stage4_draft.txt"), opened_paths)
            self.assertIn(os.path.join(out_dir, "Q08_stage4_fact_contract.txt"), opened_paths)
            self.assertIn(os.path.join(out_dir, "Q08_stage4_grounding_claims.txt"), opened_paths)
        finally:
            cfg.EVAL_DEBUG_ARTIFACTS_ENABLED = old_enabled


class TestExecuteStructuredQueryStream(unittest.TestCase):
    def _collect(self, gen):
        return list(gen)

    @patch("rag.query_pipeline.translate_to_traditional_chinese")
    @patch("rag.query_pipeline.run_grounding_check")
    @patch("rag.query_pipeline.run_subqueries_parallel")
    @patch("rag.query_pipeline.build_subquery_tasks")
    @patch("rag.query_pipeline.plan_sub_questions")
    @patch("rag.query_pipeline.detect_target_paper")
    @patch("rag.query_pipeline.cfg")
    @patch("rag.query_pipeline.Settings")
    def test_yields_status_tokens_and_content(
        self, mock_settings, mock_cfg,
        mock_detect, mock_plan, mock_build, mock_run, *_
    ):
        _setup_cfg(mock_cfg)
        mock_detect.return_value = "paper_a"
        mock_plan.return_value = [{"paper": "paper_a", "sub_q": "Q?"}]
        mock_build.return_value = ([], {})
        mock_run.return_value = [(
            "【paper_a】",
            "The synthesis used Fe3O4 at 80°C for 4 hours with EDTA and iron chloride.",
        )]
        chunk = MagicMock()
        chunk.delta = "Streamed answer token."
        mock_settings.llm.stream_complete.return_value = [chunk]

        tokens = self._collect(
            pipeline_module.execute_structured_query_stream(
                "What is the synthesis?", {"paper_a": MagicMock()}
            )
        )
        status_tokens = [t for t in tokens if t.startswith("[STATUS]")]
        content_tokens = [t for t in tokens if not t.startswith("[STATUS]")]

        self.assertTrue(len(status_tokens) > 0, "Expected at least one [STATUS] token")
        self.assertIn("Streamed answer token.", "".join(content_tokens))

    @patch("rag.query_pipeline.translate_to_traditional_chinese")
    @patch("rag.query_pipeline.run_grounding_check")
    @patch("rag.query_pipeline.run_subqueries_parallel")
    @patch("rag.query_pipeline.build_subquery_tasks")
    @patch("rag.query_pipeline.plan_sub_questions")
    @patch("rag.query_pipeline.detect_target_paper")
    @patch("rag.query_pipeline.cfg")
    @patch("rag.query_pipeline.Settings")
    def test_stream_method_fact_renderer_skips_stage4_llm(
        self, mock_settings, mock_cfg,
        mock_detect, mock_plan, mock_build, mock_run,
        mock_grounding, mock_translate,
    ):
        _setup_cfg(mock_cfg)
        mock_cfg.SYNTHESIS_ENABLED = True
        mock_cfg.CITATION_GROUNDING_ENABLED = True
        mock_cfg.METHOD_FACT_LIST_DIRECT_RENDER_ENABLED = True
        mock_cfg.EN_DRAFT_PIPELINE = True
        mock_cfg.FINAL_TRANSLATION_ENABLED = False
        mock_detect.return_value = "PaperA"
        mock_plan.return_value = [{
            "paper": "PaperA",
            "sub_q": "Report the key steps and experimental conditions.",
        }]
        mock_build.return_value = ([], {})
        mock_run.return_value = [(
            "【PaperA】",
            "Retrieved paper evidence contains a complete source-bound synthesis method.",
        )]
        kb = """
[Fact 1] Product B was synthesized by a hybrid process. (Source: PaperA)
[Fact 2] Reagent A reacted to give product B. (Source: PaperA)
[Fact 3] The reaction was conducted in THF at -78°C. (Source: PaperA)
"""
        mock_grounding.side_effect = lambda full_text, *args, **kwargs: (full_text, "")

        with patch.object(pipeline_module._synthesizer, "synthesize", return_value=kb):
            tokens = self._collect(
                pipeline_module.execute_structured_query_stream(
                    "What synthesis process is used and what are its key steps?",
                    {"PaperA": MagicMock()},
                )
            )

        self.assertIn("THF at -78°C", "".join(tokens))
        mock_settings.llm.stream_complete.assert_not_called()
        self.assertTrue(mock_grounding.call_args.kwargs["grounding_claims"])
        mock_translate.assert_not_called()

    @patch("rag.query_pipeline.translate_to_traditional_chinese")
    @patch("rag.query_pipeline.run_grounding_check")
    @patch("rag.query_pipeline.run_subqueries_parallel")
    @patch("rag.query_pipeline.build_subquery_tasks")
    @patch("rag.query_pipeline.plan_sub_questions")
    @patch("rag.query_pipeline.select_relevant_papers")
    @patch("rag.query_pipeline._keyword_prefilter")
    @patch("rag.query_pipeline.detect_target_paper")
    @patch("rag.query_pipeline.cfg")
    @patch("rag.query_pipeline.Settings")
    def test_stream_fallback_notice_in_output(
        self, mock_settings, mock_cfg,
        mock_detect, mock_prefilter, mock_select,
        mock_plan, mock_build, mock_run, *_
    ):
        _setup_cfg(mock_cfg)
        mock_detect.return_value = None
        mock_prefilter.return_value = ["paper_a"]
        mock_select.return_value = ["paper_a"]
        mock_plan.return_value = [{"paper": "paper_a", "sub_q": "Q?"}]
        mock_build.return_value = ([], {})
        mock_run.return_value = [("【paper_a】", "此論文未涉及相關議題，無法提供答案。")]

        chunk = MagicMock()
        chunk.delta = "Fallback answer content."
        mock_settings.llm.stream_complete.return_value = [chunk]

        tokens = self._collect(
            pipeline_module.execute_structured_query_stream(
                "question?", {"paper_a": MagicMock()}
            )
        )
        full = "".join(tokens)
        self.assertIn("資料來源說明", full)
        self.assertIn("Fallback answer content.", full)


if __name__ == "__main__":
    unittest.main(verbosity=2)
