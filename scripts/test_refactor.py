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
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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
    build_subquery_tasks, run_subqueries_parallel,
)
from rag.query_grounding_flow import (
    _extract_direct_citation_section,
    _partition_results_by_section,
    run_grounding_check,
)
from rag.query_prompts import build_synthesis_prompt, build_fallback_prompt
from rag.query_translation import translate_to_traditional_chinese
import rag.query_pipeline as pipeline_module


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
        with patch("rag.citation_grounding.split_into_sentences", side_effect=self._split):
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
        with patch("rag.citation_grounding.split_into_sentences", side_effect=self._split):
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
# query_pipeline — integration (all external calls mocked)
# ══════════════════════════════════════════════════════════════════════════════
def _setup_cfg(cfg_mock):
    cfg_mock.REVIEW_MODE = False
    cfg_mock.SYNTHESIS_ENABLED = False
    cfg_mock.VERIFY_ENABLED = False
    cfg_mock.CITATION_GROUNDING_ENABLED = False
    cfg_mock.EN_DRAFT_PIPELINE = False
    cfg_mock.REASONING_MODE = "strict"
    cfg_mock.SUBQUERY_MAX_WORKERS = 1


class TestExecuteStructuredQuery(unittest.TestCase):
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
