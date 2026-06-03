# scripts/test_nli_extensions.py
# 單元測試：mDeBERTa NLI 擴展（矛盾偵測、子命題拆解、多來源聯合驗證）
# 以及 PlanExecutor（plan_executor.py）。
#
# 所有 NLI / LLM / GPU 呼叫皆 mock，不需 Ollama 也不需 GPU。
#
# 使用方式：
#   conda activate llm_env
#   cd E:\Projects\rag_project
#   python scripts/test_nli_extensions.py

import sys
import os
import types
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# ── 前置 stub：在 import citation_grounding 前注入，避免載入 GPU 模型 ────

if "torch" not in sys.modules:
    _torch = types.ModuleType("torch")
    _torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    _torch.no_grad = lambda: __import__("contextlib").nullcontext()
    _torch.softmax = None          # 不會真的被呼叫（_run_nli 全 mock）
    sys.modules["torch"] = _torch

if "transformers" not in sys.modules:
    _tf = types.ModuleType("transformers")
    sys.modules["transformers"] = _tf

if "requests" not in sys.modules:
    _req = types.ModuleType("requests")
    class _Timeout(OSError):
        pass
    _req.exceptions = types.SimpleNamespace(Timeout=_Timeout)
    # patch() 只能覆寫已存在的 attribute；加上 post 佔位讓後續 @patch 可以替換
    from unittest.mock import MagicMock as _MM
    _req.post = _MM()
    sys.modules["requests"] = _req

# ── import 後調整 config flag，讓 EN_DRAFT_PIPELINE=True 跳過翻譯步驟 ──
import config as cfg
cfg.EN_DRAFT_PIPELINE        = True
cfg.NLI_CONTRADICTION_ENABLED = True
cfg.NLI_DECOMPOSE_ENABLED     = True
cfg.NLI_JOINT_VERIFY_ENABLED  = True
cfg.NLI_TRANSLATE_TO_EN       = False   # 測試時不呼叫翻譯 LLM

from unittest import TestCase, main as _unittest_main
from unittest.mock import patch, MagicMock
import unittest

from rag.citation_grounding import (
    check_citation_grounding,
    decompose_and_verify,
    joint_verify,
    compute_grounding_score,
)
from rag.task_state import ResearchPlan, SubTask, TaskStatus
from rag.plan_executor import PlanExecutor


# ══════════════════════════════════════════════════════════════════
#  測試用 fixture 資料
# ══════════════════════════════════════════════════════════════════

_CHUNKS = [
    {"id": "C001", "text": "The synthesis uses KBH4 as the reducing agent for nZVI formation."},
    {"id": "C002", "text": "Gelatin aerogel provides high porosity and large surface area as a support matrix."},
    {"id": "C003", "text": "Glycine modifies the NZVI surface through amide bond formation with gelatin carboxyl groups."},
]


def _nli_entailment(e=0.8, n=0.15, c=0.05):
    """Helper：回傳指定 entailment/neutral/contradiction 分數的 NLI 結果 dict。"""
    return {"entailment": e, "neutral": n, "contradiction": c}


# ══════════════════════════════════════════════════════════════════
#  1. 矛盾偵測（check_citation_grounding + NLI_CONTRADICTION_ENABLED）
# ══════════════════════════════════════════════════════════════════

class TestContradictionDetection(TestCase):

    def _run(self, sentences, nli_side_effect, joint_enabled=False, decomp_enabled=False):
        """共用 helper：patch _run_nli 後執行 check_citation_grounding。"""
        old_joint  = cfg.NLI_JOINT_VERIFY_ENABLED
        old_decomp = cfg.NLI_DECOMPOSE_ENABLED
        cfg.NLI_JOINT_VERIFY_ENABLED  = joint_enabled
        cfg.NLI_DECOMPOSE_ENABLED     = decomp_enabled
        try:
            with patch("rag.citation_grounding._run_nli", side_effect=nli_side_effect):
                return check_citation_grounding(sentences, _CHUNKS)
        finally:
            cfg.NLI_JOINT_VERIFY_ENABLED  = old_joint
            cfg.NLI_DECOMPOSE_ENABLED     = old_decomp

    def test_supported_sentence(self):
        """高 entailment、低 contradiction → SUPPORTED"""
        results = self._run(
            ["KBH4 is used as the reducing agent."],
            nli_side_effect=[_nli_entailment(0.85, 0.10, 0.05)] * len(_CHUNKS),
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["status"], "SUPPORTED")
        self.assertTrue(results[0]["supported"])
        self.assertFalse(results[0]["contradiction_detected"])

    def test_unsupported_sentence(self):
        """低 entailment、低 contradiction → UNSUPPORTED"""
        results = self._run(
            ["The material uses platinum as a catalyst."],
            nli_side_effect=[_nli_entailment(0.1, 0.8, 0.1)] * len(_CHUNKS),
        )
        self.assertEqual(results[0]["status"], "UNSUPPORTED")
        self.assertFalse(results[0]["supported"])
        self.assertFalse(results[0]["contradiction_detected"])

    def test_conflict_detected(self):
        """
        高 contradiction（>0.7）且 entailment >= 0.25 → CONFLICT
        模擬答案聲稱「不需要還原劑」，但論文原文說需要 KBH4。
        """
        # entailment=0.4（有一定相關度）, contradiction=0.8（高度矛盾）
        results = self._run(
            ["No reducing agent is needed in the synthesis."],
            nli_side_effect=[_nli_entailment(0.4, 0.1, 0.8)] * len(_CHUNKS),
        )
        self.assertEqual(results[0]["status"], "CONFLICT")
        self.assertTrue(results[0]["contradiction_detected"])

    def test_low_entailment_no_conflict(self):
        """
        entailment < 0.25 時，即使 contradiction 高，也不觸發 CONFLICT（視為雜訊）
        """
        results = self._run(
            ["An unrelated claim about photosynthesis."],
            nli_side_effect=[_nli_entailment(0.1, 0.1, 0.8)] * len(_CHUNKS),
        )
        self.assertEqual(results[0]["status"], "UNSUPPORTED")
        self.assertFalse(results[0]["contradiction_detected"])

    def test_empty_sentences(self):
        """空 sentences → 回傳空 list"""
        with patch("rag.citation_grounding._run_nli"):
            results = check_citation_grounding([], _CHUNKS)
        self.assertEqual(results, [])

    def test_empty_chunks(self):
        """空 chunks → 回傳空 list"""
        with patch("rag.citation_grounding._run_nli"):
            results = check_citation_grounding(["Any sentence."], [])
        self.assertEqual(results, [])

    def test_multiple_sentences_picks_best_chunk(self):
        """每個 sentence 應分別找最高 entailment 的 chunk。"""
        # chunk C001=0.9, C002=0.3, C003=0.5 → best = C001
        side_effect = [
            _nli_entailment(0.9, 0.05, 0.05),  # C001
            _nli_entailment(0.3, 0.6, 0.1),    # C002
            _nli_entailment(0.5, 0.4, 0.1),    # C003
        ]
        results = self._run(
            ["KBH4 reduces iron ions to form nZVI."],
            nli_side_effect=side_effect,
        )
        self.assertEqual(results[0]["best_chunk"], "C001")
        self.assertAlmostEqual(results[0]["confidence"], 0.9)


# ══════════════════════════════════════════════════════════════════
#  2. 子命題拆解驗證（decompose_and_verify）
# ══════════════════════════════════════════════════════════════════

class TestDecomposeAndVerify(TestCase):

    def _mock_llm_response(self, sub_claims: list):
        """建立 mock requests.post 回傳 JSON list。"""
        mock_resp       = MagicMock()
        mock_resp.ok    = True
        mock_resp.json.return_value = {"response": json.dumps(sub_claims)}
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    def test_disabled(self):
        """NLI_DECOMPOSE_ENABLED=False → 直接回傳空 sub_claims，chain_complete=True"""
        cfg.NLI_DECOMPOSE_ENABLED = False
        result = decompose_and_verify("Any conclusion.", _CHUNKS)
        cfg.NLI_DECOMPOSE_ENABLED = True
        self.assertEqual(result["sub_claims"], [])
        self.assertTrue(result["chain_complete"])

    def test_basic_decomposition_all_supported(self):
        """兩個子命題都 SUPPORTED → chain_complete=True"""
        sub_claims_text = [
            "KBH4 acts as the reducing agent.",
            "The synthesis occurs at ambient pressure.",
        ]
        mock_resp = self._mock_llm_response(sub_claims_text)

        with patch("requests.post", return_value=mock_resp):
            with patch("rag.citation_grounding._run_nli",
                       return_value=_nli_entailment(0.8, 0.15, 0.05)):
                result = decompose_and_verify(
                    "The synthesis uses KBH4 under ambient conditions.",
                    _CHUNKS,
                )

        self.assertEqual(len(result["sub_claims"]), 2)
        self.assertTrue(result["chain_complete"])
        for sc in result["sub_claims"]:
            self.assertEqual(sc["status"], "SUPPORTED")

    def test_inference_bridge_claim(self):
        """中等分數（0.4–0.65 之間）→ INFERENCE_BRIDGE"""
        sub_claims_text = ["The aerogel enhances electron selectivity."]
        mock_resp = self._mock_llm_response(sub_claims_text)

        with patch("requests.post", return_value=mock_resp):
            with patch("rag.citation_grounding._run_nli",
                       return_value=_nli_entailment(0.52, 0.38, 0.10)):
                result = decompose_and_verify(
                    "Gelatin aerogel enhances NZVI electron selectivity.",
                    _CHUNKS,
                )

        self.assertEqual(result["sub_claims"][0]["status"], "INFERENCE_BRIDGE")
        self.assertTrue(result["chain_complete"])  # INFERENCE_BRIDGE 不算 chain break

    def test_chain_incomplete_on_unsupported(self):
        """一個子命題 UNSUPPORTED → chain_complete=False"""
        sub_claims_text = [
            "KBH4 is used as the reducing agent.",  # SUPPORTED
            "The reaction requires a nitrogen atmosphere.",  # UNSUPPORTED（論文說不需要）
        ]
        mock_resp = self._mock_llm_response(sub_claims_text)

        # 第一個 claim 三個 chunk 給高分，第二個給低分
        nli_scores = (
            [_nli_entailment(0.8, 0.1, 0.1)] * 3   # claim 1 × 3 chunks
            + [_nli_entailment(0.1, 0.8, 0.1)] * 3  # claim 2 × 3 chunks
        )

        with patch("requests.post", return_value=mock_resp):
            with patch("rag.citation_grounding._run_nli", side_effect=nli_scores):
                result = decompose_and_verify(
                    "The synthesis uses KBH4 under nitrogen atmosphere.",
                    _CHUNKS,
                )

        statuses = [sc["status"] for sc in result["sub_claims"]]
        self.assertIn("UNSUPPORTED", statuses)
        self.assertFalse(result["chain_complete"])

    def test_llm_failure_fallback_to_original(self):
        """LLM 呼叫失敗 → fallback 使用原始結論作為單一子命題"""
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("connection refused")

        with patch("requests.post", return_value=mock_resp):
            with patch("rag.citation_grounding._run_nli",
                       return_value=_nli_entailment(0.7, 0.2, 0.1)):
                result = decompose_and_verify(
                    "The original conclusion sentence.",
                    _CHUNKS,
                )

        # fallback：sub_claims 應含原始結論
        self.assertEqual(len(result["sub_claims"]), 1)
        self.assertEqual(result["sub_claims"][0]["claim"], "The original conclusion sentence.")

    def test_llm_invalid_json_fallback(self):
        """LLM 回傳非 JSON → fallback 使用原始結論"""
        mock_resp       = MagicMock()
        mock_resp.ok    = True
        mock_resp.json.return_value = {"response": "This is not JSON."}
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_resp):
            with patch("rag.citation_grounding._run_nli",
                       return_value=_nli_entailment(0.5, 0.3, 0.2)):
                result = decompose_and_verify("Any conclusion.", _CHUNKS)

        self.assertEqual(len(result["sub_claims"]), 1)


# ══════════════════════════════════════════════════════════════════
#  3. 多來源聯合驗證（joint_verify）
# ══════════════════════════════════════════════════════════════════

class TestJointVerify(TestCase):

    def test_disabled(self):
        """NLI_JOINT_VERIFY_ENABLED=False → 回傳 empty/zero 結果"""
        cfg.NLI_JOINT_VERIFY_ENABLED = False
        result = joint_verify("Any claim.", _CHUNKS)
        cfg.NLI_JOINT_VERIFY_ENABLED = True
        self.assertFalse(result["is_inference_bridge"])
        self.assertEqual(result["individual_scores"], [])
        self.assertEqual(result["joint_score"], 0.0)

    def test_inference_bridge_detected(self):
        """
        個別分數低（avg < 0.5），但 joint 高（>= 0.65） → is_inference_bridge=True
        """
        # 3 chunks：個別低，聯合後高
        individual_scores = [0.3, 0.25, 0.28]
        # _run_nli 被呼叫：先 3 次個別（各 chunk），再 1 次聯合
        nli_side = (
            [_nli_entailment(s, 0.5, 0.2) for s in individual_scores]
            + [_nli_entailment(0.72, 0.18, 0.10)]  # joint
        )

        with patch("rag.citation_grounding._run_nli", side_effect=nli_side):
            result = joint_verify("The combined mechanism explains the high degradation.", _CHUNKS)

        self.assertTrue(result["is_inference_bridge"])
        self.assertAlmostEqual(result["joint_score"], 0.72)
        self.assertEqual(len(result["individual_scores"]), 3)
        self.assertGreater(len(result["bridge_sources"]), 0)

    def test_not_bridge_high_individual(self):
        """個別分數已高（avg >= 0.5）→ 不觸發 inference_bridge"""
        individual_scores = [0.82, 0.75, 0.68]
        nli_side = (
            [_nli_entailment(s, 0.15, 0.05) for s in individual_scores]
            + [_nli_entailment(0.88, 0.10, 0.02)]  # joint
        )

        with patch("rag.citation_grounding._run_nli", side_effect=nli_side):
            result = joint_verify("KBH4 reduces iron ions directly.", _CHUNKS)

        self.assertFalse(result["is_inference_bridge"])
        self.assertEqual(result["bridge_sources"], [])

    def test_not_bridge_low_joint(self):
        """個別低 AND joint 也低（< 0.65）→ 不觸發 inference_bridge"""
        individual_scores = [0.2, 0.15, 0.18]
        nli_side = (
            [_nli_entailment(s, 0.6, 0.2) for s in individual_scores]
            + [_nli_entailment(0.35, 0.45, 0.20)]  # joint 也低
        )

        with patch("rag.citation_grounding._run_nli", side_effect=nli_side):
            result = joint_verify("Completely unsupported claim.", _CHUNKS)

        self.assertFalse(result["is_inference_bridge"])

    def test_empty_facts(self):
        """無任何 chunks → 回傳 zero 結果，不丟例外"""
        with patch("rag.citation_grounding._run_nli"):
            result = joint_verify("Some claim.", [])
        self.assertFalse(result["is_inference_bridge"])
        self.assertEqual(result["individual_scores"], [])

    def test_top3_selection(self):
        """有 4 個 chunks，應只取 top-3 最高 entailment 做聯合驗證。"""
        chunks_4 = _CHUNKS + [{"id": "C004", "text": "Extra chunk with low relevance."}]
        # C001=0.8, C002=0.6, C003=0.5, C004=0.2 → top3 = C001,C002,C003
        individual_scores = [0.8, 0.6, 0.5, 0.2]
        nli_side = (
            [_nli_entailment(s, 0.1, 0.1) for s in individual_scores]
            + [_nli_entailment(0.71, 0.19, 0.10)]  # joint
        )

        with patch("rag.citation_grounding._run_nli", side_effect=nli_side):
            result = joint_verify("A cross-chunk claim.", chunks_4)

        # individual_scores 只保留 top-3
        self.assertEqual(len(result["individual_scores"]), 3)
        # top-3 的最低分應是 0.5，不含 0.2
        self.assertNotIn(0.2, result["individual_scores"])


# ══════════════════════════════════════════════════════════════════
#  4. compute_grounding_score
# ══════════════════════════════════════════════════════════════════

class TestComputeGroundingScore(TestCase):

    def test_all_supported(self):
        results = [
            {"supported": True,  "status": "SUPPORTED"},
            {"supported": True,  "status": "SUPPORTED"},
        ]
        self.assertEqual(compute_grounding_score(results), 1.0)

    def test_none_supported(self):
        results = [
            {"supported": False, "status": "UNSUPPORTED"},
            {"supported": False, "status": "CONFLICT"},
        ]
        self.assertEqual(compute_grounding_score(results), 0.0)

    def test_inference_bridge_counts_as_supported(self):
        """INFERENCE_BRIDGE 應計入 supported（spec 定義）"""
        results = [
            {"supported": False, "status": "INFERENCE_BRIDGE"},
            {"supported": False, "status": "UNSUPPORTED"},
        ]
        # 1/2 = 0.5
        self.assertAlmostEqual(compute_grounding_score(results), 0.5)

    def test_empty(self):
        self.assertEqual(compute_grounding_score([]), 1.0)


# ══════════════════════════════════════════════════════════════════
#  5. PlanExecutor
# ══════════════════════════════════════════════════════════════════

def _make_engine_mock(return_value="Mock answer from engine."):
    """建立可被 _retrieve_nodes / _generate_from_nodes 使用的 engine mock。"""
    engine      = MagicMock()
    engine.retriever = MagicMock()
    engine.retriever.retrieve.return_value = [MagicMock()]
    return engine


class TestPlanExecutor(TestCase):

    def _make_executor(self, engine_answers=None):
        """Helper：建立 PlanExecutor，paper engines 全部 mock。"""
        engines = {
            "paper_a": _make_engine_mock(),
            "paper_b": _make_engine_mock(),
        }
        return PlanExecutor(engines, on_status=lambda _: None), engines

    # ── 5-1：單一任務（無依賴）────────────────────────────────────

    def test_single_task_completes(self):
        executor, _ = self._make_executor()
        plan = ResearchPlan(
            question="What is nZVI?",
            tasks=[SubTask(id="T1", question="Define nZVI.", depends_on=[])],
        )

        with patch("rag.query_embedding_guard.prepare_query_text", side_effect=lambda q: q):
            with patch("rag.query_retrieval._retrieve_nodes", return_value=[]):
                with patch("rag.query_retrieval._generate_from_nodes",
                           return_value="nZVI is nano zero-valent iron."):
                    result = executor.execute(plan)

        self.assertTrue(result.all_completed())
        self.assertEqual(result.tasks[0].status, TaskStatus.COMPLETED)
        self.assertIn("nZVI", result.tasks[0].result)

    # ── 5-2：依賴鏈（T2 等 T1 完成）─────────────────────────────

    def test_dependency_chain_executes_in_order(self):
        executor, _ = self._make_executor()
        # 每篇論文都跑，所以每個任務會被記錄兩次（paper_a / paper_b）
        # 用 set 確認兩個 task ID 都有執行，順序以 T1 全部先於 T2 驗證
        execution_log = []

        def _fake_generate(engine, nodes, query):
            if "T1 question" in query:
                execution_log.append("T1")
                return "T1 result."
            if "T2 question" in query:
                execution_log.append("T2")
                return "T2 result."
            return "Unknown."

        plan = ResearchPlan(
            question="Complex research question.",
            tasks=[
                SubTask(id="T1", question="T1 question.", depends_on=[]),
                SubTask(id="T2", question="T2 question.", depends_on=["T1"]),
            ],
        )

        with patch("rag.query_embedding_guard.prepare_query_text", side_effect=lambda q: q):
            with patch("rag.query_retrieval._retrieve_nodes", return_value=[]):
                with patch("rag.query_retrieval._generate_from_nodes",
                           side_effect=_fake_generate):
                    result = executor.execute(plan)

        # T1 的所有 engine 執行應先於任何 T2 執行
        last_t1 = max((i for i, v in enumerate(execution_log) if v == "T1"), default=-1)
        first_t2 = min((i for i, v in enumerate(execution_log) if v == "T2"), default=999)
        self.assertLess(last_t1, first_t2)
        self.assertTrue(result.all_completed())

    # ── 5-3：前置任務失敗 → 後置任務也標 FAILED ─────────────────

    def test_failed_dependency_cascades(self):
        executor, _ = self._make_executor()

        plan = ResearchPlan(
            question="Question with failing sub-task.",
            tasks=[
                SubTask(id="T1", question="T1 question.", depends_on=[]),
                SubTask(id="T2", question="T2 question.", depends_on=["T1"]),
            ],
        )

        with patch("rag.query_embedding_guard.prepare_query_text", side_effect=lambda q: q):
            with patch("rag.query_retrieval._retrieve_nodes",
                       side_effect=RuntimeError("embedding failed")):
                result = executor.execute(plan)

        self.assertEqual(result.tasks[0].status, TaskStatus.FAILED)
        self.assertEqual(result.tasks[1].status, TaskStatus.FAILED)
        self.assertIn("T1", result.tasks[1].error)

    # ── 5-4：_build_context_for_task 內容驗證 ────────────────────

    def test_build_context_includes_dep_results(self):
        executor, _ = self._make_executor()
        plan = ResearchPlan(
            question="Full question.",
            tasks=[
                SubTask(id="T1", question="T1?", depends_on=[],
                        status=TaskStatus.COMPLETED, result="T1 conclusion text."),
                SubTask(id="T2", question="T2?", depends_on=["T1"]),
            ],
        )

        context = executor._build_context_for_task(plan.tasks[1], plan)

        self.assertIn("T1 conclusion text.", context)
        self.assertIn("T1", context)

    def test_build_context_empty_for_no_deps(self):
        executor, _ = self._make_executor()
        plan = ResearchPlan(
            question="Q.",
            tasks=[SubTask(id="T1", question="T1?", depends_on=[])],
        )

        context = executor._build_context_for_task(plan.tasks[0], plan)
        self.assertEqual(context, "")

    # ── 5-5：synthesize_final_answer 退回 to_summary() ───────────

    def test_synthesize_fallback_on_llm_failure(self):
        executor, _ = self._make_executor()
        plan = ResearchPlan(
            question="What is nZVI?",
            tasks=[
                SubTask(id="T1", question="Define nZVI.", depends_on=[],
                        status=TaskStatus.COMPLETED, result="nZVI is nano zero-valent iron."),
            ],
        )

        import sys as _sys
        requests_mod = _sys.modules["requests"]
        old_post = getattr(requests_mod, "post", None)
        requests_mod.post = MagicMock(side_effect=Exception("connection refused"))

        try:
            answer = executor.synthesize_final_answer(plan)
        finally:
            if old_post is not None:
                requests_mod.post = old_post
            else:
                del requests_mod.post

        # fallback → to_summary() 內容
        self.assertIn("T1", answer)
        self.assertIn("nZVI", answer)


# ══════════════════════════════════════════════════════════════════
#  執行
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    for cls in [
        TestContradictionDetection,
        TestDecomposeAndVerify,
        TestJointVerify,
        TestComputeGroundingScore,
        TestPlanExecutor,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
