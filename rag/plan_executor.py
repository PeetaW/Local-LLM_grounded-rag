# rag/plan_executor.py
# Plan-and-Execute 執行器（Task 3-C）
# PLAN_EXECUTE_ENABLED=True 時替代 Stage 2-3 的順序執行邏輯。
#
# 核心優勢：後置任務在查詢時能看到前置任務的結論，
# 推論品質高於無依賴資訊的並行查詢。
#
# 使用方式（由 query_pipeline.py 在 PLAN_EXECUTE_ENABLED=True 時呼叫）：
#   executor = PlanExecutor(paper_engines, on_status=_status)
#   plan = ResearchPlan(question=question, tasks=[...])
#   plan = executor.execute(plan)
#   final_answer = executor.synthesize_final_answer(plan)

import time
import config as cfg
from rag.task_state import ResearchPlan, SubTask, TaskStatus


class PlanExecutor:

    def __init__(self, paper_engines: dict, on_status=None):
        self._paper_engines = paper_engines
        self._on_status     = on_status or print

    def _status(self, msg: str):
        self._on_status(msg)

    # ──────────────────────────────────────────────────────────────
    #  Public
    # ──────────────────────────────────────────────────────────────

    def execute(self, plan: ResearchPlan) -> ResearchPlan:
        """
        按依賴順序執行 ResearchPlan 中的所有子任務。
        每一輪取出「前置全部 COMPLETED 且自身 PENDING」的任務一起執行。
        回傳填好 result / status 的 plan 物件。
        """
        max_rounds = len(plan.tasks) + 1  # 防止依賴環造成無限迴圈

        for round_i in range(max_rounds):
            ready = plan.get_ready_tasks()

            if not ready:
                if plan.all_completed():
                    break
                # 有未完成但無 ready tasks → 前置任務有失敗，把剩餘 PENDING 一起標 FAILED
                failed_ids = [t.id for t in plan.tasks if t.status == TaskStatus.FAILED]
                self._status(
                    f"  ⚠️  [plan-execute] 前置任務失敗：{failed_ids}，"
                    f"取消剩餘 pending 任務"
                )
                for t in plan.tasks:
                    if t.status == TaskStatus.PENDING:
                        t.status = TaskStatus.FAILED
                        t.error  = f"前置任務失敗：{[d for d in t.depends_on if d in failed_ids]}"
                break

            self._status(
                f"  🔄 [plan-execute] Round {round_i + 1}："
                f"執行 {len(ready)} 個任務 "
                f"({', '.join(t.id for t in ready)})"
            )

            for task in ready:
                task.status = TaskStatus.RUNNING
                t0 = time.perf_counter()
                context = self._build_context_for_task(task, plan)
                try:
                    result       = self._run_task(task, context)
                    task.result  = result
                    task.status  = TaskStatus.COMPLETED
                    elapsed_ms   = int((time.perf_counter() - t0) * 1000)
                    self._status(
                        f"  ✅ [{task.id}] 完成  elapsed_ms={elapsed_ms}"
                    )
                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.error  = str(e)
                    self._status(f"  ❌ [{task.id}] 失敗：{e}")

        return plan

    def synthesize_final_answer(self, plan: ResearchPlan) -> str:
        """
        把所有已完成任務的結論整合成最終答案（直接呼叫 LLM）。
        若 LLM 失敗則退回 plan.to_summary()。
        """
        import requests as _req

        summary = plan.to_summary()
        if not summary:
            return "（無法整合：所有子任務均失敗）"

        if cfg.EN_DRAFT_PIPELINE:
            prompt = (
                "You are synthesizing a comprehensive academic answer to the following "
                "research question based on structured sub-task conclusions.\n\n"
                f"Research question: {plan.question}\n\n"
                "Sub-task conclusions (in dependency order):\n"
                f"{summary}\n\n"
                "Write a complete, well-structured academic answer that integrates all "
                "sub-task conclusions. Preserve all citations, specific values, and "
                "section headings (## [Direct Paper Evidence], ## [Cross-Literature "
                "Inference], ## [Knowledge Extension and Speculation])."
            )
            system = (
                "You are a professional academic synthesis assistant. "
                "Output only the final answer in English."
            )
        else:
            prompt = (
                "請根據以下子任務結論，針對原始問題整合出完整的學術回答。\n\n"
                f"原始問題：{plan.question}\n\n"
                "子任務結論（按依賴順序排列）：\n"
                f"{summary}\n\n"
                "請整合所有結論，寫出一份完整的學術答案，"
                "保留所有引用標注與具體數值。\n"
                "格式請包含【論文直接依據】、【跨文獻推論】、【知識延伸與推測】三個段落。"
            )
            system = cfg.LLM_SYSTEM_PROMPT

        try:
            resp = _req.post(
                f"{cfg.OLLAMA_BASE_URL}/api/generate",
                json={
                    "model":  cfg.SYNTHESIS_MODEL,
                    "system": system,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_ctx":     cfg.STAGE3_NUM_CTX,
                        "num_predict": -1,
                    },
                },
                timeout=cfg.LLM_TIMEOUT,
            )
            resp.raise_for_status()
            answer = resp.json().get("response", "").strip()
            return answer if answer else summary
        except Exception as e:
            self._status(f"  ⚠️  [plan-execute] 最終整合失敗，退回 to_summary()：{e}")
            return summary

    # ──────────────────────────────────────────────────────────────
    #  Private helpers
    # ──────────────────────────────────────────────────────────────

    def _build_context_for_task(self, task: SubTask, plan: ResearchPlan) -> str:
        """
        為一個任務組裝前置結論作為 LLM 生成時的上下文。
        只包含 depends_on 列表中已 COMPLETED 的任務結論。
        目標 8000–12000 tokens（粗估 1 token ≈ 4 chars → CHAR_BUDGET=40000）。
        """
        if not task.depends_on:
            return ""

        lines       = ["【前置任務結論（供本任務參考）】"]
        total_chars = 0
        CHAR_BUDGET = 40_000

        for dep_id in task.depends_on:
            dep = plan.get_task_by_id(dep_id)
            if not (dep and dep.status == TaskStatus.COMPLETED and dep.result):
                continue
            entry = f"\n## 前置任務 {dep_id}：{dep.question}\n結論：{dep.result}\n"
            if total_chars + len(entry) > CHAR_BUDGET:
                remaining = CHAR_BUDGET - total_chars
                entry     = entry[:remaining] + "\n...[截短]"
            lines.append(entry)
            total_chars += len(entry)
            if total_chars >= CHAR_BUDGET:
                break

        return "\n".join(lines) if len(lines) > 1 else ""

    def _run_task(self, task: SubTask, prior_context: str) -> str:
        """
        對單一子任務執行 RAG：
          Phase A — 用純問題文字 embed + 向量檢索（不混入 prior_context，避免污染向量）
          Phase B — 用「問題 + prior_context」組合後送 LLM 生成（利用前置結論輔助推理）

        對所有論文跑後合併，空結果丟出 RuntimeError。
        """
        from rag.query_embedding_guard import prepare_query_text
        from rag.query_retrieval import _retrieve_nodes, _generate_from_nodes

        retrieve_text = prepare_query_text(task.question)

        # Generation 用 augmented question（含前置結論）
        if prior_context:
            gen_question = f"{task.question}\n\n{prior_context}"
        else:
            gen_question = task.question

        answers = []
        for name, engine in self._paper_engines.items():
            try:
                nodes  = _retrieve_nodes(engine, retrieve_text)
                result = _generate_from_nodes(engine, nodes, gen_question)
                if result:
                    answers.append(f"【{name}】\n{result}")
            except Exception as e:
                self._status(f"  ⚠️  [{task.id}] {name} 查詢失敗：{e}")

        if not answers:
            raise RuntimeError("所有論文均無法查詢")

        return "\n\n".join(answers)
