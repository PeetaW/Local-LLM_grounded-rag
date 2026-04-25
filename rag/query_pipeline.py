# rag/query_pipeline.py
# Public entry point for the query pipeline.
# Coordinates all pipeline stages; delegates implementation to sub-modules.
#
# Public API (stable):
#   execute_structured_query(...)        → str
#   execute_structured_query_stream(...) → Generator[str, None, None]

import time

from llama_index.core import Settings

import config as cfg
from rag.knowledge_synthesizer import KnowledgeSynthesizer
from rag.answer_verifier import AnswerVerifier
from rag.query_planning import detect_target_paper, _keyword_prefilter, select_relevant_papers, plan_sub_questions
from rag.query_retrieval import build_subquery_tasks, run_subqueries_parallel, is_empty_result, extract_paper_name
from rag.query_grounding_flow import run_grounding_check
from rag.query_translation import translate_to_traditional_chinese
from rag.query_prompts import build_synthesis_prompt, build_fallback_prompt

_synthesizer = KnowledgeSynthesizer()
_verifier    = AnswerVerifier()

_FALLBACK_NOTICE = (
    "⚠️ **資料來源說明**：本地學術文獻資料庫中未找到與此問題直接相關的內容。"
    "以下回答來自模型自身知識，非論文原文，請謹慎參考並自行查證。\n\n"
)


def _build_memory_section(memory_context: str, is_fallback: bool) -> str:
    if not memory_context:
        return ""
    if is_fallback:
        return "【相關歷史問答記憶，僅供參考】\n" + memory_context + "\n"
    return "---\n【相關歷史問答記憶，僅供參考】" + memory_context


# ══════════════════════════════════════════════════════════════════
#  Non-streaming entry point
# ══════════════════════════════════════════════════════════════════

def execute_structured_query(
    question: str,
    paper_engines: dict,
    memory_context: str = "",
    on_status=None,
) -> str:
    """
    Full query pipeline (non-streaming).
    Stages: planning → retrieval → synthesis → LLM → verification → grounding → translation
    """
    def _status(msg):
        if on_status:
            on_status(msg)
        else:
            print(msg)

    t0 = time.perf_counter()
    all_paper_names = list(paper_engines.keys())

    # ── Stage 1: Planning ────────────────────────────────────────────
    _status("\n[planning] 開始")
    detected = detect_target_paper(question, all_paper_names)
    if cfg.REVIEW_MODE:
        _status("\n  📖 REVIEW_MODE 已啟用，使用全部論文，跳過篩選")
        paper_names = all_paper_names
        paper_engines_to_use = paper_engines
    elif detected:
        paper_names = all_paper_names
        paper_engines_to_use = paper_engines
    else:
        _status("\n  🔎 先篩選相關論文...")
        prefiltered = _keyword_prefilter(question, all_paper_names)
        paper_names = select_relevant_papers(question, prefiltered)
        paper_engines_to_use = {k: v for k, v in paper_engines.items() if k in paper_names}

    _status("\n  📋 拆解子問題中...")
    sub_questions = plan_sub_questions(question, paper_names)
    _status(f"  → 拆出 {len(sub_questions)} 個子問題")
    _status(f"[planning] 完成 paper_count={len(paper_names)} subquery_count={len(sub_questions)} "
            f"elapsed_ms={int((time.perf_counter()-t0)*1000)}")

    # ── Stage 2: Retrieval ───────────────────────────────────────────
    t1 = time.perf_counter()
    _status(f"\n[retrieval] 開始")
    _status(f"\n  ⚡ 並行檢索 {len(sub_questions)} 個子問題中（workers={cfg.SUBQUERY_MAX_WORKERS}）...")
    valid_tasks, prefilled = build_subquery_tasks(sub_questions, paper_engines_to_use, paper_engines)
    ordered_results = run_subqueries_parallel(valid_tasks, prefilled)

    sub_answers = []
    rag_found_anything = False
    for label, result in ordered_results:
        sub_answers.append(f"{label}\n{result}")
        if not is_empty_result(result):
            rag_found_anything = True
        _status(f"\n  ── {label} 回覆 ──\n  {result[:200]}")

    _status(f"\n[retrieval] 完成 rag_found={rag_found_anything} "
            f"elapsed_ms={int((time.perf_counter()-t1)*1000)}")

    # ── Stage 3: Knowledge synthesis (distillation) ──────────────────
    t2 = time.perf_counter()
    _status("\n  🔗 綜合所有子答案中...")
    if cfg.SYNTHESIS_ENABLED and rag_found_anything:
        _status("\n  🧪 [synthesis] 知識蒸餾中...")
        synthesis_chunks = [
            {"text": ans, "source": extract_paper_name(ans, f"retrieved_chunk_{i}")}
            for i, ans in enumerate(sub_answers)
        ]
        knowledge_base = _synthesizer.synthesize(
            chunks=synthesis_chunks, query=question, on_status=on_status,
        )
    else:
        knowledge_base = "\n\n".join(sub_answers)
    _status(f"[synthesis] 完成 elapsed_ms={int((time.perf_counter()-t2)*1000)}")

    # ── Stage 4: LLM synthesis ───────────────────────────────────────
    t3 = time.perf_counter()
    if not rag_found_anything:
        _status("  ℹ️  RAG 資料庫未找到相關內容，切換至模型推理模式...")
        fallback_notice = _FALLBACK_NOTICE
        synthesis_prompt = build_fallback_prompt(
            question, _build_memory_section(memory_context, is_fallback=True)
        )
    else:
        fallback_notice = ""
        lang = "en" if cfg.EN_DRAFT_PIPELINE else "zh"
        print(f"  {'🧠 推理' if cfg.REASONING_MODE == 'reasoning' else '📋 嚴格'}模式"
              f"（{cfg.REASONING_MODE}）  target_paper_detected={bool(detected)}"
              f"  streaming_mode=False  translation_applied={cfg.EN_DRAFT_PIPELINE}")
        synthesis_prompt = build_synthesis_prompt(
            knowledge_base, question,
            _build_memory_section(memory_context, is_fallback=False),
            cfg.REASONING_MODE, lang,
        )

    print("\n 最終綜合回答（Stage 4 初稿）：")
    full_text = fallback_notice
    for chunk in Settings.llm.stream_complete(synthesis_prompt):
        print(chunk.delta, end="", flush=True)
        full_text += chunk.delta
    print("\n")
    _status(f"[synthesis-llm] 完成 elapsed_ms={int((time.perf_counter()-t3)*1000)}")

    # ── Stage 5: Verification ────────────────────────────────────────
    if cfg.VERIFY_ENABLED and rag_found_anything:
        t4 = time.perf_counter()
        _status("\n  🔍 [verification] Stage 5: 邏輯自洽驗證中...")
        full_text = _verifier.verify_and_correct(
            draft_answer=full_text, knowledge_base=knowledge_base, on_status=on_status,
        )
        _status(f"[verification] 完成 elapsed_ms={int((time.perf_counter()-t4)*1000)}")

    # ── Stage 6: Citation grounding ──────────────────────────────────
    nli_report = ""
    if cfg.CITATION_GROUNDING_ENABLED and rag_found_anything:
        t5 = time.perf_counter()
        _status("\n[grounding] 開始")
        try:
            full_text, nli_report = run_grounding_check(
                full_text, sub_answers, knowledge_base, on_status=_status,
            )
            print(nli_report)
        except Exception as e:
            _status(f"  ⚠️  答案品質審查失敗（不影響主流程）：{e}")
        _status(f"[grounding] 完成 elapsed_ms={int((time.perf_counter()-t5)*1000)}")

    # ── Stage 7: Translation ─────────────────────────────────────────
    if cfg.EN_DRAFT_PIPELINE and rag_found_anything:
        t6 = time.perf_counter()
        _status("\n[translation] 開始")
        full_text = translate_to_traditional_chinese(full_text, on_status=on_status)
        _status(f"[translation] 完成 elapsed_ms={int((time.perf_counter()-t6)*1000)}")

    if nli_report:
        full_text += nli_report

    _status(f"[pipeline] 完成 total_elapsed_ms={int((time.perf_counter()-t0)*1000)}")
    return full_text


# ══════════════════════════════════════════════════════════════════
#  Streaming entry point
# ══════════════════════════════════════════════════════════════════

def execute_structured_query_stream(
    question: str,
    paper_engines: dict,
    memory_context: str = "",
    on_status=None,
):
    """
    Streaming generator version of execute_structured_query.
    Yields two token types:
      [STATUS] prefix → progress message (rendered as blockquote by api.py)
      other           → LLM output tokens written directly to the response
    """
    t0 = time.perf_counter()
    all_paper_names = list(paper_engines.keys())

    # ── Stage 1: Planning ────────────────────────────────────────────
    detected = detect_target_paper(question, all_paper_names)
    if cfg.REVIEW_MODE:
        yield "[STATUS] 📖 REVIEW_MODE 已啟用，使用全部論文...\n"
        paper_names = all_paper_names
        paper_engines_to_use = paper_engines
    elif detected:
        paper_names = all_paper_names
        paper_engines_to_use = paper_engines
    else:
        yield "[STATUS] 🔎 篩選相關論文中...\n"
        prefiltered = _keyword_prefilter(question, all_paper_names)
        paper_names = select_relevant_papers(question, prefiltered)
        paper_engines_to_use = {k: v for k, v in paper_engines.items() if k in paper_names}
        yield f"[STATUS] 📌 已選出 {len(paper_names)} 篇相關論文\n"

    yield "[STATUS] 📋 拆解子問題中...\n"
    sub_questions = plan_sub_questions(question, paper_names)
    yield f"[STATUS] → 拆出 {len(sub_questions)} 個子問題，開始檢索...\n"
    yield (f"[STATUS] [planning] 完成 paper_count={len(paper_names)} "
           f"subquery_count={len(sub_questions)} "
           f"elapsed_ms={int((time.perf_counter()-t0)*1000)}\n")

    # ── Stage 2: Retrieval ───────────────────────────────────────────
    t1 = time.perf_counter()
    sub_answers = []
    rag_found_anything = False
    yield f"[STATUS] ⚡ 並行檢索 {len(sub_questions)} 個子問題中（workers={cfg.SUBQUERY_MAX_WORKERS}）...\n"
    valid_tasks, prefilled = build_subquery_tasks(sub_questions, paper_engines_to_use, paper_engines)
    ordered_results = run_subqueries_parallel(valid_tasks, prefilled)

    for label, result in ordered_results:
        sub_answers.append(f"{label}\n{result}")
        if not is_empty_result(result):
            rag_found_anything = True
        preview = result[:120].replace("\n", " ")
        yield f"[STATUS] {label} → {preview}...\n"

    yield (f"[STATUS] [retrieval] 完成 rag_found={rag_found_anything} "
           f"elapsed_ms={int((time.perf_counter()-t1)*1000)}\n")

    # ── Stage 3: Knowledge synthesis (distillation) ──────────────────
    t2 = time.perf_counter()
    if cfg.SYNTHESIS_ENABLED and rag_found_anything:
        yield "[STATUS] 🧪 [synthesis] 知識蒸餾中...\n"
        synthesis_chunks = [
            {"text": ans, "source": extract_paper_name(ans, f"retrieved_chunk_{i}")}
            for i, ans in enumerate(sub_answers)
        ]
        knowledge_base = _synthesizer.synthesize(
            chunks=synthesis_chunks, query=question, on_status=on_status,
        )
        yield "[STATUS] 📋 事實清單已整理完成\n"
    else:
        knowledge_base = "\n\n".join(sub_answers)
    yield f"[STATUS] [synthesis] 完成 elapsed_ms={int((time.perf_counter()-t2)*1000)}\n"

    # ── Stage 4: LLM synthesis ───────────────────────────────────────
    t3 = time.perf_counter()
    if not rag_found_anything:
        yield "[STATUS] ⚠️ RAG 未找到相關內容，切換至模型知識推理...\n"
        fallback_notice = _FALLBACK_NOTICE
        synthesis_prompt = build_fallback_prompt(
            question, _build_memory_section(memory_context, is_fallback=True)
        )
    else:
        fallback_notice = ""
        lang = "en" if cfg.EN_DRAFT_PIPELINE else "zh"
        if cfg.REASONING_MODE == "reasoning":
            yield "[STATUS] 🧠 推理模式，LLM 綜合推論中...\n"
        else:
            yield "[STATUS] 📋 嚴格模式，LLM 整理論文內容中...\n"
        synthesis_prompt = build_synthesis_prompt(
            knowledge_base, question,
            _build_memory_section(memory_context, is_fallback=False),
            cfg.REASONING_MODE, lang,
        )

    if fallback_notice:
        yield fallback_notice
    full_text = fallback_notice
    for chunk in Settings.llm.stream_complete(synthesis_prompt):
        yield chunk.delta
        full_text += chunk.delta
    yield f"\n[STATUS] [synthesis-llm] 完成 elapsed_ms={int((time.perf_counter()-t3)*1000)}\n"

    # ── Stage 5: Verification ────────────────────────────────────────
    if cfg.VERIFY_ENABLED and rag_found_anything:
        t4 = time.perf_counter()
        yield "[STATUS] 🔍 [verification] Stage 5: 邏輯自洽驗證中...\n"
        corrected = _verifier.verify_and_correct(
            draft_answer=full_text, knowledge_base=knowledge_base, on_status=on_status,
        )
        if corrected != full_text:
            yield "\n\n---\n📝 **已根據邏輯自洽驗證修正如下：**\n\n"
            yield corrected
            full_text = corrected
        else:
            yield "[STATUS] ✅ [verification] 邏輯驗證通過（VERIFY_PASS），答案無需修正\n"
        yield f"[STATUS] [verification] 完成 elapsed_ms={int((time.perf_counter()-t4)*1000)}\n"

    # ── Stage 6: Citation grounding ──────────────────────────────────
    nli_report = ""
    if cfg.CITATION_GROUNDING_ENABLED and rag_found_anything:
        t5 = time.perf_counter()
        yield "[STATUS] [grounding] 開始\n"
        try:
            grounding_msgs = []
            full_text, nli_report = run_grounding_check(
                full_text, sub_answers, knowledge_base,
                on_status=lambda msg: grounding_msgs.append(msg),
            )
            for msg in grounding_msgs:
                yield f"[STATUS] {msg.strip()}\n"
        except Exception as e:
            nli_report = f"\n\n⚠️ 答案品質審查失敗：{e}"
        yield f"[STATUS] [grounding] 完成 elapsed_ms={int((time.perf_counter()-t5)*1000)}\n"

    # ── Stage 7: Translation ─────────────────────────────────────────
    if cfg.EN_DRAFT_PIPELINE and rag_found_anything:
        t6 = time.perf_counter()
        yield "[STATUS] 🌏 [translation] 翻譯英文答案為繁體中文...\n"
        translated = translate_to_traditional_chinese(full_text, on_status=on_status)
        if translated != full_text:
            yield "\n\n---\n🌏 **繁體中文最終版本：**\n\n"
            yield translated
            full_text = translated
        yield f"[STATUS] [translation] 完成 elapsed_ms={int((time.perf_counter()-t6)*1000)}\n"

    if nli_report:
        yield nli_report

    yield f"[STATUS] [pipeline] 完成 total_elapsed_ms={int((time.perf_counter()-t0)*1000)}\n"
