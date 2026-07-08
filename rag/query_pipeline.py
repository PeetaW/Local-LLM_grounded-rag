# rag/query_pipeline.py
# Public entry point for the query pipeline.
# Coordinates all pipeline stages; delegates implementation to sub-modules.
#
# Public API (stable):
#   execute_structured_query(...)        → str
#   execute_structured_query_stream(...) → Generator[str, None, None]

import json
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


def _comparison_json_from_knowledge_base(knowledge_base: str) -> dict:
    text = (knowledge_base or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    comparison = data.get("comparison_json") if isinstance(data, dict) else None
    return comparison if isinstance(comparison, dict) else {}


def _stage4_answer_validation_issues(answer: str, knowledge_base: str, question: str) -> str:
    if not getattr(cfg, "STAGE4_ANSWER_VALIDATION_ENABLED", False):
        return ""
    comparison = _comparison_json_from_knowledge_base(knowledge_base)
    if not comparison:
        return ""

    lower = (answer or "").lower()
    issues = []
    if any(marker in lower for marker in ("no relevant query results", "no paper data", "please provide the query results")):
        issues.append(
            "Stage4Validation | False no-data answer | The Known Facts List contains comparison_json with paper evidence; do not ask the user to provide data."
        )

    for route in comparison.get("direct_routes", []):
        if not isinstance(route, dict):
            continue
        phrase = str(route.get("route_phrase", "")).strip()
        if phrase and phrase.lower() not in lower:
            issues.append(f"Stage4Validation | Missing direct route phrase | Include exactly: {phrase}")

    for review in comparison.get("review_comparison_sources", []):
        if not isinstance(review, dict):
            continue
        source = str(review.get("source", "")).strip()
        if source and source.lower() not in lower:
            issues.append(f"Stage4Validation | Missing review/comparison source | Include {source} as review/comparison evidence, not as a direct route.")

    dimensions = comparison.get("dimensions") if isinstance(comparison.get("dimensions"), dict) else {}
    dim_terms = {
        "isotopic_enrichment": ("isotopic enrichment", "10b", "boron-10"),
        "scalability": ("scalability", "scalable", "scale-up", "route efficiency", "practical synthesis"),
        "cost_effectiveness": ("cost-effectiveness", "cost effectiveness", "cost", "lower process burden", "fewer protecting groups"),
        "safety": ("safety", "safe", "risk"),
    }
    for key, terms in dim_terms.items():
        item = dimensions.get(key)
        if not isinstance(item, dict) or not item.get("requested"):
            continue
        if item.get("evidence_found") and not any(term in lower for term in terms):
            issues.append(f"Stage4Validation | Missing requested dimension | Cover {key} using the comparison_json text and sources.")
        if key in ("scalability", "cost_effectiveness") and any(
            marker in lower for marker in ("did not provide", "does not provide", "not provide", "no data", "missing")
        ):
            issues.append(
                f"Stage4Validation | Wrong missing-evidence claim | Do not say {key} is missing when review/comparison evidence supports a qualitative comparison."
            )

    if "central trade-off" not in lower and "core trade-off" not in lower:
        issues.append("Stage4Validation | Missing central trade-off | Add one concise Central trade-off sentence using the requested dimensions.")

    return "VERIFY_FAIL\n" + "\n".join(f"- {issue}" for issue in issues) if issues else ""


def _rewrite_stage4_if_needed(full_text: str, knowledge_base: str, question: str, on_status=None) -> str:
    def _status(msg):
        if on_status:
            on_status(msg)
        else:
            print(msg)

    current = full_text
    retries = getattr(cfg, "STAGE4_ANSWER_REWRITE_RETRIES", 1)
    for attempt in range(retries):
        issues = _stage4_answer_validation_issues(current, knowledge_base, question)
        if not issues:
            return current
        preview = "; ".join(line[2:] for line in issues.splitlines()[1:3])
        _status(f"  🔁 [stage4-validator] rewrite {attempt + 1}/{retries}: {preview}")
        rewritten = _verifier.correct(current, knowledge_base, issues, on_status=on_status)
        if not rewritten or rewritten == current:
            return current
        current = rewritten
    return current


# ══════════════════════════════════════════════════════════════════
#  Non-streaming entry point
# ══════════════════════════════════════════════════════════════════

def execute_structured_query(
    question: str,
    paper_engines: dict,
    memory_context: str = "",
    on_status=None,
    on_artifact=None,
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
    ordered_results = run_subqueries_parallel(valid_tasks, prefilled, on_status=_status)

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
    if on_artifact:
        on_artifact("knowledge_base", knowledge_base)
    _status(f"[synthesis] 完成 elapsed_ms={int((time.perf_counter()-t2)*1000)}")

    # ── Stage 3.5: 可答性 gate（Phase 2：接路由）──────────────────────
    # NOT_ANSWERABLE→硬棄答（跳 Stage 4-7）；PARTIAL→軟警告橫幅；ANSWERABLE→正常。
    gate_abstain, gate_notice = False, ""
    if cfg.ANSWERABILITY_GATE_ENABLED and rag_found_anything:
        from rag.answerability import assess_answerability, gate_route
        _ans = assess_answerability(question, knowledge_base)
        gate_abstain, gate_notice = gate_route(_ans["verdict"])
        _kb_head = " ".join((knowledge_base or "")[:240].split())
        _status(f"[answerability] verdict={_ans['verdict']} abstain={gate_abstain} "
                f"kb_chars={len(knowledge_base or '')} kb_head={_kb_head} reason={_ans['reason'][:160]}")

    # ── Stage 4: LLM synthesis ───────────────────────────────────────
    t3 = time.perf_counter()
    if gate_abstain:
        _status("  🚪 [answerability] NOT_ANSWERABLE → 誠實棄答，跳過生成")
        full_text = gate_notice
    else:
        if not rag_found_anything:
            _status("  ℹ️  RAG 資料庫未找到相關內容，切換至模型推理模式...")
            fallback_notice = _FALLBACK_NOTICE
            synthesis_prompt = build_fallback_prompt(
                question, _build_memory_section(memory_context, is_fallback=True)
            )
        else:
            fallback_notice = ""  # 軟警告橫幅在翻譯後才加（見 Stage 7 後），避免中文橫幅被送進翻譯
            lang = "en" if cfg.EN_DRAFT_PIPELINE else "zh"
            final_translation_enabled = getattr(cfg, "FINAL_TRANSLATION_ENABLED", True)
            print(f"  {'🧠 推理' if cfg.REASONING_MODE == 'reasoning' else '📋 嚴格'}模式"
                  f"（{cfg.REASONING_MODE}）  target_paper_detected={bool(detected)}"
                  f"  streaming_mode=False"
                  f"  translation_applied={cfg.EN_DRAFT_PIPELINE and final_translation_enabled}")
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

    if rag_found_anything and not gate_abstain:
        full_text = _rewrite_stage4_if_needed(full_text, knowledge_base, question, on_status=on_status)

    # ── Stage 5: Verification ────────────────────────────────────────
    if cfg.VERIFY_ENABLED and rag_found_anything and not gate_abstain:
        t4 = time.perf_counter()
        _status("\n  🔍 [verification] Stage 5: 邏輯自洽驗證中...")
        full_text = _verifier.verify_and_correct(
            draft_answer=full_text, knowledge_base=knowledge_base, on_status=on_status,
        )
        _status(f"[verification] 完成 elapsed_ms={int((time.perf_counter()-t4)*1000)}")

    # ── Stage 6: Citation grounding ──────────────────────────────────
    nli_report = ""
    if cfg.CITATION_GROUNDING_ENABLED and rag_found_anything and not gate_abstain:
        t5 = time.perf_counter()
        _status("\n[grounding] 開始")
        try:
            full_text, nli_report = run_grounding_check(
                full_text, sub_answers, knowledge_base,
                question=question, paper_engines_to_use=paper_engines_to_use,
                on_status=_status,
            )
            print(nli_report)
        except Exception as e:
            _status(f"  ⚠️  答案品質審查失敗（不影響主流程）：{e}")
        _status(f"[grounding] 完成 elapsed_ms={int((time.perf_counter()-t5)*1000)}")

    if on_artifact:
        on_artifact("answer_for_judge", full_text)

    # ── Stage 7: Translation ─────────────────────────────────────────
    # 棄答橫幅本來就是中文，不需翻譯（也避免 gemma 翻一段固定中文）。
    if (
        cfg.EN_DRAFT_PIPELINE
        and getattr(cfg, "FINAL_TRANSLATION_ENABLED", True)
        and rag_found_anything
        and not gate_abstain
    ):
        t6 = time.perf_counter()
        _status("\n[translation] 開始")
        full_text = translate_to_traditional_chinese(full_text, on_status=on_status)
        _status(f"[translation] 完成 elapsed_ms={int((time.perf_counter()-t6)*1000)}")

    # PARTIAL 軟警告：翻譯後才加（橫幅本身已是中文，不送進翻譯）。棄答時 full_text 已是橫幅，不重複。
    if gate_notice and not gate_abstain:
        full_text = gate_notice + full_text

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
    retrieval_msgs = []
    ordered_results = run_subqueries_parallel(valid_tasks, prefilled, on_status=retrieval_msgs.append)
    for msg in retrieval_msgs:
        yield f"[STATUS] {msg}\n"

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

    # ── Stage 3.5: 可答性 gate（Phase 2：接路由）──────────────────────
    gate_abstain, gate_notice = False, ""
    if cfg.ANSWERABILITY_GATE_ENABLED and rag_found_anything:
        from rag.answerability import assess_answerability, gate_route
        _ans = assess_answerability(question, knowledge_base)
        gate_abstain, gate_notice = gate_route(_ans["verdict"])
        yield f"[STATUS] [answerability] verdict={_ans['verdict']} abstain={gate_abstain}\n"

    # ── Stage 4: LLM synthesis ───────────────────────────────────────
    t3 = time.perf_counter()
    if gate_abstain:
        yield "[STATUS] 🚪 [answerability] NOT_ANSWERABLE → 誠實棄答，跳過生成\n"
        yield gate_notice
        full_text = gate_notice
    else:
        if not rag_found_anything:
            yield "[STATUS] ⚠️ RAG 未找到相關內容，切換至模型知識推理...\n"
            fallback_notice = _FALLBACK_NOTICE
            synthesis_prompt = build_fallback_prompt(
                question, _build_memory_section(memory_context, is_fallback=True)
            )
        else:
            fallback_notice = ""  # PARTIAL 軟警告在翻譯後才加，避免中文橫幅被送進翻譯
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

    if rag_found_anything and not gate_abstain:
        rewrite_msgs = []
        corrected = _rewrite_stage4_if_needed(
            full_text, knowledge_base, question, on_status=lambda msg: rewrite_msgs.append(msg)
        )
        for msg in rewrite_msgs:
            yield f"[STATUS] {msg.strip()}\n"
        if corrected != full_text:
            yield "\n\n---\n"
            yield corrected
            full_text = corrected

    # ── Stage 5: Verification ────────────────────────────────────────
    if cfg.VERIFY_ENABLED and rag_found_anything and not gate_abstain:
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
    if cfg.CITATION_GROUNDING_ENABLED and rag_found_anything and not gate_abstain:
        t5 = time.perf_counter()
        yield "[STATUS] [grounding] 開始\n"
        try:
            grounding_msgs = []
            full_text, nli_report = run_grounding_check(
                full_text, sub_answers, knowledge_base,
                question=question, paper_engines_to_use=paper_engines_to_use,
                on_status=lambda msg: grounding_msgs.append(msg),
            )
            for msg in grounding_msgs:
                yield f"[STATUS] {msg.strip()}\n"
        except Exception as e:
            nli_report = f"\n\n⚠️ 答案品質審查失敗：{e}"
        yield f"[STATUS] [grounding] 完成 elapsed_ms={int((time.perf_counter()-t5)*1000)}\n"

    # ── Stage 7: Translation ─────────────────────────────────────────
    if (
        cfg.EN_DRAFT_PIPELINE
        and getattr(cfg, "FINAL_TRANSLATION_ENABLED", True)
        and rag_found_anything
        and not gate_abstain
    ):
        t6 = time.perf_counter()
        yield "[STATUS] 🌏 [translation] 翻譯英文答案為繁體中文...\n"
        translated = translate_to_traditional_chinese(full_text, on_status=on_status)
        if translated != full_text:
            yield "\n\n---\n🌏 **繁體中文最終版本：**\n\n"
            yield translated
            full_text = translated
        yield f"[STATUS] [translation] 完成 elapsed_ms={int((time.perf_counter()-t6)*1000)}\n"

    # PARTIAL 軟警告：翻譯後才加（橫幅已是中文，不送進翻譯）。棄答時 full_text 已是橫幅，不重複。
    if gate_notice and not gate_abstain:
        yield "\n\n---\n"
        yield gate_notice
        full_text = gate_notice + full_text

    if nli_report:
        yield nli_report

    yield f"[STATUS] [pipeline] 完成 total_elapsed_ms={int((time.perf_counter()-t0)*1000)}\n"
