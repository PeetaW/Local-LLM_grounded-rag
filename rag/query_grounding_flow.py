# rag/query_grounding_flow.py
# Pipeline Stage 6: citation grounding check and fallback correction.
# Parses answer sections, runs NLI, and re-prompts the LLM when evidence is weak.

import re

import config as cfg


def split_into_sentences(text: str) -> list:
    lines = text.split("\n")
    joined_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            joined_lines.append("")
            continue
        is_new_item = bool(re.match(r"^(\*+\s|-\s|\d+\.\s|\#{1,6}\s|\[)", stripped))
        if joined_lines and not is_new_item and joined_lines[-1]:
            joined_lines[-1] += " " + stripped
        else:
            joined_lines.append(stripped)
    text = "\n".join(joined_lines)
    text = re.sub(r"\*\*|##|###|【.*?】", "", text)
    sentences = re.split(r"(?<=[。！？\!\?])\s*|\n+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) >= 20]

    def _is_non_proposition(sentence: str) -> bool:
        if re.match(r"^\[.*\]\s*$", sentence):
            return True
        if re.search(r"[：:]\s*$", sentence) and not re.search(r"[。！？!?]", sentence):
            return True
        if re.match(r"^\d+[\.\s]\s*\S", sentence) and not re.search(r"[。！？!?]", sentence):
            return True
        return bool(re.match(r"^\*\s+第[一二三四五六七八九十百千\d]+[階段步品]", sentence))

    return [sentence for sentence in sentences if not _is_non_proposition(sentence)]


def _cited_sources_in_sentence(sentence: str, known_sources) -> tuple[str, ...]:
    citations = {
        part.strip().lower()
        for value in re.findall(r"\[([^\]]+)\]", sentence or "")
        for part in re.sub(r"^Source:\s*", "", value, flags=re.IGNORECASE).split(",")
    }
    return tuple(str(source) for source in known_sources if str(source).lower() in citations)


def _extract_direct_citation_section(text: str) -> str:
    """
    Extract only the 【論文直接依據】 section from an answer.
    Grounding fallback should only fire on direct-citation content —
    low scores in inference/speculation sections are expected and should not trigger correction.
    Returns empty string if no direct-citation section exists.
    """
    matches = re.findall(
        r'(##[^\n]*(?:論文直接依據|直接依據|直引|Direct.*Evidence)[^\n]*\n[\s\S]*?)(?=\n##|\Z)',
        text
    )
    return "\n\n".join(m.strip() for m in matches)


def _partition_results_by_section(citation_results: list, full_text: str) -> dict:
    """
    Group citation_results by answer section to avoid re-running NLI per section.
    Returns {"direct": [...], "inference": [...], "speculation": [...]}
    Only includes keys where the section exists and has at least one sentence.
    """
    _SECTION_PATTERNS = {
        "direct":      r'(##[^\n]*(?:論文直接依據|直接依據|Direct.*Evidence)[^\n]*\n[\s\S]*?)(?=\n##|\Z)',
        "inference":   r'(##[^\n]*(?:跨文獻推論|Cross.*Literature.*Inference)[^\n]*\n[\s\S]*?)(?=\n##|\Z)',
        "speculation": r'(##[^\n]*(?:知識延伸|Knowledge.*Extension)[^\n]*\n[\s\S]*?)(?=\n##|\Z)',
    }

    partitioned = {}
    for key, pattern in _SECTION_PATTERNS.items():
        matches = re.findall(pattern, full_text)
        if not matches:
            continue
        section_text = "\n\n".join(m.strip() for m in matches)
        section_sent_set = set(split_into_sentences(section_text))
        section_results = [r for r in citation_results if r["sentence"] in section_sent_set]
        if section_results:
            partitioned[key] = section_results

    return partitioned


def _retriever_node_count(retriever) -> int | None:
    """從 hybrid retriever 的子檢索器推得索引的 node 數；推不出回 None。"""
    for r in getattr(retriever, "_retrievers", []):
        idx = getattr(r, "_index", None) or getattr(r, "index", None)
        if idx is not None:
            try:
                return len(idx.docstore.docs)
            except Exception:
                continue
    return None


def _fetch_grounding_chunks(
    question: str,
    paper_engines_to_use: dict,
    sentences: list[str] | None = None,
) -> list[dict]:
    """
    Retrieve raw PDF chunks from the vector index for NLI grounding.
    Uses GROUNDING_TOP_K (higher than SIMILARITY_TOP_K) for broader sentence coverage.
    Returns [] on any failure; caller must fall back to sub_answers.
    """
    from rag.query_embedding_guard import prepare_query_text
    from rag.query_retrieval import _strip_context_summary

    try:
        base_query = prepare_query_text(question)
    except Exception:
        return []

    chunks = []
    for name, engine in paper_engines_to_use.items():
        try:
            query_text = base_query
            if getattr(cfg, "GROUNDING_CITATION_AWARE_ENABLED", False):
                cited_claims = [
                    sentence for sentence in (sentences or [])
                    if _cited_sources_in_sentence(sentence, (name,))
                ]
                if cited_claims:
                    claim_text = "\n".join(re.sub(r"\[[^\]]+\]", "", claim) for claim in cited_claims)
                    try:
                        query_text = prepare_query_text(f"{question}\n{claim_text}")
                    except Exception:
                        pass

            retriever = engine.retriever if hasattr(engine, "retriever") else None
            if retriever is None:
                continue

            # 夾住 top_k：小論文 chunk 數可能 < GROUNDING_TOP_K，
            # BM25(bm25s) 在 k > corpus size 時會報錯，導致整篇被跳過、grounding 誤判為 0（P2）。
            node_count = _retriever_node_count(retriever)
            effective_k = cfg.GROUNDING_TOP_K if node_count is None else min(cfg.GROUNDING_TOP_K, node_count)

            # Temporarily raise top_k; restore via finally to survive retrieval errors
            old_top_k = getattr(retriever, "similarity_top_k", cfg.SIMILARITY_TOP_K)
            retriever.similarity_top_k = effective_k
            for r in getattr(retriever, "_retrievers", []):
                r.similarity_top_k = effective_k

            try:
                nodes = retriever.retrieve(query_text)
            finally:
                retriever.similarity_top_k = old_top_k
                for r in getattr(retriever, "_retrievers", []):
                    r.similarity_top_k = old_top_k

            for nws in nodes:
                chunks.append({
                    "id":     f"{name[:25]}-{nws.node.node_id[:8]}",
                    "text":   _strip_context_summary(nws.node.get_content()),
                    "source": name,
                    "score":  nws.score or 0.0,
                })
        except Exception as e:
            print(f"  ⚠️  [Grounding] {name} raw chunk 取得失敗：{e}")

    chunks.sort(key=lambda c: c["score"], reverse=True)
    return chunks


def _run_grounding_fallback(full_text: str, unsupported: list, knowledge_base: str) -> str | None:
    """
    Send low-evidence statements back to the LLM for re-citation.
    Returns corrected text, or None on failure.
    """
    import requests as _req

    if cfg.EN_DRAFT_PIPELINE:
        bad_sentences = "\n".join(
            f"- {r['sentence']} (confidence: {r['confidence']:.1%})"
            for r in unsupported
        )
        fallback_prompt = (
            f"The following statements lack clear evidence in the papers. "
            f"Please re-verify them against the Known Facts List:\n\n"
            f"{bad_sentences}\n\n"
            f"Known Facts List:\n{knowledge_base}\n\n"
            f"Original answer:\n{full_text}\n\n"
            "For each low-evidence statement, find the corresponding sentence in the original answer and correct it:\n"
            "- If the Facts List has supporting evidence: correct the citation to be precise\n"
            "- If the Facts List has no supporting evidence: mark it as [Unverified] with a brief reason\n"
            "Output the complete corrected answer in English. No preamble or explanation.\n"
            "IMPORTANT: Preserve all section headers exactly as they appear "
            "(## [Direct Paper Evidence], ## [Cross-Literature Inference], ## [Knowledge Extension and Speculation])."
        )
        fallback_system = "You are a professional academic answer editor. Output only the corrected answer in English."
    else:
        bad_sentences = "\n".join(
            f"- {r['sentence']}（信心度：{r['confidence']:.1%}）"
            for r in unsupported
        )
        fallback_prompt = (
            f"以下陳述在論文中找不到明確依據，請根據「已知事實清單」重新確認：\n\n"
            f"{bad_sentences}\n\n"
            f"已知事實清單：\n{knowledge_base}\n\n"
            f"原始答案：\n{full_text}\n\n"
            "請針對上列低依據陳述，在原始答案中找到對應句子並修正：\n"
            "- 若事實清單有對應依據：修正引用標注使其精確\n"
            "- 若事實清單完全沒有依據：標注 [待確認] 並說明原因\n"
            "輸出完整修正後的答案，不要輸出說明或前言。"
        )
        fallback_system = cfg.LLM_SYSTEM_PROMPT

    try:
        resp = _req.post(
            f"{cfg.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": cfg.SYNTHESIS_MODEL,
                "system": fallback_system,
                "prompt": fallback_prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_ctx": 65536, "num_predict": -1},
            },
            timeout=cfg.LLM_TIMEOUT,
        )
        if resp.ok:
            corrected = resp.json().get("response", "").strip()
            return corrected if corrected else None
    except Exception as fe:
        print(f"  ⚠️  [Grounding Fallback] 修正失敗，保留原答案：{fe}")
    return None


def run_grounding_check(
    full_text: str,
    sub_answers: list,
    knowledge_base: str,
    question: str | None = None,
    paper_engines_to_use: dict | None = None,
    grounding_claims: list[str] | None = None,
    on_status=None,
) -> tuple[str, str]:
    """
    Run citation grounding NLI check and optionally apply fallback correction.
    Returns (updated_full_text, nli_report).

    When question + paper_engines_to_use are provided, NLI premises are raw PDF
    chunks retrieved from the vector index (true grounding against source text).
    Otherwise falls back to sub_answers (LLM-generated summaries) as premises.

    on_status is called with progress messages; falls back to print() if None.
    """
    from rag.citation_grounding import (
        check_citation_grounding,
        format_grounding_report,
        compute_grounding_score,
        reset_grounding_timers,
        get_grounding_timers,
        release_nli_gpu,
        selfcorrect_flagged,
        _add_llm_time,
    )

    def _status(msg):
        if on_status:
            on_status(msg)
        else:
            print(msg)

    reset_grounding_timers()
    sentences = list(grounding_claims) if grounding_claims else split_into_sentences(full_text)

    if question and paper_engines_to_use:
        _status("  🔍 執行答案品質審查（對象：PDF raw chunks）...")
        raw_chunks = _fetch_grounding_chunks(question, paper_engines_to_use, sentences)
        if raw_chunks:
            chunks = raw_chunks
            citation_aware = getattr(cfg, "GROUNDING_CITATION_AWARE_ENABLED", False)
            _status(
                f"  → 取得 {len(chunks)} 個 raw chunks（top_k={cfg.GROUNDING_TOP_K}, "
                f"citation_aware={citation_aware}）"
            )
        else:
            _status("  ⚠️  raw chunk 取得失敗，改用 sub_answers 作為比對基準")
            chunks = [
                {"id": f"CHUNK-{i:03d}", "text": ans}
                for i, ans in enumerate(sub_answers)
            ]
    else:
        _status("  🔍 執行答案品質審查...")
        chunks = [
            {"id": f"CHUNK-{i:03d}", "text": ans}
            for i, ans in enumerate(sub_answers)
        ]

    citation_results = check_citation_grounding(sentences, chunks)

    partitioned = _partition_results_by_section(citation_results, full_text)
    direct_results = partitioned.get("direct", [])
    direct_score = compute_grounding_score(direct_results) if direct_results else 1.0

    section_scores = {
        key: {
            "score": compute_grounding_score(results),
            "n_supported": sum(1 for r in results if r["supported"]),
            "n_total": len(results),
        }
        for key, results in partitioned.items()
    }

    grounding_score = compute_grounding_score(citation_results)
    unsupported = [r for r in direct_results if not r["supported"]]

    if getattr(cfg, "GROUNDING_FALLBACK_ENABLED", True) and unsupported and direct_score < 0.8:
        _status(
            f"  🔄 [Grounding Fallback] {len(unsupported)} 個陳述依據不足"
            f"（整體 {grounding_score:.1%}），送回 gemma4 重新引用..."
        )
        import time as _t_mod
        _t0 = _t_mod.perf_counter()
        corrected = _run_grounding_fallback(full_text, unsupported, knowledge_base)
        _add_llm_time(_t_mod.perf_counter() - _t0)
        if corrected:
            full_text = corrected
            _status("  ✅ [Grounding Fallback] gemma4 修正完成，重新執行 grounding 審查...")
            sentences = split_into_sentences(full_text)
            citation_results = check_citation_grounding(sentences, chunks)

    # ── 生成自我修正 loop（便宜版）：一次 batched gemma4 裁定 NLI 標記的直引句 ──
    # 只送「信心極低」的句子（真捏造高發區），borderline 多是 NLI 假陰性 → 不送、省成本。
    _sc_max = getattr(cfg, "SELFCORRECT_ENTAIL_MAX", 0.2)
    confident_unsup = [r for r in unsupported if r.get("confidence", 1.0) < _sc_max]
    if getattr(cfg, "GENERATION_SELFCORRECT_ENABLED", False) and confident_unsup:
        _status(f"  🔧 [Self-Correct] {len(confident_unsup)} 句低信心(<{_sc_max})，送 gemma4 一次裁定...")
        import re as _re_mod, time as _t_mod
        _t0 = _t_mod.perf_counter()
        verdicts = selfcorrect_flagged(confident_unsup, chunks)
        _add_llm_time(_t_mod.perf_counter() - _t0)
        n_keep = n_fix = n_del = 0
        for r, v in zip(confident_unsup, verdicts):
            sent = r["sentence"]
            if v["verdict"] == "CORRECT" and v["fixed"]:
                full_text = full_text.replace(sent, v["fixed"], 1); n_fix += 1
            elif v["verdict"] == "UNVERIFIED":
                full_text = full_text.replace(sent, "", 1); n_del += 1   # 無中生有 → 刪除
            else:
                n_keep += 1  # SUPPORTED → NLI 假陰性，保留不動
        if n_del:  # 清掉刪除後殘留的空 bullet / 空行
            full_text = _re_mod.sub(r'(?m)^\s*[\*\-]\s*$\n?', '', full_text)
            full_text = _re_mod.sub(r'\n{3,}', '\n\n', full_text)
        print(f"[selfcorrect] flagged={len(confident_unsup)} keep={n_keep} fix={n_fix} delete={n_del}", flush=True)
        _status(f"  ✅ [Self-Correct] 保留(假陰性) {n_keep} / 修正 {n_fix} / 刪除幻覺 {n_del}")
        if n_fix or n_del:  # 答案有變才需重算報告
            sentences = split_into_sentences(full_text)
            citation_results = check_citation_grounding(sentences, chunks)
            partitioned = _partition_results_by_section(citation_results, full_text)
            section_scores = {
                key: {"score": compute_grounding_score(results),
                      "n_supported": sum(1 for r in results if r["supported"]),
                      "n_total": len(results)}
                for key, results in partitioned.items()
            }
            grounding_score = compute_grounding_score(citation_results)

    nli_report = format_grounding_report(citation_results, section_scores=section_scores)

    _gt = get_grounding_timers()
    _status(f"[grounding-timing] nli_ms={int(_gt['nli_s'] * 1000)} llm_ms={int(_gt['llm_s'] * 1000)}")

    # ponytail: NLI 跑完把 mDeBERTa 搬下 GPU + 清快取，VRAM 完整讓給 gemma4 翻譯。
    # 下一題 _get_nli_model() 會自動把模型搬回 GPU（~0.5GB 搬移，毫秒級）。
    release_nli_gpu()

    return full_text, nli_report
