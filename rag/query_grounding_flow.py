# rag/query_grounding_flow.py
# Pipeline Stage 6: citation grounding check and fallback correction.
# Parses answer sections, runs NLI, and re-prompts the LLM when evidence is weak.

import re

import config as cfg


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
    from rag.citation_grounding import split_into_sentences

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
    on_status=None,
) -> tuple[str, str]:
    """
    Run citation grounding NLI check and optionally apply fallback correction.
    Returns (updated_full_text, nli_report).

    on_status is called with progress messages; falls back to print() if None.
    """
    from rag.citation_grounding import (
        split_into_sentences,
        check_citation_grounding,
        format_grounding_report,
        compute_grounding_score,
    )

    def _status(msg):
        if on_status:
            on_status(msg)
        else:
            print(msg)

    _status("  🔍 執行答案品質審查...")
    sentences = split_into_sentences(full_text)
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

    if unsupported and direct_score < 0.8:
        _status(
            f"  🔄 [Grounding Fallback] {len(unsupported)} 個陳述依據不足"
            f"（整體 {grounding_score:.1%}），送回 gemma4 重新引用..."
        )
        corrected = _run_grounding_fallback(full_text, unsupported, knowledge_base)
        if corrected:
            full_text = corrected
            _status("  ✅ [Grounding Fallback] gemma4 修正完成，重新執行 grounding 審查...")
            sentences = split_into_sentences(full_text)
            citation_results = check_citation_grounding(sentences, chunks)

    nli_report = format_grounding_report(citation_results, section_scores=section_scores)
    return full_text, nli_report
