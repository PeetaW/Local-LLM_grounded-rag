# rag/query_planning.py
# Pipeline Stage 1: paper selection and sub-question planning.
# No retrieval, no answer generation, no translation.

import re
import json
from collections import Counter

import config as cfg


# 共用檔名樣板：ScienceDirect / 期刊 PDF 常見的非識別性片段。
# 問題裡出現這些字（如 "main findings"）不該觸發單篇鎖定。
_FILENAME_BOILERPLATE = {
    "main", "s2.0", "article", "supplement", "supplementary",
    "info", "full", "text", "pdf",
}
_METHOD_QUERY_HINTS = (
    "method", "process", "procedure", "protocol", "synthesis", "synthesize",
    "preparation", "protection", "reaction route", "key step",
)
_COMPARISON_QUERY_HINTS = (
    "compare", "comparison", "across the papers", "different routes",
    "different approaches", "across the literature", "scalability",
    "cost-effectiveness", "trade-off",
)
_OUTCOME_QUERY_HINTS = (
    "therapeutic effect", "therapeutic efficacy", "efficacy", "supporting data",
    "treatment outcome", "antitumor", "anti-tumor", "survival",
)


def _is_comparison_query(question: str) -> bool:
    lower = (question or "").lower()
    return any(term in lower for term in _COMPARISON_QUERY_HINTS)


def _without_redundant_supplements(question: str, paper_names: list) -> list:
    """Keep a parent paper instead of spending a comparison slot on its SI duplicate."""
    if not _is_comparison_query(question):
        return paper_names
    known = {name.casefold() for name in paper_names}
    kept = []
    for name in paper_names:
        parent = re.sub(
            r"(?:[\s_-]?(?:supp(?:lementary)?|si))$",
            "",
            name,
            flags=re.IGNORECASE,
        ).rstrip()
        if parent != name and parent.casefold() in known:
            continue
        kept.append(name)
    return kept


def _add_method_retrieval_facet(
    question: str,
    paper_names: list,
    sub_questions: list[dict],
) -> list[dict]:
    lower = (question or "").lower()
    if (
        not getattr(cfg, "METHOD_RETRIEVAL_FACET_GUARD_ENABLED", False)
        or not any(term in lower for term in _METHOD_QUERY_HINTS)
        or any(term in lower for term in _COMPARISON_QUERY_HINTS)
    ):
        return sub_questions

    existing_text = " ".join(
        str(item.get("sub_q", "")).lower()
        for item in sub_questions
        if isinstance(item, dict)
    )
    if all(term in existing_text for term in ("yield", "temperature", "control")):
        return sub_questions

    specific = list(dict.fromkeys(
        str(item.get("paper", "")).strip()
        for item in sub_questions
        if isinstance(item, dict) and str(item.get("paper", "")).strip() not in {"", "ALL"}
    ))
    paper = specific[0] if len(specific) == 1 else (
        paper_names[0] if len(paper_names) == 1 else "ALL"
    )
    facet = (
        "For the method or protocol in the user question, what exact reactants or reagents, "
        "solvent, catalyst loading, temperature, reaction time, optimized yield, and "
        "control or comparison outcomes are reported?"
    )
    return [*sub_questions, {"paper": paper, "sub_q": facet}]


def _add_outcome_retrieval_facet(
    question: str,
    paper_names: list,
    sub_questions: list[dict],
) -> list[dict]:
    lower = (question or "").lower()
    if (
        not getattr(cfg, "OUTCOME_RETRIEVAL_FACET_GUARD_ENABLED", False)
        or not any(term in lower for term in _OUTCOME_QUERY_HINTS)
    ):
        return sub_questions

    existing = " ".join(
        str(item.get("sub_q", "")).lower()
        for item in sub_questions
        if isinstance(item, dict)
    )
    if "survival" in existing and any(
        term in existing for term in ("tumor", "accumulation", "retention")
    ):
        return sub_questions

    specific = list(dict.fromkeys(
        str(item.get("paper", "")).strip()
        for item in sub_questions
        if isinstance(item, dict) and str(item.get("paper", "")).strip() not in {"", "ALL"}
    ))
    paper = specific[0] if len(specific) == 1 else (
        paper_names[0] if len(paper_names) == 1 else "ALL"
    )
    facet = (
        "What quantitative treatment outcomes are reported, including survival time, "
        "tumor burden, accumulation or retention, and comparisons between treatment "
        "and control groups?"
    )
    return [*sub_questions, {"paper": paper, "sub_q": facet}]


def detect_target_paper(question: str, paper_names: list) -> str | None:
    """
    Deterministic string match — no LLM.
    Scans the question for *distinctive* filename segments that identify a
    specific paper. 共用樣板片段（如 s2.0、main）與出現在多數檔名中的片段會被排除，
    避免問題只是含有 "main" 之類的字就誤鎖到錯誤論文（P3）。
    """
    question_lower = question.lower()

    # 計算每個片段的檔名出現頻率：出現在很多檔名中的片段＝非識別性樣板
    seg_df = Counter()
    for name in paper_names:
        for s in set(name.lower().split("-")):
            seg_df[s] += 1
    df_threshold = max(2, len(paper_names) // 6)

    def _is_distinctive(seg: str) -> bool:
        return (
            len(seg) > 3
            and seg not in _FILENAME_BOILERPLATE
            and seg_df[seg] <= df_threshold
        )

    best_match = None
    best_score = 0
    for name in paper_names:
        segments = [s for s in name.lower().split("-") if _is_distinctive(s)]
        if not segments:
            continue
        matches = sum(1 for seg in segments if seg in question_lower)
        score = matches / len(segments)
        if score > best_score and score >= 0.3:
            best_score = score
            best_match = name

    return best_match


def _keyword_prefilter(question: str, paper_names: list) -> list:
    """
    Pure keyword match against paper metadata — no LLM.
    Drops papers with zero keyword overlap; falls back to full list if none match.
    """
    from rag.metadata_manager import load_metadata

    all_metadata = load_metadata()

    question_lower = question.lower()
    en_words = set(w for w in re.split(r'[\s\-_,./;:!?()\[\]]+', question_lower) if len(w) >= 3)
    zh_chars = re.findall(r'[一-鿿]{2,}', question)
    question_terms = en_words | set(zh_chars)

    if not question_terms:
        return paper_names

    scored = []
    for p in paper_names:
        meta = all_metadata.get(p, {})
        meta_text = " ".join([
            " ".join(meta.get("keywords", [])),
            meta.get("main_topic", ""),
            meta.get("short_desc", ""),
            p,
        ]).lower()
        hits = sum(1 for term in question_terms if term in meta_text)
        if hits > 0:
            scored.append((p, hits))

    if not scored:
        return paper_names

    scored.sort(key=lambda x: x[1], reverse=True)
    result = [p for p, _ in scored]
    print(f"  🔑 關鍵字預篩：{len(paper_names)} 篇 → {len(result)} 篇")
    return result


def select_relevant_papers(question: str, paper_names: list) -> list:
    """
    Use the planning LLM to pick the most relevant papers (max 5).
    Falls back to full list on any failure.
    """
    from rag.llm_client import planning_llm
    from rag.metadata_manager import load_metadata

    candidates = _without_redundant_supplements(question, paper_names)
    all_metadata = load_metadata()
    paper_list_str = "\n".join(
        f"- {p}：{all_metadata.get(p, {}).get('short_desc', '（無描述）')}"
        f"（關鍵字：{', '.join(all_metadata.get(p, {}).get('keywords', [])[:4])}）"
        for p in candidates
    )

    prompt = f"""以下是論文清單，每篇附有簡短描述：
{paper_list_str}

使用者問題：
{question}

請判斷哪幾篇論文最可能包含這個問題的答案，最多選 5 篇。
選擇依據：論文主題、關鍵字是否與問題相關。
寧可多選也不要漏選，但不要把完全不相關的論文也選進來。

只輸出 JSON 陣列，格式如下，不要其他文字：
["論文檔名1", "論文檔名2", ...]"""

    try:
        response = planning_llm.complete(prompt)
        raw = response.text.strip()
        raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        try:
            selected = json.loads(raw)
        except json.JSONDecodeError as je:
            print(f"  ⚠️  論文篩選 JSON 解析失敗（{je}），退回查詢全部論文")
            return candidates

        valid = [p for p in selected if p in candidates] if isinstance(selected, list) else []
        if valid:
            print(f"  📌 篩選出 {len(valid)} 篇相關論文（共 {len(candidates)} 篇）：")
            for p in valid:
                print(f"     - {p[:60]}")
            return valid
        else:
            print(f"  ⚠️  論文篩選結果為空，退回查詢全部論文")
            return candidates

    except Exception as e:
        print(f"  ⚠️  論文篩選失敗（{e}），退回查詢全部論文")
        return candidates


def plan_sub_questions(question: str, paper_names: list) -> list:
    """
    Use the planning LLM to decompose a compound question into sub-questions.
    Returns [{"paper": "<name or ALL>", "sub_q": "<question>"}].
    Falls back to a single ALL query on parse failure.
    """
    from rag.llm_client import planning_llm
    from rag.metadata_manager import load_metadata

    detected_paper = detect_target_paper(question, paper_names)
    if detected_paper:
        hint = (
            f"\n⚠️  系統已確認：使用者的問題明確指向論文\n"
            f"「{detected_paper}」\n"
            f"所有子問題的 paper 欄位必須設為此論文完整檔名，"
            f"絕對禁止使用 ALL。\n"
        )
        print(f"  🎯 字串比對命中：{detected_paper[:50]}...")
    else:
        hint = ""
        print(f"  🔍 未命中特定論文，交由模型判斷")

    all_metadata = load_metadata()
    paper_list_str = "\n".join(
        f"- {p}：{all_metadata.get(p, {}).get('short_desc', '（無描述）')}"
        f"（主題：{all_metadata.get(p, {}).get('main_topic', '')}）"
        for p in paper_names
    )
    print(f"  🤖 子問題拆解使用模型：{planning_llm.model}")

    prompt = f"""你是一個查詢規劃助手。
{hint}
可用的論文清單：
{paper_list_str}

使用者的複合問題：
{question}

規則：
1. 如果問題中明確提到某篇論文的檔名或 paper ID（例如 S1878029613002417），所有子問題的 paper 欄位都必須設為該論文的完整檔名（不含.pdf），絕對不可以設為 ALL。
2. 只有當問題明確要求「比較所有論文」或「所有論文都要查」時，才可以使用 ALL。
3. 子問題請用英文撰寫，使用學術論文常見的詞彙（例如 synthesis procedure, preparation method, reagents used, experimental conditions）。
4. 若問題涉及合成或實驗方法，且不是跨論文比較題，必須額外拆出一個子問題專門詢問具體操作參數，例如：amounts, concentrations, temperature, stirring speed, reaction time。
5. 若問題詢問「重點」「主要發現」「貢獻」「結論」「這篇在說什麼」，
   子問題應包含：
   - What is the main research objective and novelty?
   - What are the key findings and conclusions?
6. 若問題要求跨論文比較，子問題必須保留原始問題的比較面向；不要只問 main objective/novelty。
   例如合成路線比較題應詢問每篇論文是否 reports, reviews, or compares high-level synthetic approaches to the target compound，
   並要求抽出與原問題指定面向相關的高層證據（例如 isotopic enrichment, scalability, cost-effectiveness, safety）。
   不要詢問 exhaustive procedural details、amounts、temperatures、step-by-step route variants，除非使用者明確要求完整實驗條件。
7. 問題若同時要求多個面向（例如 impurities 與 storage conditions、mechanism 與 supporting data、fluoride binding 與 hydrogel formation），每個面向至少要有一個聚焦子問題；不要把它們全部塞進泛泛的 main objective/novelty 問題。
8. 問題若以 since/because/assuming 等前提要求數值或結論，額外建立一個子問題驗證該前提，並查找文獻實際報告的替代事實；前提未驗證前不可直接接受。
以 JSON 陣列回傳，格式如下，只輸出 JSON，不要其他文字：
[
{{"paper": "論文檔名（不含.pdf）或 ALL", "sub_q": "子問題內容"}},
...
]
"""
    for attempt in range(2):
        response = planning_llm.complete(prompt)
        try:
            raw = response.text.strip()
            raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            sub_questions = json.loads(raw)

            deduped = []
            seen = set()
            for sq in sub_questions:
                key = (sq.get("paper", "ALL"), sq.get("sub_q", "").strip())
                if key not in seen:
                    seen.add(key)
                    deduped.append(sq)
            specific_papers = {sq.get("paper") for sq in deduped if sq.get("paper") != "ALL"}
            if set(paper_names).issubset(specific_papers):
                deduped = [sq for sq in deduped if sq.get("paper") != "ALL"]
            sub_questions = _add_method_retrieval_facet(
                question,
                paper_names,
                deduped,
            )
            sub_questions = _add_outcome_retrieval_facet(
                question,
                paper_names,
                sub_questions,
            )

            print(f"  → 子問題內容：{[sq.get('sub_q', '')[:200] for sq in sub_questions]}")
            return sub_questions
        except json.JSONDecodeError:
            print(f"  ⚠️  子問題拆解失敗（第{attempt+1}次），raw 內容：{raw[:200]}")
            if attempt == 0:
                print("       重試中...")

    print("       改為對所有論文問同一問題")
    sub_questions = _add_method_retrieval_facet(
        question,
        paper_names,
        [{"paper": "ALL", "sub_q": question}],
    )
    return _add_outcome_retrieval_facet(question, paper_names, sub_questions)
