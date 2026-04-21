# rag/answer_processor.py
# 串流與非串流兩個分支的統一後處理入口
# 負責：session 寫入 + grounding 審查 + 記憶決策
#
# 抽離自 api.py，消除 streaming / non-streaming 兩個分支的重複邏輯

import re
from rag.citation_grounding import has_speculation_keywords, has_multi_paper_reference
from rag.memory import decide_and_save


def parse_grounding_score(answer: str) -> float:
    """
    從答案末尾的品質報告中解析 grounding_score。
    格式：<!-- grounding_score=0.875 -->
    解析失敗回傳 -1.0。
    """
    match = re.search(r'grounding_score=(\d+\.?\d*)', answer)
    if match:
        return float(match.group(1))
    return -1.0


def post_process_answer(
    question: str,
    answer: str,
    session_id: str,
    session_store: dict,
    session_max_turns: int,
    session_max_count: int,
    episodic_collection,
    preference_collection,
):
    """
    串流與非串流兩個分支的統一後處理入口。
    負責：session 寫入 + grounding 審查 + 記憶決策。
    """
    # ── session 寫入 ──────────────────────────────────
    if session_id not in session_store:
        session_store[session_id] = []
    session_store[session_id].append((question, answer))
    if len(session_store[session_id]) > session_max_turns:
        session_store[session_id] = session_store[session_id][-session_max_turns:]

    # ── session_store 上限控制 ────────────────────────
    if len(session_store) > session_max_count:
        overflow = len(session_store) - session_max_count
        for key in list(session_store.keys())[:overflow]:
            del session_store[key]

    # ── grounding 審查 + 記憶決策 ─────────────────────
    grounding_score = parse_grounding_score(answer)
    is_speculation = has_speculation_keywords(answer)
    is_multi_paper = (
        has_multi_paper_reference(answer) and
        has_multi_paper_reference(question)
    )
    decide_and_save(
        question, answer,
        grounding_score, is_speculation, is_multi_paper,
        episodic_collection, preference_collection,
    )
