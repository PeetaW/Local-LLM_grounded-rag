# scripts/test_query.py
# 從 main.py 拆出來的終端機問答測試腳本
# 用途：在不啟動 FastAPI 的情況下，直接在終端機測試問答
#
# 使用方式：
#   conda activate llm_env
#   cd E:\Projects\rag_project
#   python scripts/test_query.py

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import asyncio

from main import paper_engines, episodic_collection, preference_collection
from rag.query_engine import execute_structured_query
from rag.memory import recall_memories, decide_and_save


def _parse_grounding_score(answer: str) -> float:
    """從答案的品質報告中解析 grounding_score，解析失敗回傳 -1.0。"""
    import re
    match = re.search(r'grounding_score=(\d+\.?\d*)', answer)
    return float(match.group(1)) if match else -1.0


async def query_with_memory(question: str) -> str:
    episodic_context = recall_memories(episodic_collection, question)
    preference_context = recall_memories(preference_collection, question)

    memory_context = ""
    if episodic_context:
        memory_context += f"\n【過去推論結論】\n{episodic_context}\n"
    if preference_context:
        memory_context += f"\n【使用者偏好】\n{preference_context}\n"

    if memory_context:
        print(f"  💭 找到相關記憶，注入 context")

    answer = execute_structured_query(question, paper_engines, memory_context)

    grounding_score = _parse_grounding_score(answer)
    decide_and_save(question, answer, grounding_score, False, False, episodic_collection, preference_collection)
    print(f"  💾 記憶已儲存（episodic：{episodic_collection.count()} 筆）")

    return answer


# ── 在這裡修改要測試的問題 ────────────────────────────
#
# Tier 1：單篇深挖（精準檢索）
#   期望：Stage 3 整理出具體試劑、比例、條件，不外推
#
# Tier 2：跨篇比較（子問題分解 + Stage 3 多篇合成）
#   期望：自動拆成多子問題並行查詢，最後整合出比較表
#   → 最能展示系統核心能力
#
# Tier 3：推論性問題（reasoning mode + Stage 5 驗證）
#   期望：允許跨文獻推論，但標注信心層次，Stage 5 驗證根據
#
# Tier 4：誠實性測試（邊界測試）
#   期望：資料庫無此主題時，明確回答「未涉及」而非編造
#
questions = [
    # ── Tier 1：單篇深挖 ──────────────────────────────
    # 驗證重構後 Phase A（bge-m3 並行檢索）+ Phase B（gemma4 串行生成）是否正常
    (
        "glycine 修飾 nZVI 的合成步驟中，具體使用了哪些試藥、"
        "比例與反應條件？gelatin aerogel 在其中扮演什麼角色？"
        "請盡量引用論文中的具體數值。"
    ),

    # ── Tier 2：跨篇比較 ──────────────────────────────
    # (
    #     "目前資料庫的文獻中合成 nZVI 的方法有哪幾種？"
    #     "比較胺基酸輔助法（glycine、arginine 等）與 rectorite 載體法的差異："
    #     "哪種方式對粒徑控制更有效？各自的優點與限制是什麼？"
    # ),

    # ── Tier 3：推論性問題 ────────────────────────────
    # (
    #     "如果目標是合成粒徑小於 50nm 且具有高反應活性的 nZVI，"
    #     "根據資料庫中的論文，你推薦什麼合成策略？"
    #     "具體參數（還原劑濃度、pH、溫度等）應該如何設定？"
    #     "這個建議的信心程度如何，有哪些部分是推論而非論文直接記載？"
    # ),

    # ── Tier 4：誠實性測試（邊界測試）───────────────
    # (
    #     "nZVI 在實際土壤修復工程中對重金屬（如鉛、鎘）的去除效率如何？"
    #     "在哪些土壤類型與 pH 條件下效果最好？"
    #     "有沒有現場（field scale）實驗的數據？"
    # ),
]
# ────────────────────────────────────────────────────

total_start = time.time()

for i, question in enumerate(questions, 1):
    print(f"\n{'='*65}")
    print(f"[問題 {i}/{len(questions)}] {question}")
    print(f"{'='*65}")

    q_start = time.time()
    response = asyncio.run(query_with_memory(question))
    q_elapsed = time.time() - q_start

    minutes = int(q_elapsed // 60)
    seconds = int(q_elapsed % 60)
    print(f"\n⏱ 本題耗時：{minutes}分{seconds}秒")

total_elapsed = time.time() - total_start
total_min = int(total_elapsed // 60)
total_sec = int(total_elapsed % 60)

print(f"\n{'='*65}")
print(f"✓ 完成！總耗時：{total_min}分{total_sec}秒")
print(f"{'='*65}")