# rag/query_prompts.py
# Synthesis prompt builders for all four pipeline variants.
# Pure string construction — no LLM calls, no I/O.
#
# Variants:
#   mode: "reasoning" | "strict"
#   lang: "en" | "zh"
#
# EN mode is used when EN_DRAFT_PIPELINE is enabled (higher accuracy);
# the final answer is translated to Traditional Chinese by query_translation.py.


def build_synthesis_prompt(
    knowledge_base: str,
    question: str,
    memory_section: str,
    mode: str,
    lang: str,
) -> str:
    """
    Build the synthesis prompt for Stage 4 LLM call.
      mode: "reasoning" — three-tier answer with inference and speculation allowed
      mode: "strict"    — citation-only, no cross-paper inference
      lang: "en"        — prompt and output in English (used with EN_DRAFT_PIPELINE)
      lang: "zh"        — prompt and output in Traditional Chinese
    """
    if mode == "reasoning" and lang == "en":
        return _reasoning_en(knowledge_base, question, memory_section)
    if mode == "reasoning" and lang == "zh":
        return _reasoning_zh(knowledge_base, question, memory_section)
    if mode == "strict" and lang == "en":
        return _strict_en(knowledge_base, question, memory_section)
    return _strict_zh(knowledge_base, question, memory_section)


def build_fallback_prompt(question: str, memory_section: str) -> str:
    """
    Prompt used when RAG finds no relevant content.
    Instructs the LLM to answer from its own academic knowledge and flag uncertainty.
    """
    return f"""
使用者的問題：
{question}

本地學術文獻資料庫已進行查詢，但未找到直接相關的文獻資料。

{memory_section}
請根據你自身的學術知識，盡力回答這個問題。
要求：
1. 回答請使用繁體中文，保持學術嚴謹性。
2. 若你對某個細節不確定，請明確說明「此為模型推測，建議查閱原始文獻確認」。
3. 若問題涉及具體數值或實驗參數，請提醒使用者這些數值可能因論文而異。
"""


# ── Private prompt builders ──────────────────────────────────────────────────

def _reasoning_en(knowledge_base: str, question: str, memory_section: str) -> str:
    return f"""The following is a list of known facts extracted from academic papers:

{knowledge_base}

{memory_section}

---
Original question: {question}

Please write a comprehensive answer in English. The answer must be organized into the following three tiers, with each statement clearly attributed to its tier:

## [Direct Paper Evidence]
Content drawn directly from the papers above.
Each statement must be labeled with its source as [Paper Name].
Only state facts explicitly recorded in the papers; do not add any inference.

## [Cross-Literature Inference]
Conclusions that combine information from multiple papers and are reasonably derivable even if not directly stated.
Format: "Cross-paper inference (based on [Paper A] and [Paper B]): ..."
The reasoning must be explained; readers should be able to trace the derivation.

## [Knowledge Extension and Speculation]
Extrapolations beyond the above papers, based on academic knowledge.
Format: "Model speculation (insufficient literature basis): ..."
If the question involves a hypothetical scenario, clearly reason through likely outcomes and state uncertainties.

Key principles:
- Honesty about epistemic limits is more important than completeness of the answer
- If the literature is insufficient to support an inference, explicitly state "insufficient literature basis"
- Speculative content must have academic logical grounding; do not fabricate
- If a tier has no content, it may be omitted
"""


def _reasoning_zh(knowledge_base: str, question: str, memory_section: str) -> str:
    return f"""
以下是從學術論文中整理出的已知事實清單：

{knowledge_base}

{memory_section}

---
原始問題：{question}

請用繁體中文撰寫完整回答。回答必須依以下三個層次組織，每個陳述都要清楚標注所屬層次：

## 【論文直接依據】
直接來自上述論文原文的內容。
每個陳述必須以【論文名稱】標注來源。
只陳述論文明確記載的事實，不加入任何推論。

## 【跨文獻推論】
結合多篇論文的資訊，推導出論文沒有直接說明但合理可得的結論。
格式：「綜合推論（基於【論文A】與【論文B】）：...」
必須說明推導邏輯，讀者應能追溯推導過程。

## 【知識延伸與推測】
超出上述文獻範圍，基於學術知識所做的推演。
格式：「模型推測（文獻依據不足）：...」
若問題涉及假設情境（如改變實驗條件、預測未測試的結果），
請明確推演可能結果並說明不確定性與建議驗證方向。

重要原則：
- 認知邊界的誠實比答案的完整更重要
- 若文獻資料不足以支持某個推論，請明確說「文獻依據不足」，不要假裝有論文支持
- 推測內容必須有學術邏輯依據，不能憑空捏造
- 各層次若無內容可填，可省略該層次
"""


def _strict_en(knowledge_base: str, question: str, memory_section: str) -> str:
    return f"""The following are query results for each sub-question:

{knowledge_base}

{memory_section}

---
Original question: {question}

Based on the above data, write a comprehensive and well-organized synthesized answer in English.
If there are differences across papers, clearly compare them.
Only use the content from the above data; do not add your own information.
Every factual statement must be labeled with its source [Paper Name].
If a paper's query result indicates it does not address this topic, do not fill the gap with content from other papers; state that this paper has no relevant data.
"""


def _strict_zh(knowledge_base: str, question: str, memory_section: str) -> str:
    return f"""
以下是針對各子問題的查詢結果：

{knowledge_base}

{memory_section}

---
原始問題：{question}

請根據以上資料，用繁體中文撰寫一份完整、有條理的綜合回答。
如果各論文有差異，請明確比較。
只使用上述資料中的內容，不要自行補充。
每個事實陳述都必須以【論文名稱】標注來源，不得混用不同論文的內容。
如果某篇論文的查詢結果顯示「此論文未涉及此議題」，則不得用其他論文的內容來填補，應直接說明該論文無相關資料。
"""
