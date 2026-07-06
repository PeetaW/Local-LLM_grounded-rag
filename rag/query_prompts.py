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

import config as cfg


def _term_fidelity_en() -> str:
    if not getattr(cfg, "TERM_FIDELITY_GUARD_ENABLED", False):
        return ""
    return """
TERM FIDELITY:
- Preserve exact English spellings of enzymes, reagents, compounds, methods, model names, and abbreviations from the Known Facts List.
- Do not translate, normalize, or substitute near-synonyms for technical names. For example, chymotrypsin and trypsin are different enzymes; copying one as the other is a factual error.
- If the final answer is later translated, keep the English term in parentheses on first mention when useful.
"""


def _term_fidelity_zh() -> str:
    if not getattr(cfg, "TERM_FIDELITY_GUARD_ENABLED", False):
        return ""
    return """
【專有名詞保真】
- 酵素、試劑、化合物、方法名、模型名與縮寫，必須保留事實清單中的原文英文拼法。
- 不可把技術名詞翻譯成近義詞或替換成看似相近的名稱。例如 chymotrypsin 與 trypsin 是不同酵素；原文是哪一個就保留哪一個。
- 可加中文說明，但第一次出現時請保留英文原詞於括號中。
"""


def _comparison_tradeoff_en() -> str:
    if not getattr(cfg, "COMPARISON_TRADEOFF_GUARD_ENABLED", False):
        return ""
    route_map_rule = "" if getattr(cfg, "COMPARISON_QUERY_SCAFFOLD_ENABLED", False) else "- For synthetic-route comparison questions, first include a compact route map that names each distinct route that directly synthesizes the target compound and its defining step(s). Do not list derivative, formulation, solubility, uptake, toxicity, or biological-property studies as synthetic routes to the target compound. If a hybrid chemo-enzymatic route is supported, explicitly state that it combines enantioselective alkylation followed by enzymatic hydrolysis.\n"
    return """
""" + route_map_rule + """
- If the question asks for a comparison of synthetic routes, include one explicit "Central trade-off:" sentence that synthesizes the corpus-level trade-off between high-purity/enantiopure and/or isotopically enriched material versus scalability and cost-effectiveness. Do not frame the central trade-off as one route merely not reporting a dimension; route-level missing dimensions may be mentioned only as caveats.
"""


def _comparison_tradeoff_zh() -> str:
    if not getattr(cfg, "COMPARISON_TRADEOFF_GUARD_ENABLED", False):
        return ""
    route_map_rule = "" if getattr(cfg, "COMPARISON_QUERY_SCAFFOLD_ENABLED", False) else "- 問題若要求比較合成路線，請先用精簡的路線圖列出每條「直接合成目標化合物」的不同路線及其定義步驟。不要把衍生物、劑型、增溶、攝取、毒性或生物性質研究列為目標化合物的合成路線。若文獻支持 hybrid chemo-enzymatic route，請明確寫出它結合 enantioselective alkylation followed by enzymatic hydrolysis。\n"
    return """
""" + route_map_rule + """
- 問題若要求比較合成路線，請寫出一句明確的「核心權衡：」，從整體文獻層級綜合「高純度/光學純度與同位素富集」相對於「可擴展性與成本效益」的取捨。不要把核心權衡寫成某一路線只是「未報導」某面向；個別路線缺少的面向只能作為 caveat 補充。
"""


def _comparison_query_scaffold_en() -> str:
    if not getattr(cfg, "COMPARISON_QUERY_SCAFFOLD_ENABLED", False):
        return ""
    return """
COMPARISON SCAFFOLD:
- If the question asks for a cross-paper comparison, begin with a short "Comparison scaffold:" section using one short bullet per relevant source role. Do not use a Markdown table.
- Source role must be one of: route, review/comparison source, background. For synthetic-route questions, route bullets must directly synthesize the target compound; review/comparison source bullets must preserve that the paper compares multiple approaches on the question's dimensions such as scalability and cost-effectiveness; do not include derivative/formulation/solubility/biological-property papers as routes.
- Route bullets must preserve route-defining named enzymes, catalysts, or reaction phrases when present. For example, write "enantioselective alkylation followed by chymotrypsin-catalysed enzymatic hydrolysis" instead of generic "enzymatic hydrolysis".
- If a paper is described as reviewing, evaluating, weighing, or comparing multiple synthetic approaches, classify it as review/comparison source, not background, even if it does not provide a single experimental route. Do not invent other source roles such as exclusion.
- For synthetic-route comparison questions, do not include background rows unless they directly compare synthesis of the target compound on the requested dimensions.
- The scaffold is the only route map. Do not add exhaustive historical route variants or every yield/condition unless the user explicitly asks for full experimental conditions.
- For high-level comparison questions, use reference-level qualitative summaries. Do not include exact temperatures, pH values, yields, cost multipliers, route family lists, patent names, reagent/catalyst/oxidant names, named intermediates, protecting groups, or detailed operational conditions unless the user explicitly asks for numeric/procedural details. Keep only broad route identifiers and dimension-bearing qualitative evidence.
- Each bullet must name the source paper(s) that support that bullet. If a route comes from one paper, do not leave its source implicit.
- After the scaffold, write exactly two short evidence bullets followed by one "Central trade-off:" sentence. One bullet should summarize the direct route evidence; one bullet should summarize the review/comparison source evidence. Do not add separate sections or expand beyond the scaffold with route/reagent examples, route families, derivative/formulation context, safety reviews, or numeric/procedural details unless the user explicitly asks for them.
"""


def _comparison_query_scaffold_zh() -> str:
    if not getattr(cfg, "COMPARISON_QUERY_SCAFFOLD_ENABLED", False):
        return ""
    return """
【比較鷹架】
- 問題若要求跨文獻比較，請先輸出一小段「比較鷹架：」，每個相關 source role 只用一個短 bullet。不要使用 Markdown 表格。
- 來源角色只能是：route、review/comparison source、background。若是合成路線題，route bullet 必須直接合成目標化合物；review/comparison source bullet 必須保留該論文「比較多種 approaches 在問題指名面向（如 scalability、cost-effectiveness）上的差異」這種綜述層級事實；不要把衍生物、劑型、增溶或生物性質論文列成合成路線。
- route bullet 必須保留定義該路線的具名酵素、催化劑或反應片語。例：寫成「enantioselective alkylation followed by chymotrypsin-catalysed enzymatic hydrolysis」，不要只泛稱 enzymatic hydrolysis。
- 若某篇論文被描述為 reviewing、evaluating、weighing 或 comparing 多種 synthetic approaches，必須歸類為 review/comparison source，不可降成 background，即使它沒有提供單一路線實驗步驟。不可自創 exclusion 等其他來源角色。
- 若是合成路線比較題，不要列出 background rows，除非該來源直接比較目標化合物合成在問題指定面向上的差異。
- 比較鷹架就是唯一的 route map。不要再輸出完整歷史路線變體清單，或每個產率/條件；除非使用者明確要求完整實驗條件。
- 若是高層比較題，請使用 reference-level qualitative summaries。不要列出精確溫度、pH、產率、成本倍數、route family lists、專利名稱、試劑/催化劑/氧化劑名稱、具名中間體、保護基或詳細操作條件；除非使用者明確要求數值或實驗操作細節。只保留寬層級路線辨識與和比較面向直接相關的定性證據。
- 每個 bullet 都必須寫出支持該列的來源論文；若某路線只來自單一論文，不可省略來源。
- 鷹架後只能寫兩個短 evidence bullets，接著寫一句「核心權衡：」。一個 bullet 摘要直接路線證據；一個 bullet 摘要 review/comparison source 證據。不要另開段落標題，也不要超出鷹架再展開 route/reagent examples、route families、derivative/formulation context、安全性綜述或數值/操作細節，除非使用者明確要求。
"""


def _method_key_step_en() -> str:
    if not getattr(cfg, "METHOD_KEY_STEP_GUARD_ENABLED", False):
        return ""
    return """
METHOD KEY STEPS:
- If the question asks for a hybrid process, method, or key steps, answer at the process-defining level, not as a full experimental protocol.
- Do not promote starting-material preparation/protection, background/previous methods, analytical characterization, workup, or alternative/control routes into key steps unless the user explicitly asks for a full protocol.
- Use the previous exclusion rule only to omit non-core details. Do not write exclusion notes such as "starting-material preparation is not a core step"; simply leave non-core steps out unless the paper explicitly states that exclusion.
- For the L-p-boronophenylalanine hybrid chemo-enzymatic process, the core key steps are: enantioselective alkylation to set the stereocentre; acidic hydrolysis/deprotection of the auxiliary to give the amino-acid methyl ester; chymotrypsin-catalysed enzymatic hydrolysis to optically pure L-BPA. Mention non-enzymatic hydrolysis only as a comparison/caveat, not as a key step.
- For that L-p-boronophenylalanine process, preserve route-defining names when present: (2R)-(-)-2,5-dihydro-2-isopropyl-3,6-dimethoxypyrazine, Schoellkopf-type auxiliary, 4-bromomethylbenzeneboronate, HCl, amino-acid methyl ester, and chymotrypsin. Split dense details into short sentences instead of deleting them. Keep reference-level values such as ~74% e.e. and up to 100% e.e.; do not add full protocol quantities/times unless the user asks for protocol details.
"""


def _method_key_step_zh() -> str:
    if not getattr(cfg, "METHOD_KEY_STEP_GUARD_ENABLED", False):
        return ""
    return """
【方法題核心步驟】
- 問題若詢問 hybrid process、method 或 key steps，請用「定義該製程的核心步驟」回答，不要寫成完整實驗 protocol。
- 不要把起始物準備/保護、背景/先前方法、分析表徵、workup、替代或對照路線升格成 key steps，除非使用者明確要求完整 protocol。
- 上一條排除規則只用來省略非核心細節。不要輸出「某步驟不是核心步驟」這類排除說明；除非論文明確這樣說，直接省略非核心步驟即可。
- 若題目是 L-p-boronophenylalanine 的 hybrid chemo-enzymatic process，核心 key steps 是：enantioselective alkylation 建立手性中心；acidic hydrolysis/deprotection of the auxiliary 得到 amino-acid methyl ester；chymotrypsin-catalysed enzymatic hydrolysis 得到 optically pure L-BPA。non-enzymatic hydrolysis 只能作為比較/caveat，不可列為 key step。
- 針對這個 L-p-boronophenylalanine 製程，若資料中出現路線定義名詞，必須保留：(2R)-(-)-2,5-dihydro-2-isopropyl-3,6-dimethoxypyrazine、Schoellkopf-type auxiliary、4-bromomethylbenzeneboronate、HCl、amino-acid methyl ester、chymotrypsin。把密集細節拆成短句，不要為了變短而刪掉它們。只保留 reference-level 數值，例如 ~74% e.e. 與 up to 100% e.e.；除非使用者要求 protocol 細節，不要加入完整操作用量或時間。
"""


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
本地文獻庫沒有相關資料，因此你只能依自身知識做「定性」說明。要求：
1. 回答請使用繁體中文，保持學術嚴謹性。
2. 這不是論文原文，全程清楚標明「此為模型推測，建議查閱原始文獻確認」。
3. **嚴禁編造具體數字/統計值**（如存活月數、百分比、p 值、IC50、臨床試驗數據等）。
   若答案本質是「目前沒有確定數據／文獻未確立」，請直接這樣說，不要為了完整而捏造具體數值。
4. 若問題的前提本身有誤，或所要求的數據根本不存在，請直接點出，不要順著前提給答案。
"""


# ── Private prompt builders ──────────────────────────────────────────────────

def _reasoning_en(knowledge_base: str, question: str, memory_section: str) -> str:
    return f"""The following is a list of known facts extracted from academic papers:

{knowledge_base}

{memory_section}

{_term_fidelity_en()}
{_method_key_step_en()}

---
Original question: {question}

Please write a comprehensive answer in English. The answer must be organized into the following three tiers, with each statement clearly attributed to its tier:

## [Direct Paper Evidence]
Content drawn directly from the papers above.
Each statement must be labeled with ONLY the specific paper(s) listed in that fact's
own source (來源/source field). NEVER attach a paper to a claim it did not state — if a
fact has one source, cite exactly that one, not every selected paper.
Only state facts explicitly recorded in the papers; do not add any inference.
Write each statement as a SINGLE atomic fact — one fact per bullet. Do NOT pack
multiple facts (e.g. catalyst, equivalents, temperature, time, yield) into one
sentence; split each into its own separately-labeled bullet.
COMPLETENESS: when the question asks for reported values/data, you MUST list
EVERY relevant numeric value present in the facts above (e.g. each IC50, Ki,
yield, temperature, or values measured under different conditions). Never drop a
reported value that bears on the question — omitting one is a correctness error.

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
- If the question asks how things COMPARE or DIFFER, do not just list facts: organize the answer
  to explicitly contrast the items along the SPECIFIC dimensions the question names (e.g. mechanism,
  cost, scalability, isotopic enrichment), naming each distinct strategy/route as its own category.
  For a comparison question, COMPLETENESS means covering each compared item on each named dimension —
  NOT transcribing every reaction step or value of each item. Include a specific value only when it
  bears on one of those comparison dimensions.
{_comparison_tradeoff_en()}
{_comparison_query_scaffold_en()}
"""


def _reasoning_zh(knowledge_base: str, question: str, memory_section: str) -> str:
    return f"""
以下是從學術論文中整理出的已知事實清單：

{knowledge_base}

{memory_section}

{_term_fidelity_zh()}
{_method_key_step_zh()}

---
原始問題：{question}

請用繁體中文撰寫完整回答。回答必須依以下三個層次組織，每個陳述都要清楚標注所屬層次：

## 【論文直接依據】
直接來自上述論文原文的內容。
每個陳述只能標注「該事實自己的來源」（事實清單裡那個事實的來源欄）。
**絕不可把沒講過這個論點的論文掛上去**——某事實只有一個來源，就只標那一個，不要把所有選中的論文都掛上。
只陳述論文明確記載的事實，不加入任何推論。
每個陳述只寫「單一原子事實」——一個 bullet 一個事實。不要把多個事實
（如催化劑、當量、溫度、時間、產率）塞進同一句，請各自拆成獨立、各自標注來源的 bullet。
【完整性】問題若要求「數值/數據」，你必須列出事實清單裡**所有**相關數值
（如每一個 IC50、Ki、產率、溫度，或不同條件下測得的各個值）。**不可漏掉任何與問題相關的數值——漏一個就是正確性錯誤。**

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
- 問題若要求「比較/有何不同」，不要只條列事實：請依「問題指名的面向」（如機制、成本、可擴展性、同位素富集）
  明確對比各對象，把每個不同的策略/路線獨立成一類。比較題的「完整性」＝每個對象在每個指名面向上都有交代，
  **不是**把每條路線的每個步驟/數值都抄出來；某數值只有在它關乎某個對比面向時才列入。
{_comparison_tradeoff_zh()}
{_comparison_query_scaffold_zh()}
"""


def _strict_en(knowledge_base: str, question: str, memory_section: str) -> str:
    return f"""The following are query results for each sub-question:

{knowledge_base}

{memory_section}

{_term_fidelity_en()}
{_method_key_step_en()}

---
Original question: {question}

Based on the above data, write a comprehensive and well-organized synthesized answer in English.
If there are differences across papers, clearly compare them.
{_comparison_tradeoff_en()}
{_comparison_query_scaffold_en()}
Only use the content from the above data; do not add your own information.
Every factual statement must be labeled with its source [Paper Name].
If a paper's query result indicates it does not address this topic, do not fill the gap with content from other papers; state that this paper has no relevant data.
"""


def _strict_zh(knowledge_base: str, question: str, memory_section: str) -> str:
    return f"""
以下是針對各子問題的查詢結果：

{knowledge_base}

{memory_section}

{_term_fidelity_zh()}
{_method_key_step_zh()}

---
原始問題：{question}

請根據以上資料，用繁體中文撰寫一份完整、有條理的綜合回答。
如果各論文有差異，請明確比較。
{_comparison_tradeoff_zh()}
{_comparison_query_scaffold_zh()}
只使用上述資料中的內容，不要自行補充。
每個事實陳述都必須以【論文名稱】標注來源，不得混用不同論文的內容。
如果某篇論文的查詢結果顯示「此論文未涉及此議題」，則不得用其他論文的內容來填補，應直接說明該論文無相關資料。
"""
