# rag/query_engine.py
# 負責 StructuredPlanning：子問題拆解 + 逐篇查詢 + 綜合回答
#
# 更新記錄：
#   - 加入 RAG Fallback 邏輯：資料庫沒找到時切換模型自身知識推理
#   - 修正 _safe_query：先清洗特殊字元再截短，避免語意損失
#   - 加入 select_relevant_papers：先篩選相關論文再查詢，避免問遍所有論文
#   - 修正 streaming 版本 STATUS 訊息：移除 sub_q[:40] 截斷，顯示完整子問題

import re
import json
import concurrent.futures
from llama_index.core import Settings

import config as cfg
from rag.knowledge_synthesizer import KnowledgeSynthesizer
from rag.answer_verifier import AnswerVerifier


def _build_subquery_tasks(sub_questions, paper_engines_to_use, paper_engines):
    """
    將 sub_questions × papers 展平為帶有 original_index 的 task list。
    回傳：
      valid_tasks  — list of (idx, label, engine, sub_q)，送進 thread pool
      prefilled    — dict of {idx: (label, result_str)}，engine 找不到時直接填入
    """
    valid_tasks = []
    prefilled = {}
    idx = 0

    for sq in sub_questions:
        paper = sq.get("paper", "ALL")
        sub_q = sq.get("sub_q", "")

        if paper == "ALL":
            for name, engine in paper_engines_to_use.items():
                valid_tasks.append((idx, f"【{name}】", engine, sub_q))
                idx += 1
        else:
            engine = paper_engines_to_use.get(paper)
            if engine is None:
                matched = next((k for k in paper_engines_to_use if paper in k), None)
                if matched is None:
                    matched = next((k for k in paper_engines if paper in k), None)
                    engine = paper_engines.get(matched) if matched else None
                else:
                    engine = paper_engines_to_use.get(matched)

            if engine:
                valid_tasks.append((idx, f"【{paper}】", engine, sub_q))
            else:
                prefilled[idx] = (f"【{paper}】", f"【{paper}】找不到對應論文")
            idx += 1

    return valid_tasks, prefilled


def _run_subqueries_parallel(valid_tasks, prefilled):
    """
    兩階段執行：
      Phase A（並行）：所有任務同時做 embed 清洗 + 向量檢索（bge-m3），
                       Ollama 只需維持 bge-m3，不會觸發模型切換。
      Phase B（串行）：依序用 LLM 生成每個子問題的答案（gemma4），
                       Ollama 載入一次 gemma4 後連續處理所有任務。

    這樣避免 bge-m3 / gemma4 交錯切換造成的 unload/load 開銷。
    回傳：list of (label, result_str)，順序與原始 sub_questions 一致。
    """
    results = dict(prefilled)

    # ── Phase A：並行 embed 清洗 + 向量檢索 ──────────────────────
    def _retrieve_one(task):
        task_idx, label, engine, sub_q = task
        try:
            query_text = _prepare_query_text(sub_q)
            nodes = _retrieve_nodes(engine, query_text)
            return task_idx, label, engine, query_text, nodes
        except Exception as e:
            print(f"  ⚠️  [Phase A] {label} 檢索失敗：{e}")
            return task_idx, label, engine, sub_q, None

    retrieved = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.SUBQUERY_MAX_WORKERS) as ex:
        futures = [ex.submit(_retrieve_one, t) for t in valid_tasks]
        for f in concurrent.futures.as_completed(futures):
            task_idx, label, engine, query_text, nodes = f.result()
            retrieved[task_idx] = (label, engine, query_text, nodes)

    # ── Phase B：串行 LLM 生成（gemma4 只載入一次）──────────────
    for task_idx in sorted(retrieved.keys()):
        label, engine, query_text, nodes = retrieved[task_idx]
        try:
            result = _generate_from_nodes(engine, nodes, query_text)
            results[task_idx] = (label, result)
        except Exception as e:
            results[task_idx] = (label, f"{label}生成失敗：{e}")

    return [(label, result) for _, (label, result) in sorted(results.items())]

_synthesizer = KnowledgeSynthesizer()
_verifier    = AnswerVerifier()


# ══════════════════════════════════════════════════════════════════
#  輔助函數：embedding 預檢 + 空結果偵測 + 段落抽取
# ══════════════════════════════════════════════════════════════════

def _extract_direct_citation_section(text: str) -> str:
    """
    從答案中只抽取【論文直接依據】段落。
    grounding fallback 只應針對直引段落觸發：
    推論與知識延伸段落的 grounding score 低是預期行為，不應觸發修正。
    若找不到直引段落（表示答案全是推論），回傳空字串。
    """
    import re
    matches = re.findall(
        r'(##[^\n]*(?:論文直接依據|直接依據|直引|Direct.*Evidence)[^\n]*\n[\s\S]*?)(?=\n##|\Z)',
        text
    )
    return "\n\n".join(m.strip() for m in matches)

def _translate_to_traditional_chinese(text: str, on_status=None) -> str:
    """
    Translate the verified English answer to Traditional Chinese (final step of EN_DRAFT_PIPELINE).
    Section headers are mapped back to their Chinese equivalents.
    """
    import requests as _req

    def _status(msg):
        if on_status:
            on_status(msg)
        else:
            print(msg)

    _status("\n  🌏 翻譯英文答案為繁體中文...")
    prompt = (
        "Translate the following academic answer from English to Traditional Chinese (繁體中文).\n"
        "Rules:\n"
        "- Section headers must be translated as follows:\n"
        "  '## [Direct Paper Evidence]' → '## 【論文直接依據】'\n"
        "  '## [Cross-Literature Inference]' → '## 【跨文獻推論】'\n"
        "  '## [Knowledge Extension and Speculation]' → '## 【知識延伸與推測】'\n"
        "- Paper name labels [Paper Name] → 【Paper Name】 (keep the name itself unchanged)\n"
        "- Preserve all numbers, units (wt%, °C, rpm, g, mL, h), and chemical formulas exactly.\n"
        "- Keep label tags unchanged: [Fact N], [Insufficient Evidence], [Unverified], VERIFY_PASS, VERIFY_FAIL.\n"
        "- Do not add any explanation, preamble, or markdown fence.\n\n"
        f"Answer to translate:\n{text}"
    )
    try:
        resp = _req.post(
            f"{cfg.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": cfg.SYNTHESIS_MODEL,
                "system": "You are a professional academic translator specializing in Traditional Chinese (繁體中文).",
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": -1, "num_ctx": 65536},
            },
            timeout=cfg.LLM_TIMEOUT,
        )
        if resp.ok:
            translated = resp.json().get("response", "").strip()
            if translated:
                _status(f"  ✅ 翻譯完成（{len(translated):,} 字元）")
                return translated
    except Exception as e:
        _status(f"  ⚠️  翻譯失敗，保留英文版本：{e}")
    return text


def _clean_for_embed(text: str) -> str:
    """
    移除可能導致 bge-m3 產生 NaN 的字元，盡量保留完整語意。
    模組層級函數，供 _prepare_query_text 使用。
    """
    import unicodedata
    removed_ctrl = [
        f"U+{ord(c):04X}({unicodedata.name(c, '?')})"
        for c in text if unicodedata.category(c).startswith('C')
    ]
    if removed_ctrl:
        print(f"  🔬 [embed-debug] 移除控制字元：{removed_ctrl[:5]}")

    text = ''.join(c for c in text if not unicodedata.category(c).startswith('C'))
    text = text.replace('（', '(').replace('）', ')')

    angle_matches = re.findall(r'[<>]\s*\d+\s*\w+', text)
    if angle_matches:
        print(f"  🔬 [embed-debug] 角括號數值（替換）：{angle_matches}")
    text = re.sub(r'<\s*(\d+\s*\w+)', r'less than \1', text)
    text = re.sub(r'>\s*(\d+\s*\w+)', r'greater than \1', text)

    long_parens = re.findall(r'\([^)]{25,}\)', text)
    if long_parens:
        print(f"  🔬 [embed-debug] 移除長括號內容：{[p[:30] for p in long_parens]}")
    text = re.sub(r'\([^)]{25,}\)', '', text)

    return text.strip()


def _test_embed(text: str, label: str = "") -> str:
    """
    對單一文字做 embedding 預檢。
    回傳 'ok'/'nan'/'timeout'/'error'。
    模組層級函數，thread-safe（只做 HTTP GET，無共享狀態）。
    """
    import requests
    try:
        r = requests.post(
            f"{cfg.OLLAMA_BASE_URL}/api/embeddings",
            json={"model": cfg.EMBED_MODEL, "prompt": text},
            timeout=120,
        )
        if r.status_code != 200:
            # HTTP 500 + NaN 訊息 = bge-m3 產生 NaN，Ollama 無法序列化
            # 這是文字內容問題，回傳 "nan" 觸發清洗/截短，而非等待重試
            if r.status_code == 500 and "NaN" in r.text:
                print(f"  🔬 [embed-debug] HTTP 500 NaN（bge-m3 輸出 NaN），text={text[:60]!r}")
                return "nan"
            print(f"  🔬 [embed-debug] HTTP {r.status_code}，text={text[:60]!r}")
            return "error"
        embedding = r.json().get("embedding", [])
        if not embedding:
            print(f"  🔬 [embed-debug] embedding 為空，text={text[:60]!r}")
            return "error"
        nan_count = sum(1 for x in embedding if x != x)
        if nan_count:
            print(f"  🔬 [embed-debug] NaN {nan_count}/{len(embedding)} 維，"
                  f"label={label!r}，text={text[:80]!r}")
            return "nan"
        return "ok"
    except requests.exceptions.Timeout:
        print(f"  🔬 [embed-debug] Timeout（Ollama 忙碌），text={text[:60]!r}")
        return "timeout"
    except Exception as e:
        print(f"  🔬 [embed-debug] 例外：{e}，text={text[:60]!r}")
        return "error"


def _embed_with_retry(text: str, label: str = "", max_retries: int = 5) -> bool:
    """
    timeout 時等待後重試，NaN 時回傳 False（讓呼叫端截短）。
    模組層級函數。
    """
    import time
    for i in range(max_retries):
        result = _test_embed(text, label=label)
        if result == "ok":
            return True
        if result == "nan":
            return False
        wait = 15 * (i + 1)
        print(f"  ⏳ [embed] Ollama 忙碌，{wait}s 後重試（第{i+1}/{max_retries}次）...")
        time.sleep(wait)
    return False


def _prepare_query_text(query_str: str) -> str:
    """
    Phase A-1：清洗文字 + embed 預檢，回傳可安全送進 retriever 的 query text。
    只使用 bge-m3，不涉及 LLM，thread-safe。
    """
    cleaned = _clean_for_embed(query_str)
    if cleaned != query_str:
        print(f"  🧹 偵測到特殊字元，清洗後重試...")
        if _embed_with_retry(cleaned, label="cleaned"):
            return cleaned

    if _embed_with_retry(query_str, label="original"):
        return query_str

    # 最後手段：截短
    current = cleaned if cleaned else query_str
    for attempt in range(3):
        cut = int(len(current) * 2 / 3)
        current = current[:cut].strip()
        print(f"  ⚠️  embedding NaN，截短至 {len(current)} 字元後重試（第{attempt+1}次）")
        print(f"  🔬 [embed-debug] 截短後內容：{current!r}")
        if _embed_with_retry(current, label=f"truncated-{attempt+1}"):
            return current

    print(f"  ❌ embedding 反覆失敗，使用截短版本強制查詢")
    return current


def _retrieve_nodes(engine, query_text: str):
    """
    Phase A-2：只做向量檢索，回傳 retrieved nodes。
    使用 bge-m3，不呼叫 LLM，thread-safe（read-only index）。
    """
    retriever = engine.retriever if hasattr(engine, 'retriever') else None
    if retriever is not None:
        return retriever.retrieve(query_text)
    # fallback：若 engine 沒有暴露 retriever，降級為整體查詢
    return None


def _generate_from_nodes(engine, nodes, query_text: str) -> str:
    """
    Phase B：只做 LLM 生成（gemma4）。
    nodes 已由 Phase A 取得，此函數串行呼叫，避免模型切換。
    """
    if nodes is None:
        # fallback：retriever 無法單獨存取，退回整體查詢
        return str(engine.query(query_text))
    from llama_index.core.response_synthesizers import get_response_synthesizer
    from llama_index.core import QueryBundle
    synthesizer = get_response_synthesizer()
    response = synthesizer.synthesize(query=QueryBundle(query_text), nodes=nodes)
    return str(response)


# ── 空結果偵測 ────────────────────────────────────────────────────
_NO_RESULT_PATTERNS = [
    "此論文未涉及",
    "empty response",
    "no information",
    "找不到",
    "未找到",
    "沒有相關",
    "無相關",
    "i don't have",
    "i cannot find",
    "not mentioned",
    "the context does not",
    "no relevant",
    "does not contain",
    "not found",
    "no context",
    "沒有找到",
    "查詢失敗",
]

def _is_empty_result(text: str) -> bool:
    """
    判斷 RAG 回傳的是否為無效結果。
    回傳 True 代表這個結果沒有實質內容，不應算進有效檢索。
    """
    text_lower = text.lower().strip()
    if len(text_lower) < 30:
        return True
    return any(pat in text_lower for pat in _NO_RESULT_PATTERNS)


def _extract_paper_name(ans: str, fallback: str) -> str:
    """從 sub_answer 文字中提取第一個【論文名稱】作為來源標籤。"""
    m = re.search(r'【(.+?)】', ans)
    return m.group(1) if m else fallback


# ══════════════════════════════════════════════════════════════════
#  論文篩選：先選出相關論文，再查詢
# ══════════════════════════════════════════════════════════════════

def _keyword_prefilter(question: str, paper_names: list) -> list:
    """
    純關鍵字比對預篩，完全不呼叫 LLM，速度極快。

    做法：
    1. 把問題斷詞（英文用空格，中文用字元）
    2. 跟每篇論文的 keywords、main_topic、short_desc 比對
    3. 有任何詞命中就保留這篇論文
    4. 全部沒命中（問題太通用）就回傳全部論文

    目的：把明顯不相關的論文在 LLM 篩選之前就過濾掉，
    減少 select_relevant_papers 要處理的論文數量。
    """
    from rag.metadata_manager import load_metadata

    all_metadata = load_metadata()

    # 問題斷詞：英文取長度 >= 3 的詞，中文取所有字元組合
    question_lower = question.lower()
    # 英文詞
    en_words = set(w for w in re.split(r'[\s\-_,./;:!?()\[\]]+', question_lower) if len(w) >= 3)
    # 中文詞（2字以上的子字串）
    zh_chars = re.findall(r'[\u4e00-\u9fff]{2,}', question)
    question_terms = en_words | set(zh_chars)

    if not question_terms:
        return paper_names

    scored = []
    for p in paper_names:
        meta = all_metadata.get(p, {})
        # 把這篇論文的所有 metadata 文字合併成一個字串
        meta_text = " ".join([
            " ".join(meta.get("keywords", [])),
            meta.get("main_topic", ""),
            meta.get("short_desc", ""),
            p,  # 檔名本身也算
        ]).lower()

        # 計算命中詞數
        hits = sum(1 for term in question_terms if term in meta_text)
        if hits > 0:
            scored.append((p, hits))

    if not scored:
        # 全部沒命中，問題可能太通用（如「這些論文的共同點」），退回全部
        return paper_names

    # 按命中數排序，回傳有命中的論文
    scored.sort(key=lambda x: x[1], reverse=True)
    result = [p for p, _ in scored]

    print(f"  🔑 關鍵字預篩：{len(paper_names)} 篇 → {len(result)} 篇")
    return result


def select_relevant_papers(question: str, paper_names: list) -> list:
    """
    用 qwen2.5:14b 從論文清單中選出跟問題最相關的幾篇。

    設計原則：
    - 問題沒有指定特定論文時，先用模型判斷哪幾篇最可能有答案
    - 最多選 5 篇，避免查詢所有論文浪費時間
    - 失敗時退回全部論文（安全降級）
    """
    from rag.llm_client import planning_llm
    from rag.metadata_manager import load_metadata

    all_metadata = load_metadata()
    paper_list_str = "\n".join(
        f"- {p}：{all_metadata.get(p, {}).get('short_desc', '（無描述）')}"
        f"（關鍵字：{', '.join(all_metadata.get(p, {}).get('keywords', [])[:4])}）"
        for p in paper_names
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
            return paper_names

        # 驗證：只保留確實存在的論文名稱
        valid = [p for p in selected if p in paper_names]

        if valid:
            print(f"  📌 篩選出 {len(valid)} 篇相關論文（共 {len(paper_names)} 篇）：")
            for p in valid:
                print(f"     - {p[:60]}")
            return valid
        else:
            print(f"  ⚠️  論文篩選結果為空，退回查詢全部論文")
            return paper_names

    except Exception as e:
        print(f"  ⚠️  論文篩選失敗（{e}），退回查詢全部論文")
        return paper_names


# ══════════════════════════════════════════════════════════════════
#  論文比對 + 子問題拆解
# ══════════════════════════════════════════════════════════════════

def detect_target_paper(question: str, paper_names: list) -> str | None:
    """
    確定性字串比對，不依賴 LLM。
    掃描問題裡是否出現論文檔名的關鍵片段。
    """
    question_lower = question.lower()

    best_match = None
    best_score = 0

    for name in paper_names:
        segments = [s for s in name.lower().split("-") if len(s) > 3]
        matches = sum(1 for seg in segments if seg in question_lower)
        score = matches / len(segments) if segments else 0

        if score > best_score and score >= 0.3:
            best_score = score
            best_match = name

    return best_match


def plan_sub_questions(question: str, paper_names: list) -> list:
    """
    用小模型（qwen2.5:14b）把複合問題拆成子問題清單。
    只做 JSON 格式化，不需要深度推理。
    若問題明確指定特定論文，paper 欄位設為該論文名稱，不設為 ALL。
    只有明確需要跨論文比較時，才使用 ALL。
    回傳格式：[{"paper": "論文名稱或ALL", "sub_q": "子問題"}]
    """
    from rag.llm_client import planning_llm
    from rag.metadata_manager import load_metadata

    # 第一層：確定性比對
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

    # 載入所有論文的描述
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
4. 若問題涉及合成或實驗方法，必須額外拆出一個子問題專門詢問具體操作參數，例如：amounts, concentrations, temperature, stirring speed, reaction time。
5. 若問題詢問「重點」「主要發現」「貢獻」「結論」「這篇在說什麼」，
   子問題應包含：
   - What is the main research objective and novelty?
   - What are the key findings and conclusions?
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
            print(f"  → 子問題內容：{[sq.get('sub_q', '')[:200] for sq in sub_questions]}")
            return sub_questions
        except json.JSONDecodeError:
            print(f"  ⚠️  子問題拆解失敗（第{attempt+1}次），raw 內容：{raw[:200]}")
            if attempt == 0:
                print("       重試中...")

    print("       改為對所有論文問同一問題")
    return [{"paper": "ALL", "sub_q": question}]


# ══════════════════════════════════════════════════════════════════
#  主查詢流程（含論文篩選 + Fallback 邏輯）
# ══════════════════════════════════════════════════════════════════

def execute_structured_query(
    question: str,
    paper_engines: dict,
    memory_context: str = "",
    on_status=None,
) -> str:
    """
    StructuredPlanning 主流程：
    0. 先篩選相關論文（避免問遍所有論文）
    1. LLM 拆解子問題（記憶體內）
    2. 逐一查詢（線性，避免 VRAM 爆炸）
    3. 合併所有子答案，LLM 做最終綜合

    Fallback 邏輯：
    - 若所有子查詢均未返回有效內容（rag_found_anything = False），
      自動切換為模型自身知識推理，並在回答開頭標注來源說明。
    """
    def _status(msg):
        if on_status:
            on_status(msg)
        else:
            print(msg)

    all_paper_names = list(paper_engines.keys())

    # ── Step 0：先篩選相關論文 ────────────────────────────────────
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
        selected_names = select_relevant_papers(question, prefiltered)
        paper_names = selected_names
        paper_engines_to_use = {k: v for k, v in paper_engines.items() if k in selected_names}

    _status("\n  📋 拆解子問題中...")
    sub_questions = plan_sub_questions(question, paper_names)
    _status(f"  → 拆出 {len(sub_questions)} 個子問題")

    sub_answers = []
    rag_found_anything = False

    # ── Stage 2：並行子查詢 ────────────────────────────────────────
    _status(f"\n  ⚡ 並行檢索 {len(sub_questions)} 個子問題中（workers={cfg.SUBQUERY_MAX_WORKERS}）...")
    valid_tasks, prefilled = _build_subquery_tasks(sub_questions, paper_engines_to_use, paper_engines)
    ordered_results = _run_subqueries_parallel(valid_tasks, prefilled)

    for label, result in ordered_results:
        sub_answers.append(f"{label}\n{result}")
        if not _is_empty_result(result):
            rag_found_anything = True
        _status(f"\n  ── {label} 回覆 ──\n  {result[:200]}")

    _status("\n  🔗 綜合所有子答案中...")
    combined = "\n\n".join(sub_answers)

    # ── Stage 3：知識蒸餾 ────────────────────────────────────
    if cfg.SYNTHESIS_ENABLED and rag_found_anything:
        _status("\n  🧪 Stage 3: 知識蒸餾中...")
        synthesis_chunks = [
            {"text": ans, "source": _extract_paper_name(ans, f"retrieved_chunk_{i}")}
            for i, ans in enumerate(sub_answers)
        ]
        knowledge_base = _synthesizer.synthesize(
            chunks=synthesis_chunks,
            query=question,
            on_status=on_status,
        )
    else:
        knowledge_base = combined

    if not rag_found_anything:
        # ── Fallback：RAG 完全沒找到，切換至模型自身知識 ──────────
        _status("  ℹ️  RAG 資料庫未找到相關內容，切換至模型推理模式...")
        fallback_notice = (
            "⚠️ **資料來源說明**：本地學術文獻資料庫中未找到與此問題直接相關的內容。"
            "以下回答來自模型自身知識，非論文原文，請謹慎參考並自行查證。\n\n"
        )
        memory_section = (
            ("【相關歷史問答記憶，僅供參考】" + chr(10) + memory_context + chr(10))
            if memory_context else ""
        )
        synthesis_prompt = f"""
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
    else:
        # ── 正常路徑：有 RAG 結果，根據 REASONING_MODE 選 prompt ──────
        fallback_notice = ""
        memory_section = (
            ("---" + chr(10) + "【相關歷史問答記憶，僅供參考】" + memory_context)
            if memory_context else ""
        )

        if cfg.REASONING_MODE == "reasoning":
            # ── 推理模式：允許跨文獻推論與知識延伸，但標注認知層次 ──
            print("  🧠 推理模式（reasoning）：允許跨文獻推論")
            if cfg.EN_DRAFT_PIPELINE:
                synthesis_prompt = f"""The following is a list of known facts extracted from academic papers:

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
            else:
                synthesis_prompt = f"""
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
        else:
            # ── strict 模式：只引用，不推理（原本邏輯）────────────
            print("  📋 嚴格模式（strict）：只引用論文原文")
            if cfg.EN_DRAFT_PIPELINE:
                synthesis_prompt = f"""The following are query results for each sub-question:

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
            else:
                synthesis_prompt = f"""
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

    print("\n 最終綜合回答（Stage 4 初稿）：")
    full_text = fallback_notice
    for chunk in Settings.llm.stream_complete(synthesis_prompt):
        print(chunk.delta, end="", flush=True)
        full_text += chunk.delta
    print("\n")

    # ── Stage 5：邏輯自洽驗證 ────────────────────────────────────
    if cfg.VERIFY_ENABLED and rag_found_anything:
        _status("\n  🔍 Stage 5: 邏輯自洽驗證中...")
        full_text = _verifier.verify_and_correct(
            draft_answer=full_text,
            knowledge_base=knowledge_base,
            on_status=on_status,
        )

    # ── Citation Grounding（EN_DRAFT_PIPELINE 時在翻譯前執行，確保 EN vs EN）──
    nli_report = ""
    if cfg.CITATION_GROUNDING_ENABLED and rag_found_anything:
        try:
            from rag.citation_grounding import (
                split_into_sentences,
                check_citation_grounding,
                format_grounding_report,
                compute_grounding_score,
            )

            print("  🔍 執行答案品質審查...")
            sentences = split_into_sentences(full_text)
            chunks = [
                {"id": f"CHUNK-{i:03d}", "text": ans}
                for i, ans in enumerate(sub_answers)
            ]
            citation_results = check_citation_grounding(sentences, chunks)

            # ── Grounding Fallback：只針對【論文直接依據】段落 ──
            direct_section = _extract_direct_citation_section(full_text)
            if direct_section:
                direct_sentences = split_into_sentences(direct_section)
                direct_results = check_citation_grounding(direct_sentences, chunks)
                direct_score = compute_grounding_score(direct_results)
            else:
                direct_results = []
                direct_score = 1.0

            grounding_score = compute_grounding_score(citation_results)
            unsupported = [r for r in direct_results if not r["supported"]]
            if unsupported and direct_score < 0.8:
                print(
                    f"  🔄 [Grounding Fallback] {len(unsupported)} 個陳述依據不足"
                    f"（整體 {grounding_score:.1%}），送回 gemma4 重新引用..."
                )
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
                try:
                    import requests as _req
                    resp = _req.post(
                        f"{cfg.OLLAMA_BASE_URL}/api/generate",
                        json={
                            "model": cfg.SYNTHESIS_MODEL,
                            "system": cfg.LLM_SYSTEM_PROMPT,
                            "prompt": fallback_prompt,
                            "stream": False,
                            "options": {
                                "temperature": 0.1,
                                "num_ctx": 65536,
                                "num_predict": -1,
                            },
                        },
                        timeout=cfg.LLM_TIMEOUT,
                    )
                    if resp.ok:
                        corrected = resp.json().get("response", "").strip()
                        if corrected:
                            full_text = corrected
                            print("  ✅ [Grounding Fallback] gemma4 修正完成，重新執行 grounding 審查...")
                            sentences = split_into_sentences(full_text)
                            citation_results = check_citation_grounding(sentences, chunks)
                except Exception as fe:
                    print(f"  ⚠️  [Grounding Fallback] 修正失敗，保留原答案：{fe}")

            nli_report = format_grounding_report(citation_results)
            print(nli_report)

        except Exception as e:
            print(f"  ⚠️  答案品質審查失敗（不影響主流程）：{e}")

    # ── EN_DRAFT_PIPELINE：NLI 完成後翻譯為繁體中文 ─────────────
    if cfg.EN_DRAFT_PIPELINE and rag_found_anything:
        full_text = _translate_to_traditional_chinese(full_text, on_status=on_status)

    if nli_report:
        full_text += nli_report

    return full_text


# ══════════════════════════════════════════════════════════════════
#  Streaming 版本查詢（供 /v1/chat/completions SSE endpoint 使用）
# ══════════════════════════════════════════════════════════════════

def execute_structured_query_stream(
    question: str,
    paper_engines: dict,
    memory_context: str = "",
    on_status=None,
):
    """
    execute_structured_query 的 streaming generator 版本。
    前置流程完全相同，差別只在最後 LLM 綜合推論改成 yield chunk。

    yield 兩種格式：
      [STATUS] 開頭 → 進度訊息（篩論文、拆子問題等）——由 api.py 轉換成 blockquote 顯示
      其他          → LLM 實際輸出文字——直接顯示，寫入記憶
    """
    all_paper_names = list(paper_engines.keys())

    # ── Step 0：篩選相關論文 ──────────────────────────────────────
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
        selected_names = select_relevant_papers(question, prefiltered)
        paper_names = selected_names
        paper_engines_to_use = {k: v for k, v in paper_engines.items() if k in selected_names}
        yield f"[STATUS] 📌 已選出 {len(selected_names)} 篇相關論文\n"

    # ── Step 1：拆解子問題 ────────────────────────────────────────
    yield "[STATUS] 📋 拆解子問題中...\n"
    sub_questions = plan_sub_questions(question, paper_names)
    yield f"[STATUS] → 拆出 {len(sub_questions)} 個子問題，開始檢索...\n"

    # ── Step 2：並行子查詢 ────────────────────────────────────────
    sub_answers = []
    rag_found_anything = False

    yield f"[STATUS] ⚡ 並行檢索 {len(sub_questions)} 個子問題中（workers={cfg.SUBQUERY_MAX_WORKERS}）...\n"
    valid_tasks, prefilled = _build_subquery_tasks(sub_questions, paper_engines_to_use, paper_engines)
    ordered_results = _run_subqueries_parallel(valid_tasks, prefilled)

    for label, result in ordered_results:
        sub_answers.append(f"{label}\n{result}")
        if not _is_empty_result(result):
            rag_found_anything = True
        preview = result[:120].replace("\n", " ")
        yield f"[STATUS] {label} → {preview}...\n"

    # ── Step 3：組合 synthesis prompt ────────────────────────────
    combined = "\n\n".join(sub_answers)

    # ── Stage 3：知識蒸餾 ────────────────────────────────────
    if cfg.SYNTHESIS_ENABLED and rag_found_anything:
        yield "[STATUS] 🧪 Stage 3: 知識蒸餾中...\n"
        synthesis_chunks = [
            {"text": ans, "source": _extract_paper_name(ans, f"retrieved_chunk_{i}")}
            for i, ans in enumerate(sub_answers)
        ]
        knowledge_base = _synthesizer.synthesize(
            chunks=synthesis_chunks,
            query=question,
            on_status=on_status,
        )
        yield "[STATUS] 📋 事實清單已整理完成\n"
    else:
        knowledge_base = combined

    if not rag_found_anything:
        yield "[STATUS] ⚠️ RAG 未找到相關內容，切換至模型知識推理...\n"
        fallback_notice = (
            "⚠️ **資料來源說明**：本地學術文獻資料庫中未找到與此問題直接相關的內容。"
            "以下回答來自模型自身知識，非論文原文，請謹慎參考並自行查證。\n\n"
        )
        memory_section = (
            ("【相關歷史問答記憶，僅供參考】" + chr(10) + memory_context + chr(10))
            if memory_context else ""
        )
        synthesis_prompt = f"""
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
    else:
        fallback_notice = ""
        memory_section = (
            ("---" + chr(10) + "【相關歷史問答記憶，僅供參考】" + memory_context)
            if memory_context else ""
        )
        if cfg.REASONING_MODE == "reasoning":
            yield "[STATUS] 🧠 推理模式，LLM 綜合推論中...\n"
            if cfg.EN_DRAFT_PIPELINE:
                synthesis_prompt = f"""The following is a list of known facts extracted from academic papers:

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
            else:
                synthesis_prompt = f"""
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
若問題涉及假設情境，請明確推演可能結果並說明不確定性。

重要原則：
- 認知邊界的誠實比答案的完整更重要
- 若文獻資料不足以支持某個推論，請明確說「文獻依據不足」
- 各層次若無內容可填，可省略該層次
"""
        else:
            yield "[STATUS] 📋 嚴格模式，LLM 整理論文內容中...\n"
            if cfg.EN_DRAFT_PIPELINE:
                synthesis_prompt = f"""The following are query results for each sub-question:

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
            else:
                synthesis_prompt = f"""
以下是針對各子問題的查詢結果：

{knowledge_base}

{memory_section}

---
原始問題：{question}

請根據以上資料，用繁體中文撰寫一份完整、有條理的綜合回答。
如果各論文有差異，請明確比較。
只使用上述資料中的內容，不要自行補充。
每個事實陳述都必須以【論文名稱】標注來源。
"""

    # ── Step 4：LLM Streaming 輸出 ───────────────────────────────
    if fallback_notice:
        yield fallback_notice

    full_text = fallback_notice
    for chunk in Settings.llm.stream_complete(synthesis_prompt):
        yield chunk.delta
        full_text += chunk.delta

    # ── Stage 5：邏輯自洽驗證（streaming 版：完成後才驗證）────
    if cfg.VERIFY_ENABLED and rag_found_anything:
        yield "[STATUS] 🔍 Stage 5: 邏輯自洽驗證中...\n"
        corrected = _verifier.verify_and_correct(
            draft_answer=full_text,
            knowledge_base=knowledge_base,
            on_status=on_status,
        )
        if corrected != full_text:
            yield "\n\n---\n📝 **已根據邏輯自洽驗證修正如下：**\n\n"
            yield corrected
            full_text = corrected
        else:
            yield "[STATUS] ✅ Stage 5 邏輯驗證通過（VERIFY_PASS），答案無需修正\n"

    # ── Citation Grounding（EN_DRAFT_PIPELINE 時在翻譯前執行，確保 EN vs EN）──
    nli_report = ""
    if cfg.CITATION_GROUNDING_ENABLED and rag_found_anything:
        try:
            from rag.citation_grounding import (
                split_into_sentences,
                check_citation_grounding,
                format_grounding_report,
                compute_grounding_score,
            )
            sentences = split_into_sentences(full_text)
            chunks_data = [
                {"id": f"CHUNK-{i:03d}", "text": ans}
                for i, ans in enumerate(sub_answers)
            ]
            citation_results = check_citation_grounding(sentences, chunks_data)

            # ── Grounding Fallback：只針對【論文直接依據】段落 ──
            direct_section = _extract_direct_citation_section(full_text)
            if direct_section:
                direct_sentences = split_into_sentences(direct_section)
                direct_results = check_citation_grounding(direct_sentences, chunks_data)
                direct_score = compute_grounding_score(direct_results)
            else:
                direct_results = []
                direct_score = 1.0

            unsupported = [r for r in direct_results if not r["supported"]]
            if unsupported and direct_score < 0.8:
                yield (
                    f"[STATUS] 🔄 [Grounding Fallback] 直引段落 {len(unsupported)} 個陳述依據不足"
                    f"（直引 {direct_score:.1%}），送回 gemma4 重新引用...\n"
                )
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
                try:
                    import requests as _req
                    resp = _req.post(
                        f"{cfg.OLLAMA_BASE_URL}/api/generate",
                        json={
                            "model": cfg.SYNTHESIS_MODEL,
                            "system": cfg.LLM_SYSTEM_PROMPT,
                            "prompt": fallback_prompt,
                            "stream": False,
                            "options": {
                                "temperature": 0.1,
                                "num_ctx": 65536,
                                "num_predict": -1,
                            },
                        },
                        timeout=cfg.LLM_TIMEOUT,
                    )
                    if resp.ok:
                        corrected = resp.json().get("response", "").strip()
                        if corrected:
                            full_text = corrected
                            yield "[STATUS] ✅ [Grounding Fallback] gemma4 修正完成，重新執行 grounding 審查...\n"
                            sentences = split_into_sentences(full_text)
                            citation_results = check_citation_grounding(sentences, chunks_data)
                except Exception as fe:
                    yield f"[STATUS] ⚠️ [Grounding Fallback] 修正失敗，保留原答案：{fe}\n"

            nli_report = format_grounding_report(citation_results)
        except Exception as e:
            nli_report = f"\n\n⚠️ 答案品質審查失敗：{e}"

    # ── EN_DRAFT_PIPELINE：NLI 完成後翻譯為繁體中文 ─────────────
    if cfg.EN_DRAFT_PIPELINE and rag_found_anything:
        yield "[STATUS] 🌏 翻譯英文答案為繁體中文...\n"
        translated = _translate_to_traditional_chinese(full_text, on_status=on_status)
        if translated != full_text:
            yield "\n\n---\n🌏 **繁體中文最終版本：**\n\n"
            yield translated
            full_text = translated

    # NLI 報告在翻譯後輸出
    if nli_report:
        yield nli_report