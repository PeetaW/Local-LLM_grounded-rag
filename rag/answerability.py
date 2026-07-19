# rag/answerability.py
# 可答性 gate（answerability gate）— Stage 3 之後、Stage 4 生成之前的前置判斷。
# 問題：檢索/蒸餾出的事實是否「真的包含」問題要的答案，而非只是「同主題」。
# 比既有的 rag_found_anything（只看檢索非空）更細：擋掉「相關但不含答案」→ 走誠實棄答。
# 用 LLM_MODEL（gemma4，與 Stage 3 同模型 → 不觸發 VRAM swap）。保守偏 ANSWERABLE，避免誤殺好答案。
import re
import requests
import config as cfg

_SYSTEM = (
    "You decide how well a set of distilled facts can answer a question. You are NOT answering the "
    "question yourself. Judge ONE thing: do the facts contain the underlying information the "
    "question asks about? Being about the SAME TOPIC is NOT enough — the facts must actually "
    "contain the requested information.\n"
    "SYNTHESIS questions: if the question asks to COMPARE, CONTRAST, or AGGREGATE, judge whether "
    "the underlying per-item INPUTS are present (e.g. the per-route yields/costs needed for a "
    "comparison). The comparison itself need NOT be pre-written — assembling it is the answer step.\n"
    "VALUE or CONDITION LIST questions: ANSWERABLE requires the facts to contain the requested "
    "value or condition for each requested comparison arm. If the facts say one arm differs from "
    "another but omit that other arm's requested value, choose PARTIAL.\n"
    "Choose exactly one verdict:\n"
    "- ANSWERABLE: the facts contain a direct answer, or the full underlying inputs for a synthesis "
    "question. The question can be answered well.\n"
    "- PARTIAL: the facts contain SOME relevant information bearing on the question, but it is "
    "incomplete or thin — the question can be only partially answered, with gaps.\n"
    "- NOT_ANSWERABLE: the requested information is genuinely ABSENT — the question asks for "
    "value/result X and the facts simply never state X (no amount of synthesis could produce it), "
    "Use this only when the facts are merely on the same topic but do not contain the requested "
    "thing at all. If the facts disprove a false premise AND provide the correct alternative, choose "
    "ANSWERABLE because the answer can correct the premise.\n"
    "Bias: when unsure between ANSWERABLE and PARTIAL, prefer ANSWERABLE; when unsure between "
    "PARTIAL and NOT_ANSWERABLE, prefer PARTIAL. Reserve NOT_ANSWERABLE for genuine absence."
)
_VERDICTS = ("ANSWERABLE", "PARTIAL", "NOT_ANSWERABLE")

# Phase 2 路由用的告示文字。
ABSTAIN_NOTICE = (
    "⚠️ **誠實棄答**：檢索到的內容與此問題相關，但並不包含可直接回答的資訊。"
    "為避免編造不存在的數據，本系統選擇不作答。建議換個問法，或確認該主題是否涵蓋於文獻庫中。\n"
)
WEAK_NOTICE = (
    "⚠️ **資料可能不足**：檢索內容僅部分涵蓋此問題，以下回答可能不完整，請謹慎參考並自行查證。\n\n"
)


def gate_route(verdict: str):
    """verdict → (abstain: bool, notice: str)。
    NOT_ANSWERABLE→硬棄答（跳生成）；PARTIAL→軟警告（照常生成、加橫幅）；其餘→正常。
    verdict=None（未判定/呼叫失敗）保守當正常，不棄答。"""
    if verdict == "NOT_ANSWERABLE":
        return True, ABSTAIN_NOTICE
    if verdict == "PARTIAL":
        return False, WEAK_NOTICE
    return False, ""


def assess_answerability(question: str, knowledge_base: str,
                         model: str = None, base_url: str = None, timeout: int = 600) -> dict:
    """回傳 {"verdict": "ANSWERABLE"|"NOT_ANSWERABLE"|None, "reason": str}。
    呼叫失敗或無法解析 → verdict=None（上層當『未判定』，保守不棄答）。"""
    model = model or cfg.LLM_MODEL
    base_url = base_url or cfg.OLLAMA_BASE_URL
    if not (knowledge_base or "").strip():
        return {"verdict": "NOT_ANSWERABLE", "reason": "empty knowledge base"}

    prompt = (
        f"FACTS:\n{knowledge_base}\n\n"
        f"QUESTION:\n{question}\n\n"
        "Output exactly two lines:\nVERDICT: <ANSWERABLE|PARTIAL|NOT_ANSWERABLE>\nREASON: <one sentence>"
    )
    payload = {
        "model": model,
        "system": _SYSTEM,
        "prompt": prompt,
        "stream": False,
        # 關 thinking：gemma4 是 thinking 模型，判定任務若開思考會把 num_predict 全燒在思考通道、
        # response 吐空（done_reason=length, eval_count 滿但 response 空）。同 judge.py 對 qwen3 的修法。
        "think": False,
        "options": {"temperature": 0.1, "num_predict": 256, "num_ctx": cfg.STAGE3_NUM_CTX,
                    "thinking": False},
    }
    try:
        out = ""
        for _ in range(2):  # 空回應重試一次（保險，gemma 偶發吐空）
            resp = requests.post(f"{base_url}/api/generate", json=payload, timeout=timeout)
            resp.raise_for_status()
            out = resp.json().get("response", "") or ""
            if out.strip():
                break
    except Exception as e:
        return {"verdict": None, "reason": f"answerability call failed: {e}"}

    m = re.search(r"VERDICT:\s*(ANSWERABLE|PARTIAL|NOT[_ ]ANSWERABLE)", out, re.I)
    if not m:
        # 容錯：整段找關鍵詞（NOT 先於 ANSWERABLE，因後者是前者子字串）
        if re.search(r"\bNOT[_ ]ANSWERABLE\b", out, re.I):
            verdict = "NOT_ANSWERABLE"
        elif re.search(r"\bPARTIAL\b", out, re.I):
            verdict = "PARTIAL"
        elif re.search(r"\bANSWERABLE\b", out, re.I):
            verdict = "ANSWERABLE"
        else:
            return {"verdict": None, "reason": f"unparseable: {out[:100]}"}
    else:
        verdict = m.group(1).upper().replace(" ", "_")
    rm = re.search(r"REASON:\s*(.+)", out, re.S)
    reason = (rm.group(1).strip() if rm else out.strip())
    return {"verdict": verdict, "reason": reason}


if __name__ == "__main__":
    # 自檢：不呼叫模型，只測解析與邊界（離線可跑）。
    _rx = r"VERDICT:\s*(ANSWERABLE|PARTIAL|NOT[_ ]ANSWERABLE)"
    assert re.search(_rx, "VERDICT: ANSWERABLE\nREASON: x", re.I).group(1).upper() == "ANSWERABLE"
    assert re.search(_rx, "VERDICT: PARTIAL\nREASON: x", re.I).group(1).upper() == "PARTIAL"
    # NOT_ANSWERABLE 不可被誤判成 ANSWERABLE（子字串陷阱）
    assert re.search(_rx, "VERDICT: NOT_ANSWERABLE\nREASON: x", re.I).group(1).upper().replace(" ", "_") == "NOT_ANSWERABLE"
    assert assess_answerability("q", "")["verdict"] == "NOT_ANSWERABLE"  # 空 KB → 棄答
    print("answerability.py self-check OK")
