# rag/answerability.py
# 可答性 gate（answerability gate）— Stage 3 之後、Stage 4 生成之前的前置判斷。
# 問題：檢索/蒸餾出的事實是否「真的包含」問題要的答案，而非只是「同主題」。
# 比既有的 rag_found_anything（只看檢索非空）更細：擋掉「相關但不含答案」→ 走誠實棄答。
# 用 LLM_MODEL（gemma4，與 Stage 3 同模型 → 不觸發 VRAM swap）。保守偏 ANSWERABLE，避免誤殺好答案。
import re
import requests
import config as cfg

_SYSTEM = (
    "You decide whether a set of distilled facts CONTAINS the information needed to answer a "
    "question. You are NOT answering the question yourself. Judge ONE thing: do the facts contain "
    "the underlying information the question asks about? Being about the SAME TOPIC is NOT enough — "
    "the facts must actually contain the requested information.\n"
    "IMPORTANT — synthesis questions: if the question asks to COMPARE, CONTRAST, or AGGREGATE, the "
    "facts are ANSWERABLE as long as they contain the underlying per-item information needed to "
    "build that answer (e.g. the per-route yields/costs needed for a comparison). The comparison "
    "itself need NOT be pre-written in the facts — assembling it is the answer step. Judge whether "
    "the INPUTS are present, not whether the finished synthesized form is present.\n"
    "Be conservative toward ANSWERABLE: if the facts contain a direct OR substantially partial "
    "answer, or the underlying inputs for a synthesis question, say ANSWERABLE. Say NOT_ANSWERABLE "
    "ONLY when the underlying information itself is absent — the question asks for value/result X "
    "and the facts simply never state X (no amount of synthesis could produce it). A question "
    "resting on a false premise the facts contradict is also NOT_ANSWERABLE (the requested thing "
    "does not exist in the facts)."
)


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
        "Output exactly two lines:\nVERDICT: <ANSWERABLE|NOT_ANSWERABLE>\nREASON: <one sentence>"
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

    m = re.search(r"VERDICT:\s*(ANSWERABLE|NOT_ANSWERABLE)", out, re.I)
    if not m:
        # 容錯：整段找關鍵詞
        if re.search(r"\bNOT[_ ]ANSWERABLE\b", out, re.I):
            verdict = "NOT_ANSWERABLE"
        elif re.search(r"\bANSWERABLE\b", out, re.I):
            verdict = "ANSWERABLE"
        else:
            return {"verdict": None, "reason": f"unparseable: {out[:100]}"}
    else:
        verdict = m.group(1).upper().replace(" ", "_")
    rm = re.search(r"REASON:\s*(.+)", out, re.S)
    reason = (rm.group(1).strip()[:200] if rm else out[:160].strip())
    return {"verdict": verdict, "reason": reason}


if __name__ == "__main__":
    # 自檢：不呼叫模型，只測解析與邊界（離線可跑）。
    assert re.search(r"VERDICT:\s*(ANSWERABLE|NOT_ANSWERABLE)", "VERDICT: ANSWERABLE\nREASON: x", re.I).group(1) == "ANSWERABLE"
    assert re.search(r"\bNOT[_ ]ANSWERABLE\b", "I think this is NOT ANSWERABLE here", re.I)
    assert assess_answerability("q", "")["verdict"] == "NOT_ANSWERABLE"  # 空 KB → 棄答
    print("answerability.py self-check OK")
