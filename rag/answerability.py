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
_VALUE_QUESTION_MARKERS = ("value", "values", "reported value", "數值", "數據", "值")
_COMPARISON_ARM_RE = re.compile(
    r"\b(?:lower|higher|greater|less|more|smaller|larger)\s+than\s+"
    r"(?:(?:that|those)\s+of\s+)?([^.;(\n]{2,80})",
    re.IGNORECASE,
)
_PREMISE_VALUE_QUERY_RE = re.compile(
    r"^\s*(?:since|because|assuming|given\s+that)\b.*\bvalues?\b",
    re.IGNORECASE | re.DOTALL,
)
_REQUESTED_VALUE_RE = re.compile(
    r"\bwhat\s+(?:are\s+)?(?:the\s+)?(?P<metric>[^?]{2,80}?)\s+values?\b",
    re.IGNORECASE,
)
_FACT_LINE_RE = re.compile(
    r"^\s*\[(?:Fact|事實)\s*\d+\]\s*(?P<claim>.*?)\s*"
    r"\((?:Source|來源)\s*[:：]\s*(?P<source>.*?)\)\s*\.?\s*$",
    re.IGNORECASE,
)
_ALTERNATIVE_ROUTE_RE = re.compile(
    r"\b(?:infus\w*|inject\w*|intraven\w*)\b",
    re.IGNORECASE,
)


def _missing_value_arm(question: str, knowledge_base: str) -> str:
    """Return a comparison arm named in the KB but never given its own value."""
    if not any(marker in (question or "").lower() for marker in _VALUE_QUESTION_MARKERS):
        return ""

    lines = []
    for line in (knowledge_base or "").splitlines():
        if not line.strip():
            continue
        clean = re.sub(
            r"\((?:Source|來源)\s*[:：].*?\)\s*$",
            "",
            " ".join(line.split()),
            flags=re.IGNORECASE,
        )
        lines.append(re.sub(r"^\[(?:Fact|事實)\s*\d+\]\s*", "", clean, flags=re.IGNORECASE))
    for line in lines:
        for match in _COMPARISON_ARM_RE.finditer(line):
            arm = re.sub(r"^(?:the|an?)\s+", "", match.group(1).strip(), flags=re.IGNORECASE)
            if re.search(r"\d", arm):
                continue
            arm_pattern = re.compile(re.escape(arm), re.IGNORECASE)
            quantified = False
            for candidate in lines:
                arm_match = arm_pattern.search(candidate)
                if not arm_match:
                    continue
                before = candidate[max(0, arm_match.start() - 70):arm_match.start()]
                after = candidate[arm_match.end():arm_match.end() + 70]
                if re.search(
                    r"(?:lower|higher|greater|less|more|smaller|larger)\s+than\s+"
                    r"(?:(?:that|those)\s+of\s+)?$",
                    before,
                    re.IGNORECASE,
                ):
                    quantified = bool(re.search(r"\d", after))
                else:
                    quantified = bool(re.search(r"\d", before[-50:] + after[:50]))
                if quantified:
                    break
            if not quantified and re.search(r"\b(?:alone|only)\b", arm, re.IGNORECASE):
                arm_core = re.sub(r"\b(?:alone|only)\b", "", arm, flags=re.IGNORECASE).strip()
                core_pattern = re.compile(re.escape(arm_core), re.IGNORECASE)
                for candidate in lines:
                    core_match = core_pattern.search(candidate)
                    if not core_match:
                        continue
                    nearby = candidate[
                        max(0, core_match.start() - 90):core_match.end() + 90
                    ]
                    combined = bool(re.search(
                        r"\b(?:combined|combination|pre-plus)\b|pre\s*\+\s*co|addition of preincubation",
                        nearby,
                        re.IGNORECASE,
                    ))
                    if "preincubation" in arm_core.lower() and "co-incubation" in nearby.lower():
                        combined = True
                    if not combined and re.search(r"\d", nearby):
                        quantified = True
                        break
            if not quantified:
                return arm
    return ""


def _missing_premise_value(question: str, knowledge_base: str) -> str:
    if not _PREMISE_VALUE_QUERY_RE.search(question or ""):
        return ""
    match = _REQUESTED_VALUE_RE.search(question or "")
    if not match:
        return ""
    metric = " ".join(match.group("metric").split()).strip(" ,.;:")
    tokens = [
        token for token in re.findall(r"[a-z0-9]+", metric.lower())
        if len(token) > 3 and token not in {"reported", "paper", "papers", "these"}
    ]
    if not tokens:
        return ""
    for line in (knowledge_base or "").splitlines():
        lower = line.lower()
        if all(token in lower for token in tokens) and re.search(r"\d", line):
            return ""
    return metric


def _reported_alternative_route(knowledge_base: str) -> tuple[str, str] | None:
    candidates = []
    for line in (knowledge_base or "").splitlines():
        match = _FACT_LINE_RE.match(line)
        if not match or not _ALTERNATIVE_ROUTE_RE.search(match.group("claim")):
            continue
        claim = match.group("claim").strip().rstrip(".")
        score = len(_ALTERNATIVE_ROUTE_RE.findall(claim)) + 2 * bool(re.search(r"\d", claim))
        candidates.append((score, claim, match.group("source").strip()))
    if not candidates:
        return None
    _, claim, source = max(candidates, key=lambda item: item[0])
    return claim, source

# Phase 2 路由用的告示文字。
ABSTAIN_NOTICE = (
    "⚠️ **誠實棄答**：檢索到的內容與此問題相關，但並不包含可直接回答的資訊。"
    "為避免編造不存在的數據，本系統選擇不作答。建議換個問法，或確認該主題是否涵蓋於文獻庫中。\n"
)
WEAK_NOTICE = (
    "⚠️ **資料可能不足**：檢索內容僅部分涵蓋此問題，以下回答可能不完整，請謹慎參考並自行查證。\n\n"
)


def gate_route(verdict: str, question: str = "", knowledge_base: str = ""):
    """verdict → (abstain: bool, notice: str)。
    NOT_ANSWERABLE→硬棄答（跳生成）；PARTIAL→軟警告（照常生成、加橫幅）；其餘→正常。
    verdict=None（未判定/呼叫失敗）保守當正常，不棄答。"""
    if verdict == "NOT_ANSWERABLE":
        if getattr(cfg, "FALSE_PREMISE_RECOVERY_ENABLED", False):
            metric = _missing_premise_value(question, knowledge_base)
            if metric:
                route = _reported_alternative_route(knowledge_base)
                correction = (
                    "⚠️ **前提更正**：檢索文獻沒有報告可供回答的 "
                    f"`{metric}` 數值，因此不能把問題中的前提視為已成立，也不會臆測數值。"
                )
                if route:
                    claim, source = route
                    correction += (
                        "\n\n文獻實際報告的是不同的給藥途徑或 regimen：\n\n"
                        f"- {claim} [{source}]\n"
                    )
                else:
                    correction += "\n"
                return True, correction
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
    if (
        verdict == "ANSWERABLE"
        and getattr(cfg, "PARTIAL_RECOVERY_DETERMINISTIC_GUARDS_ENABLED", False)
    ):
        missing_arm = _missing_value_arm(question, knowledge_base)
        if missing_arm:
            verdict = "PARTIAL"
            reason = f"The facts compare against '{missing_arm}' but do not report that arm's requested value."
    if getattr(cfg, "FALSE_PREMISE_RECOVERY_ENABLED", False):
        missing_metric = _missing_premise_value(question, knowledge_base)
        if missing_metric:
            verdict = "NOT_ANSWERABLE"
            reason = (
                "The facts do not report a requested value for "
                f"'{missing_metric}', so the question premise cannot be accepted."
            )
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
