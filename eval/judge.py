# eval/judge.py
# 正確性量尺：用 LLM-judge 比對系統答案 vs reference_answer（人工依原文填的標準答案）。
# 只在 eval 時跑，不在產品 pipeline。裁判用 JUDGE_MODEL（預設 qwen3，未參與答案生成 → 降低 self-preference 偏誤）。
# 跨語言可行：系統最終答案是繁中、reference 是英文，現代 LLM 可語意比對。

import os
import sys
import re
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # 讓 standalone 也能 import config
import config as cfg

# 強化版（2026-06-23 A/B 採用）：correctness judge 只管「對 reference 的涵蓋+不矛盾」。
# 額外細節/標注的推論-推測層 OUT OF SCOPE（不獎不罰）——捏造偵測交給 grounding，非 judge 職責。
# 移除「正確答案因額外細節/推測層被罰」的假象（Q04/06/08/11），同時 Q07/10/12 真漏講/假前提仍被扣。
_JUDGE_SYSTEM = (
    "You are a scientific answer grader. You compare a CANDIDATE answer against a REFERENCE answer. "
    "The REFERENCE is the SOLE ground truth (from the source papers). Do NOT use your own outside "
    "knowledge to decide correctness: if the CANDIDATE agrees with the REFERENCE it is correct even "
    "if you believe otherwise.\n"
    "Score ONLY whether the candidate (a) covers the REFERENCE's key facts and (b) does not "
    "CONTRADICT them. Penalize ONLY: contradicting a reference fact, getting a reference fact wrong, "
    "or OMITTING a key reference fact.\n"
    "The candidate will usually contain MORE detail than the reference, and may include sections "
    "explicitly labeled as cross-paper inference or speculation (e.g. '跨文獻推論', 'insufficient "
    "literature basis', 'model speculation'). Extra detail beyond the reference, and such clearly-"
    "labeled inference/speculation sections, are OUT OF SCOPE for this score — do NOT reward and do "
    "NOT penalize them. Whether extra detail is faithful to the papers is measured separately and is "
    "not your job.\n"
    "Ignore style and language (the candidate may be in Chinese)."
)

_RUBRIC = (
    "Score 1-5 (coverage of and agreement with the REFERENCE's key facts ONLY):\n"
    "5 = all the reference's key facts present and correct, none contradicted\n"
    "4 = key facts mostly covered, one minor reference fact missing or slightly imprecise\n"
    "3 = some key reference facts missing, or one notable contradiction/error\n"
    "2 = most key reference facts missing or wrong\n"
    "1 = fails to address the reference's key facts, or contradicts them throughout\n"
    "For out-of-scope / false-premise questions, the REFERENCE says the system should refuse or "
    "flag the false premise; score 5 if the candidate does so, 1 if it asserts a fabricated answer "
    "to the false premise."
)


def judge_correctness(question: str, candidate: str, reference: str,
                      model: str = None, base_url: str = None, timeout: int = 600) -> dict:
    """
    回傳 {"score": float 0..1, "raw": int 1..5 | None, "reason": str}。
    呼叫失敗回 score=None，讓上層當「未評」處理而非 0 分。
    """
    model = model or getattr(cfg, "JUDGE_MODEL", cfg.VERIFY_MODEL)
    base_url = base_url or cfg.OLLAMA_BASE_URL
    if not (reference or "").strip() or not (candidate or "").strip():
        return {"score": None, "raw": None, "reason": "missing reference or candidate"}

    prompt = (
        f"{_RUBRIC}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"REFERENCE (ground truth):\n{reference}\n\n"
        f"CANDIDATE (system answer):\n{candidate}\n\n"
        "Output exactly two lines:\nSCORE: <1-5>\nREASON: <one sentence>"
    )
    try:
        resp = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "system": _JUDGE_SYSTEM,
                "prompt": prompt,
                "stream": False,
                # 關 thinking：打分是有界任務，qwen3 開思考會把 num_predict 吃光、SCORE 吐不出來。
                "think": False,
                "options": {"temperature": 0.0, "num_predict": 1024, "num_ctx": 16384, "thinking": False},
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        out = resp.json().get("response", "")
    except Exception as e:
        return {"score": None, "raw": None, "reason": f"judge call failed: {e}"}

    m = re.search(r"SCORE:\s*([1-5])", out)
    if not m:
        m = re.search(r"\b([1-5])\s*/\s*5\b", out)  # 容錯：模型寫成 N/5
    if not m:
        return {"score": None, "raw": None, "reason": f"unparseable judge output: {out[:120]}"}
    raw = int(m.group(1))
    rm = re.search(r"REASON:\s*(.+)", out, re.S)
    reason = (rm.group(1).strip()[:300] if rm else out[:200].strip())
    return {"score": (raw - 1) / 4.0, "raw": raw, "reason": reason}


if __name__ == "__main__":
    # 自檢：不呼叫模型，只測 parsing 與邊界（離線可跑）
    import types
    fake = '{"response": "SCORE: 4\\nREASON: minor value missing."}'
    # 直接測 regex 解析邏輯
    assert re.search(r"SCORE:\s*([1-5])", "SCORE: 4\nREASON: x").group(1) == "4"
    assert re.search(r"\b([1-5])\s*/\s*5\b", "I give 3/5 overall").group(1) == "3"
    assert judge_correctness("q", "ans", "")["score"] is None     # 空 reference → 未評
    assert judge_correctness("q", "", "ref")["score"] is None     # 空 candidate → 未評
    print("judge.py self-check OK")
