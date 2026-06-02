# eval/metrics.py
# 評估指標：純函數，無 I/O、無 LLM 呼叫。
# 從 pipeline 的 on_status 訊息解析每階段延遲與計數，
# 並計算論文選擇命中率、檢索 span 覆蓋率、grounding 分數。

import re

_STAGE_TAGS = [
    "planning", "retrieval", "synthesis", "synthesis-llm",
    "verification", "grounding", "translation",
]


def _normalize(s: str) -> str:
    """正規化空白與大小寫，供子字串比對。"""
    return re.sub(r'\s+', ' ', s).strip().lower()


def parse_stage_latencies(status_lines: list) -> dict:
    """
    從 status 訊息解析每階段 elapsed_ms。
    pipeline 各階段完成時會 emit「[planning] 完成 ... elapsed_ms=123」這類訊息。
    找不到的階段為 None。
    """
    text = "\n".join(status_lines)
    out = {}
    for tag in _STAGE_TAGS:
        m = re.search(rf'\[{re.escape(tag)}\]\s*完成.*?elapsed_ms=(\d+)', text)
        out[tag] = int(m.group(1)) if m else None
    m = re.search(r'total_elapsed_ms=(\d+)', text)
    out["total"] = int(m.group(1)) if m else None
    return out


def parse_counts(status_lines: list) -> dict:
    """解析 planner 選出的論文數與子問題數。"""
    text = "\n".join(status_lines)
    pc = re.search(r'paper_count=(\d+)', text)
    sc = re.search(r'subquery_count=(\d+)', text)
    return {
        "paper_count":    int(pc.group(1)) if pc else None,
        "subquery_count": int(sc.group(1)) if sc else None,
    }


def parse_grounding_score(answer: str) -> float:
    """
    從答案末尾品質報告解析 grounding_score。
    格式：<!-- grounding_score=0.875 -->；找不到回 -1.0（如 out-of-scope 無報告）。
    """
    m = re.search(r'grounding_score=(\d+\.?\d*)', answer)
    return float(m.group(1)) if m else -1.0


def paper_selection_recall(selected: list, gold: list):
    """
    gold 論文有幾成出現在 planner 選出的清單中（0.0~1.0）。
    這是 planner「單點故障」的直接量測：選錯論文，下游再強也救不回來。
    gold 為空（如 out-of-scope 題）回 None（不適用）。
    """
    if not gold:
        return None
    hit = sum(1 for g in gold if g in selected)
    return round(hit / len(gold), 3)


def retrieval_span_recall(retrieved_texts: list, gold_spans: list):
    """
    gold_spans（原文關鍵句）有幾成能在檢索回的 chunk 文字中找到（子字串比對）。
    量測檢索層是否把「答案所在的原文」撈了回來。
    gold_spans 為空回 None。
    """
    if not gold_spans:
        return None
    blob = _normalize(" ".join(retrieved_texts))
    hit = sum(1 for s in gold_spans if _normalize(s) in blob)
    return round(hit / len(gold_spans), 3)


def count_issues(answer: str) -> dict:
    """從品質報告粗略統計 CONFLICT 與未支撐陳述數。"""
    conflicts = len(re.findall(r'\[CONFLICT\]', answer))
    m = re.search(r'(\d+)\s*個陳述未找到明確論文依據', answer)
    unsupported = int(m.group(1)) if m else 0
    return {"conflicts": conflicts, "unsupported": unsupported}


def summarize(rows: list) -> dict:
    """彙整多題結果為平均指標；None 與解析失敗值自動略過。"""
    def _avg(key, sub=None):
        vals = []
        for r in rows:
            v = r.get(key) if sub is None else (r.get(key) or {}).get(sub)
            if isinstance(v, (int, float)) and v >= 0:
                vals.append(v)
        return round(sum(vals) / len(vals), 3) if vals else None

    return {
        "n_questions":          len(rows),
        "avg_grounding_score":  _avg("grounding_score"),
        "avg_paper_sel_recall": _avg("paper_selection_recall"),
        "avg_retrieval_recall": _avg("retrieval_span_recall"),
        "avg_total_ms":         _avg("latency", "total"),
        "avg_planning_ms":      _avg("latency", "planning"),
        "avg_retrieval_ms":     _avg("latency", "retrieval"),
        "avg_grounding_ms":     _avg("latency", "grounding"),
    }
