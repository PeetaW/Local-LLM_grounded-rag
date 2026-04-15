# rag/citation_grounding.py
# 負責 Citation Grounding：用 mDeBERTa 多語言 NLI 模型判斷答案陳述是否有 chunk 依據
#
# V3 修改重點：
# 1. 改用標準 NLI 三分類模式（entailment / neutral / contradiction）
#    取代 zero-shot-classification，以同時取得 entailment 和 contradiction score
# 2. NLI_CONTRADICTION_ENABLED=True 時標記知識庫內部矛盾（contradiction > 0.7）
# 3. check_citation_grounding() 回傳格式新增 contradiction_detected / status 欄位
#    （向下相容：舊程式不讀新欄位則不受影響）

import re
import torch
import config as cfg

_nli_model = None
_nli_tokenizer = None
_NLI_LABEL_MAP = None   # {0: "contradiction", 1: "neutral", 2: "entailment"} 或依模型決定


def _get_nli_model():
    """
    載入 mDeBERTa NLI 三分類模型（singleton）。
    使用 AutoModelForSequenceClassification 以同時取得三個 label 的 logits。
    """
    global _nli_model, _nli_tokenizer, _NLI_LABEL_MAP
    if _nli_model is None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        model_name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
        print("  📦 載入 mDeBERTa 多語言 NLI 模型（三分類模式）...")
        _nli_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _nli_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        device = 0 if torch.cuda.is_available() else -1
        if device == 0:
            _nli_model = _nli_model.cuda()
        _nli_model.eval()

        # 從模型 config 取得 label 對應關係
        id2label = _nli_model.config.id2label  # e.g. {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}
        _NLI_LABEL_MAP = {v.upper(): k for k, v in id2label.items()}
        print(f"  ✅ mDeBERTa 載入完成（labels: {id2label}）")
    return _nli_model, _nli_tokenizer, _NLI_LABEL_MAP


def _run_nli(premise: str, hypothesis: str) -> dict:
    """
    對單一 (premise, hypothesis) pair 執行 NLI，
    回傳 {"entailment": float, "neutral": float, "contradiction": float}。
    """
    import torch

    model, tokenizer, label_map = _get_nli_model()
    inputs = tokenizer(
        premise[:512], hypothesis[:256],
        return_tensors="pt", truncation=True, max_length=512,
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()

    result = {}
    for label, idx in label_map.items():
        result[label.lower()] = round(probs[idx], 4)
    # 確保三個 key 都存在（防止模型 label 名不同）
    for key in ("entailment", "neutral", "contradiction"):
        result.setdefault(key, 0.0)
    return result


def split_into_sentences(text: str) -> list:
    text = re.sub(r'\*\*|##|###|【.*?】', '', text)
    sentences = re.split(r'(?<=[。！？\.\!\?])\s*|\n+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) >= 10]
    return sentences


def check_citation_grounding(sentences: list, chunks: list) -> list:
    """
    對每個 sentence 在所有 chunks 中找最佳支撐 chunk。
    回傳格式（V3 新增 contradiction_detected / status）：
    {
        "sentence": str,
        "supported": bool,
        "confidence": float,        # entailment score
        "best_chunk": str,
        "contradiction_detected": bool,   # 僅當 NLI_CONTRADICTION_ENABLED=True
        "contradiction_source": str,      # 矛盾最強的 chunk id
        "status": "SUPPORTED" | "CONFLICT" | "UNSUPPORTED",
    }
    """
    if not sentences or not chunks:
        return []

    results = []

    for sentence in sentences:
        best_entail = 0.0
        best_chunk_id = None
        best_contradict = 0.0
        best_contradict_id = None

        for chunk in chunks:
            chunk_text = chunk["text"][:512]
            try:
                scores = _run_nli(premise=chunk_text, hypothesis=sentence)
                e_score = scores["entailment"]
                c_score = scores["contradiction"]

                if e_score > best_entail:
                    best_entail = e_score
                    best_chunk_id = chunk.get("id", chunk.get("source", ""))

                if cfg.NLI_CONTRADICTION_ENABLED and c_score > best_contradict:
                    best_contradict = c_score
                    best_contradict_id = chunk.get("id", chunk.get("source", ""))

            except Exception:
                continue

        contradiction_detected = (
            cfg.NLI_CONTRADICTION_ENABLED and best_contradict > 0.7
        )

        if best_entail >= 0.5:
            status = "CONFLICT" if contradiction_detected else "SUPPORTED"
        else:
            status = "CONFLICT" if contradiction_detected else "UNSUPPORTED"

        results.append({
            "sentence": sentence,
            "supported": best_entail >= 0.5,
            "confidence": round(best_entail, 3),
            "best_chunk": best_chunk_id,
            "contradiction_detected": contradiction_detected,
            "contradiction_source": best_contradict_id if contradiction_detected else None,
            "status": status,
        })

    return results


def compute_grounding_score(citation_results: list) -> float:
    """
    計算整體依據率（0.0~1.0）。
    1.0 = 所有句子都有依據，0.0 = 完全沒有依據。
    這個分數用來取代原本的 hallucination_score 作為品質門檻。
    """
    if not citation_results:
        return 1.0  # 沒有句子，預設通過
    supported = sum(1 for r in citation_results if r["supported"])
    return round(supported / len(citation_results), 3)


def format_grounding_report(citation_results: list) -> str:
    """
    移除 hallucination_score 參數，改用 citation grounding 結果產生報告。
    同時在報告末尾附上整體依據率，供 api.py 解析品質門檻用。
    """
    lines = ["\n\n---", "📋 **答案品質報告**\n"]

    grounding_score = compute_grounding_score(citation_results)

    if grounding_score >= 0.8:
        emoji = "✅"
        label = "高（答案高度忠實於論文內容）"
    elif grounding_score >= 0.5:
        emoji = "⚠️"
        label = "中（部分陳述需要確認）"
    else:
        emoji = "❌"
        label = "低（建議重新查詢或縮小問題範圍）"

    lines.append(f"{emoji} **整體論文依據率**：{grounding_score:.1%}　{label}\n")

    # ── 矛盾偵測摘要 ──────────────────────────────────
    if cfg.NLI_CONTRADICTION_ENABLED:
        conflicts = [r for r in citation_results if r.get("status") == "CONFLICT"]
        if conflicts:
            lines.append(f"⚠️  **偵測到 {len(conflicts)} 個陳述與知識庫存在矛盾：**\n")
            for r in conflicts:
                src = f"（矛盾來源：{r['contradiction_source']}）" if r.get("contradiction_source") else ""
                lines.append(f"- [CONFLICT] {r['sentence']}{src}")
            lines.append("")

    unsupported = [r for r in citation_results if not r["supported"] and r.get("status") != "CONFLICT"]
    if not unsupported and not [r for r in citation_results if r.get("status") == "CONFLICT"]:
        lines.append("✅ **所有陳述均有論文依據**\n")
    elif unsupported:
        lines.append(
            f"⚠️  **以下 {len(unsupported)} 個陳述未找到明確論文依據，請謹慎參考：**\n"
        )
        for r in unsupported:
            chunk_info = f"，最近似來源：{r['best_chunk']}" if r.get("best_chunk") else ""
            lines.append(f"- {r['sentence']}（信心度：{r['confidence']:.1%}{chunk_info}）")

    # ★ 附上機器可解析的分數，供 api.py 的品質門檻使用
    lines.append(f"\n<!-- grounding_score={grounding_score:.3f} -->")
    lines.append("---")
    return "\n".join(lines)

# ── 推測語氣偵測 ──────────────────────────────────────

_SPECULATION_KEYWORDS = [
    # 中文
    "推測", "而得", "可能", "也許", "或許", "應該", "好像",
    "揣測", "似乎", "指向", "是否", "臆測", "猜想",
    "傾向於", "暗示", "顯示", "意味著", "有可能",
    "據推測", "初步認為", "有理由相信",
    # 英文
    "assume", "suggest", "reckon", "maybe", "might",
    "probably", "possible", "imply", "indicate",
    "appear to", "seem to", "likely", "hypothesize",
    "speculate", "tend to", "point to", "could be",
    "would suggest",
]

_NEGATION_PREFIXES_ZH = ["不", "非", "無", "沒有", "絕非", "絕不", "否"]
_NEGATION_PREFIXES_EN = ["not", "never", "impossible", "unlikely"]

# 建立否定+推測的組合 pattern（排除這些）
# 例如：「不可能」「不太可能」「never suggest」
_NEGATED_SPECULATION_RE = re.compile(
    '|'.join(
        [f"{neg}{kw}" for neg in _NEGATION_PREFIXES_ZH
                      for kw in ["可能", "太可能", "應該"]]
        + [f"{neg}\\s+{kw}" for neg in _NEGATION_PREFIXES_EN
                            for kw in ["suggest", "imply", "indicate",
                                       "possible", "likely"]]
    ),
    re.IGNORECASE
)

_SPECULATION_RE = re.compile(
    '|'.join(_SPECULATION_KEYWORDS), re.IGNORECASE
)


def has_speculation_keywords(text: str) -> bool:
    """
    偵測文字中是否有推測性語氣。
    排除「否定前綴 + 推測詞」的組合（不可能、never suggest 等）。
    雙重否定不處理，直接當否定看待。
    回傳 True 代表有推測語氣。
    """
    # 先找出所有推測詞的位置
    if not _SPECULATION_RE.search(text):
        return False

    # 把否定+推測的組合從文字中移除，再檢查是否還有推測詞
    text_without_negated = _NEGATED_SPECULATION_RE.sub("", text)
    return bool(_SPECULATION_RE.search(text_without_negated))


# ── 多文獻指稱偵測 ────────────────────────────────────

_MULTI_PAPER_KEYWORDS = [
    # 中文
    "兩者", "兩篇", "兩個研究", "多篇", "各論文", "綜合",
    "比較", "相比", "對照", "一致", "差異", "不同研究",
    "前者", "後者", "分別", "各自",
    # 英文
    "both studies", "both papers", "compared to",
    "in contrast", "whereas", "while.*study",
    "across studies", "multiple studies",
    "former.*latter", "respectively",
]

_MULTI_PAPER_RE = re.compile(
    '|'.join(_MULTI_PAPER_KEYWORDS), re.IGNORECASE
)


def has_multi_paper_reference(text: str) -> bool:
    """
    偵測文字中是否有跨文獻比較的指稱。
    回傳 True 代表有多文獻指稱。
    """
    return bool(_MULTI_PAPER_RE.search(text))


# ══════════════════════════════════════════════════════════════════
#  2-B：子命題拆解驗證（NLI_DECOMPOSE_ENABLED 控制）
# ══════════════════════════════════════════════════════════════════

def decompose_and_verify(conclusion: str, facts: list[dict]) -> dict:
    """
    對一個結論句子做子命題拆解驗證。

    流程：
    1. 呼叫 gemma4:31b 把結論拆成子命題 JSON list
    2. 每個子命題對所有 facts 跑 NLI，取最高 entailment score
    3. 依閾值標記 SUPPORTED / INFERENCE_BRIDGE / UNSUPPORTED

    輸出格式：
    {
        "conclusion": str,
        "sub_claims": [
            {
                "claim": str,
                "grounding_score": float,
                "source": str,
                "status": "SUPPORTED" | "INFERENCE_BRIDGE" | "UNSUPPORTED"
            }
        ],
        "chain_complete": bool   # 所有子命題都有 SUPPORTED 或 INFERENCE_BRIDGE
    }

    若 NLI_DECOMPOSE_ENABLED=False，直接回傳空結果。
    """
    if not cfg.NLI_DECOMPOSE_ENABLED:
        return {"conclusion": conclusion, "sub_claims": [], "chain_complete": True}

    import requests as _req
    import json as _json

    # ── Step 1：呼叫 LLM 拆解子命題 ──────────────────────────
    prompt = (
        f"請將以下結論句子拆解成 2-4 個獨立的子命題，每個子命題應能獨立被文獻支撐或反駁。\n"
        f"只輸出 JSON 陣列，格式：[\"子命題1\", \"子命題2\", ...]\n\n"
        f"結論：{conclusion}"
    )
    try:
        resp = _req.post(
            f"{cfg.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": cfg.SYNTHESIS_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 512, "num_ctx": 4096},
            },
            timeout=120,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "[]").strip()
        raw = re.sub(r'```json|```', '', raw).strip()
        sub_claims_text = _json.loads(raw)
        if not isinstance(sub_claims_text, list):
            sub_claims_text = [conclusion]
    except Exception:
        sub_claims_text = [conclusion]

    # ── Step 2：對每個子命題跑 NLI ───────────────────────────
    sub_claims = []
    for claim in sub_claims_text:
        best_score = 0.0
        best_source = None
        for fact in facts:
            try:
                scores = _run_nli(premise=fact["text"][:512], hypothesis=claim)
                if scores["entailment"] > best_score:
                    best_score = scores["entailment"]
                    best_source = fact.get("id", fact.get("source", ""))
            except Exception:
                continue

        if best_score >= 0.65:
            status = "SUPPORTED"
        elif best_score >= 0.4:
            status = "INFERENCE_BRIDGE"
        else:
            status = "UNSUPPORTED"

        sub_claims.append({
            "claim": claim,
            "grounding_score": round(best_score, 3),
            "source": best_source,
            "status": status,
        })

    chain_complete = all(sc["status"] != "UNSUPPORTED" for sc in sub_claims)

    return {
        "conclusion": conclusion,
        "sub_claims": sub_claims,
        "chain_complete": chain_complete,
    }


# ══════════════════════════════════════════════════════════════════
#  2-C：多來源聯合驗證（NLI_JOINT_VERIFY_ENABLED 控制）
# ══════════════════════════════════════════════════════════════════

def joint_verify(claim: str, facts: list[dict]) -> dict:
    """
    對一個子命題做多來源聯合驗證。

    流程：
    1. 對每個 fact 個別跑 NLI，取 entailment score
    2. 取 top-3 highest entailment score 的 facts
    3. 把 top-3 facts 文字拼接後，再跑一次 NLI
    4. individual scores 低但 joint score 高 → INFERENCE_BRIDGE（跨文獻推論橋接）

    輸出格式：
    {
        "claim": str,
        "individual_scores": [float, float, float],
        "joint_score": float,
        "is_inference_bridge": bool,
        "bridge_sources": [str, str, ...]
    }

    若 NLI_JOINT_VERIFY_ENABLED=False，直接回傳空結果。
    """
    if not cfg.NLI_JOINT_VERIFY_ENABLED:
        return {
            "claim": claim,
            "individual_scores": [],
            "joint_score": 0.0,
            "is_inference_bridge": False,
            "bridge_sources": [],
        }

    # ── Step 1：個別跑 NLI，收集所有分數 ────────────────────
    scored_facts = []
    for fact in facts:
        try:
            scores = _run_nli(premise=fact["text"][:512], hypothesis=claim)
            scored_facts.append({
                "source": fact.get("id", fact.get("source", "")),
                "text": fact["text"][:512],
                "score": scores["entailment"],
            })
        except Exception:
            continue

    if not scored_facts:
        return {
            "claim": claim,
            "individual_scores": [],
            "joint_score": 0.0,
            "is_inference_bridge": False,
            "bridge_sources": [],
        }

    # ── Step 2：取 top-3（依 entailment score 排序）────────
    top3 = sorted(scored_facts, key=lambda x: x["score"], reverse=True)[:3]
    individual_scores = [round(f["score"], 3) for f in top3]
    bridge_sources = [f["source"] for f in top3]

    # ── Step 3：拼接 top-3 文字後聯合驗證 ───────────────────
    joint_premise = "\n\n".join(f["text"] for f in top3)
    try:
        joint_scores = _run_nli(premise=joint_premise[:1024], hypothesis=claim)
        joint_score = round(joint_scores["entailment"], 3)
    except Exception:
        joint_score = max(individual_scores) if individual_scores else 0.0

    # ── 判斷是否為跨文獻推論橋接 ─────────────────────────────
    avg_individual = sum(individual_scores) / len(individual_scores) if individual_scores else 0.0
    is_inference_bridge = (avg_individual < 0.5) and (joint_score >= 0.65)

    return {
        "claim": claim,
        "individual_scores": individual_scores,
        "joint_score": joint_score,
        "is_inference_bridge": is_inference_bridge,
        "bridge_sources": bridge_sources if is_inference_bridge else [],
    }