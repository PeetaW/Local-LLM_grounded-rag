# rag/citation_grounding.py
# 負責 Citation Grounding：用 mDeBERTa 判斷答案陳述是否有 chunk 依據
# 負責 Citation Grounding：用 mDeBERTa 多語言 NLI 模型判斷答案陳述是否有 chunk 依據
#
# 本次修改重點：
# 1. NLI 模型換成 mDeBERTa-v3-base-mnli-xnli（支援中英跨語言，278MB，比原版 1.74GB 小）
# 2. 修正 NLI 使用方式：改成 premise=chunk, hypothesis=sentence 的正確 pair 比對
# 3. 對每個 chunk 逐一比對，取最高分，避免 token 截斷
# 4. Vectara 模型載入加入 trust_remote_code=True（修正載入失敗問題）

import re
import torch

_nli_pipeline = None

def get_nli_pipeline():
    global _nli_pipeline
    if _nli_pipeline is None:
        from transformers import pipeline
        print("  📦 載入 mDeBERTa 多語言 NLI 模型...")
        _nli_pipeline = pipeline(
            "zero-shot-classification",
            model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
            device=0 if torch.cuda.is_available() else -1,
        )
        print("  ✅ mDeBERTa 載入完成")
    return _nli_pipeline


def split_into_sentences(text: str) -> list:
    text = re.sub(r'\*\*|##|###|【.*?】', '', text)
    sentences = re.split(r'(?<=[。！？\.\!\?])\s*|\n+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) >= 10]
    return sentences


def check_citation_grounding(sentences: list, chunks: list) -> list:
    if not sentences or not chunks:
        return []

    nli = get_nli_pipeline()
    results = []

    for sentence in sentences:
        best_score = 0.0
        best_chunk_id = None

        for chunk in chunks:
            chunk_text = chunk["text"][:600]
            try:
                result = nli(
                    chunk_text,
                    candidate_labels=[sentence],
                    hypothesis_template="{}",
                    multi_label=False,
                )
                score = result["scores"][0]
                if score > best_score:
                    best_score = score
                    best_chunk_id = chunk["id"]
            except Exception:
                continue

        results.append({
            "sentence": sentence,
            "supported": best_score >= 0.5,
            "confidence": round(best_score, 3),
            "best_chunk": best_chunk_id,
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

    unsupported = [r for r in citation_results if not r["supported"]]
    if not unsupported:
        lines.append("✅ **所有陳述均有論文依據**\n")
    else:
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