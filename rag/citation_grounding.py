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
import time
import torch
import config as cfg
from rag.query_grounding_flow import split_into_sentences, _cited_sources_in_sentence

# ── grounding 階段計時拆解（NLI vs LLM）──────────────────
# 讓 NLI device A/B 能區分「CPU NLI 變慢」與「gemma4 因 VRAM 紓解變快」。
_grounding_nli_time = 0.0   # mDeBERTa forward pass 累計秒數
_grounding_llm_time = 0.0   # grounding 內 gemma4 呼叫累計秒數


def reset_grounding_timers():
    global _grounding_nli_time, _grounding_llm_time
    _grounding_nli_time = 0.0
    _grounding_llm_time = 0.0


def get_grounding_timers() -> dict:
    return {"nli_s": round(_grounding_nli_time, 1), "llm_s": round(_grounding_llm_time, 1)}


def _add_nli_time(dt: float):
    global _grounding_nli_time
    _grounding_nli_time += dt


def _add_llm_time(dt: float):
    global _grounding_llm_time
    _grounding_llm_time += dt


_nli_model = None
_nli_tokenizer = None
_NLI_LABEL_MAP = None   # {0: "contradiction", 1: "neutral", 2: "entailment"} 或依模型決定
_nli_error_reported = False


def _report_nli_error(e: Exception):
    global _nli_error_reported
    if not _nli_error_reported:
        msg = str(e).replace("\n", " ")[:240]
        print(f"  [NLI-error] {type(e).__name__}: {msg}")
        _nli_error_reported = True


def _nli_wants_cuda() -> bool:
    import torch
    want = getattr(cfg, "NLI_DEVICE", "auto").lower()
    return torch.cuda.is_available() and (want == "cuda" or want == "auto")


def _get_nli_model():
    """
    載入 mDeBERTa NLI 三分類模型（singleton）。
    使用 AutoModelForSequenceClassification 以同時取得三個 label 的 logits。
    每次取用都確保模型在目標 device：grounding 結束會被 release_nli_gpu() 搬下 GPU，
    下一題在此自動搬回 GPU（搬移 ~0.5GB，毫秒級，可忽略）。
    """
    global _nli_model, _nli_tokenizer, _NLI_LABEL_MAP
    import torch
    if _nli_model is None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        model_name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
        model_path = model_name
        local_only = False
        try:
            from huggingface_hub import snapshot_download
            model_path = snapshot_download(model_name, local_files_only=True)
            local_only = True
        except Exception as e:
            print(f"  [NLI] local cache lookup failed; using HF id ({type(e).__name__})")

        print("  Loading mDeBERTa multilingual NLI model (3-way)...")
        _nli_tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_only)
        _nli_model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=local_only)
        _nli_model.eval()
        id2label = _nli_model.config.id2label  # e.g. {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}
        _NLI_LABEL_MAP = {v.upper(): k for k, v in id2label.items()}
        print(f"  mDeBERTa loaded (labels: {id2label})")

    target = "cuda" if _nli_wants_cuda() else "cpu"
    if str(next(_nli_model.parameters()).device).split(":")[0] != target:
        _nli_model = _nli_model.to(target)
        print(f"  mDeBERTa device: {target}")
    return _nli_model, _nli_tokenizer, _NLI_LABEL_MAP


def release_nli_gpu():
    """
    grounding 跑完把 mDeBERTa 搬下 GPU + 清快取，VRAM 完整讓給 gemma4 翻譯。
    下一題 _get_nli_model() 會自動把它搬回 GPU。NLI_DEVICE=cpu 時本來就在 CPU，無作用。
    """
    global _nli_model
    import torch
    if _nli_model is not None and next(_nli_model.parameters()).is_cuda:
        _nli_model = _nli_model.cpu()
        torch.cuda.empty_cache()


def _window_text(text: str, size: int = 1400, overlap: int = 300) -> list[str]:
    """
    把長 chunk 切成重疊窗。mDeBERTa premise 上限 ~512 token(~1900 字元)，
    chunk 常達 3000+ 字元；直接截斷會漏掉落在後段的事實。
    1400 字元 ≈ 370 token，留足空間給 hypothesis；overlap 避免事實被切在窗邊界。
    """
    if len(text) <= size:
        return [text]
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step) if text[i:i + size]]


def _run_nli(premise: str, hypothesis: str) -> dict:
    """
    對單一 (premise, hypothesis) pair 執行 NLI，
    回傳 {"entailment": float, "neutral": float, "contradiction": float}。
    """
    import torch

    model, tokenizer, label_map = _get_nli_model()
    inputs = tokenizer(
        premise, hypothesis,
        return_tensors="pt", truncation="only_first", max_length=512,
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    _t = time.perf_counter()
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()
    _add_nli_time(time.perf_counter() - _t)

    result = {}
    for label, idx in label_map.items():
        result[label.lower()] = round(probs[idx], 4)
    # 確保三個 key 都存在（防止模型 label 名不同）
    for key in ("entailment", "neutral", "contradiction"):
        result.setdefault(key, 0.0)
    return result


def _run_nli_batch(premises: list, hypotheses: list, batch_size: int = None) -> list:
    """
    對多組 (premise, hypothesis) pair 批次執行 NLI（一次矩陣運算多組）。
    premises 與 hypotheses 必須等長；回傳等長的 score dict 清單。
    結果與逐組呼叫 _run_nli 相同（padding token 被 attention mask 忽略），
    只是省掉 per-call 的 Python / 資料傳輸開銷。
    以 batch_size 分批，避免一次塞太多爆記憶體。
    """
    import torch

    if not premises:
        return []
    if batch_size is None:
        batch_size = getattr(cfg, "NLI_BATCH_SIZE", 16)

    model, tokenizer, label_map = _get_nli_model()
    device = next(model.parameters()).device

    out = []
    for i in range(0, len(premises), batch_size):
        bp = premises[i:i + batch_size]
        bh = hypotheses[i:i + batch_size]
        inputs = tokenizer(
            bp, bh,
            return_tensors="pt", truncation="only_first",
            max_length=512, padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        _t = time.perf_counter()
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().tolist()  # [B, num_labels]
        _add_nli_time(time.perf_counter() - _t)
        for row in probs:
            r = {}
            for label, idx in label_map.items():
                r[label.lower()] = round(row[idx], 4)
            for key in ("entailment", "neutral", "contradiction"):
                r.setdefault(key, 0.0)
            out.append(r)
    return out


def _latex_to_plain(text: str) -> str:
    r"""
    將 LaTeX 數學式轉換為可讀的科學表達式，保留化學式的語義資訊。

    範例：
      $\text{FeSO}_4\cdot 7\text{H}_2\text{O}$  →  FeSO4·7H2O
      $\text{KBH}_4$                             →  KBH4
      $\text{Fe}^{2+}$                           →  Fe2+
      $\text{NH}_2$                              →  NH2
      $\text{CO}_3^{2-}$                         →  CO32-
    """
    def _convert_math_block(m: re.Match) -> str:
        inner = m.group(1)
        # \text{X} → X（移除 \text 包裝，保留內容）
        inner = re.sub(r'\\text\{([^}]+)\}', r'\1', inner)
        # _{...} 下標：移除大括號，保留內容（如 _{4} → 4, _{2+} → 2+）
        inner = re.sub(r'_\{([^}]+)\}', r'\1', inner)
        # _X 單字元下標（如 _4 → 4）
        inner = re.sub(r'_(\w)', r'\1', inner)
        # ^{...} 上標：移除大括號，保留內容（如 ^{2+} → 2+, ^{-} → -）
        inner = re.sub(r'\^\{([^}]+)\}', r'\1', inner)
        # ^X 單字元上標（如 ^2 → 2）
        inner = re.sub(r'\^(\w)', r'\1', inner)
        # 常用符號替換
        inner = re.sub(r'\\cdot', '·', inner)
        inner = re.sub(r'\\pm', '±', inner)
        inner = re.sub(r'\\times', '×', inner)
        inner = re.sub(r'\\geq', '≥', inner)
        inner = re.sub(r'\\leq', '≤', inner)
        inner = re.sub(r'\\rightarrow', '→', inner)
        inner = re.sub(r'\\leftarrow', '←', inner)
        inner = re.sub(r'\\to\b', '→', inner)
        # 移除其他殘留的 LaTeX 指令（\something）
        inner = re.sub(r'\\[a-zA-Z]+', '', inner)
        # 移除殘留大括號
        inner = re.sub(r'[{}]', '', inner)
        return inner.strip()

    # 處理 $...$ 行內數學式
    text = re.sub(r'\$([^$]+)\$', _convert_math_block, text)
    # 移除孤立殘留的 $ 符號
    text = text.replace('$', '')
    return text


def _preprocess_for_nli(text: str, citation_sources=()) -> str:
    """
    送入 NLI 前移除格式噪音並轉換 LaTeX：
    - 引用標籤 （見 [事實N]）、（原文：「...」）會讓 NLI 誤判為 contradiction
    - Markdown bullet / 標題符號與 NLI 訓練資料格式不符
    - LaTeX 數學式轉為可讀科學表達式（保留化學式語義）
    原始句子仍保留在 results["sentence"] 供報告使用。
    回傳空字串表示「跳過此句」（caller 的 `if not hypothesis: continue` 負責跳過）。
    """
    # 元陳述句：描述「資訊缺失」或「無法確認」的句子本身不是事實命題，
    # 送入 NLI 必然產生偽陽性 contradiction，直接跳過。
    _META_SKIP = (
        r'缺失資訊',              # "缺失資訊：..." 或 "*   缺失資訊：..."
        r'\[資訊不足\]',          # "[資訊不足]..." 或 "*   [資訊不足]..."
        r'文獻依據不足，無法確認', # 清除 [資訊不足] 標籤後的殘餘文字
    )
    for pat in _META_SKIP:
        if re.search(pat, text):
            return ""

    # 移除引用標籤與狀態標記
    text = re.sub(r'（見 \[事實\d+\][^）]*）', '', text)
    text = re.sub(r'（原文：「[^」]*」）', '', text)
    text = re.sub(r'\[事實\d+\]', '', text)
    text = re.sub(r'\[待確認\]|\[資訊不足\]', '', text)
    # EN 模式：移除論文 citation ID，如 [1-s2.0-S2214714425005100-main]
    text = re.sub(r'\[\d+-s[\d.]+-[A-Za-z0-9\-_]+\]', '', text)
    sources = sorted(
        (re.escape(str(source)) for source in citation_sources if source),
        key=len,
        reverse=True,
    )
    if sources:
        source = "(?:" + "|".join(sources) + ")"
        text = re.sub(
            rf'\s*\[(?:Source:\s*)?{source}(?:\s*,\s*{source})*\](?=\s*[.!?]?\s*$)',
            '',
            text,
            flags=re.IGNORECASE,
        )
    # 移除 EN 狀態標記
    text = re.sub(r'\[Unverified\]|\[Insufficient Evidence\]|\[Fact \d+\]', '', text)
    # 移除 markdown 格式
    text = re.sub(r'^\s*(?:[-+*]|\d+\.)\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
    # LaTeX → 可讀科學表達式（保留化學式語義）
    text = _latex_to_plain(text)
    text = re.sub(
        r'^(?:Route|Review/comparison source|High-purity/isotopic enrichment|'
        r'Isotopic enrichment|Scalability|Cost-effectiveness|Safety)\s*:\s*',
        '',
        text,
        flags=re.IGNORECASE,
    )
    # 移除 LLM 結構化子標題前綴（如「試劑與比例：」「操作條件：」「後處理：」等）
    # 判斷標準：句子開頭到第一個全形/半形冒號之間 ≤12 字元，且前綴不含句子標點
    # 這些是 LLM 生成的格式標籤，raw PDF chunk 原文不含這類前綴
    # 例：「試劑與比例：使用 20 wt% 的明膠」→「使用 20 wt% 的明膠」
    # 保護：「本研究的合成過程分為三個階段：...」(14字元) 不會被誤刪
    text = re.sub(r'^[^，。！？,.\n]{1,12}[：:]\s*', '', text)
    # 移除行尾孤立冒號（如「試劑與用量：」這類標題行）
    text = re.sub(r'[：:]\s*$', '', text)
    # 整理空白
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'([.!?])\1+$', r'\1', text)
    # 清洗後太短（< 15 字元）→ 跳過，避免孤立數值/標籤送進 NLI 產生誤判
    if len(text) < 15:
        return ""
    return text


_LEXICAL_STOPWORDS = {
    "a", "an", "and", "any", "are", "as", "at", "be", "been", "being", "by",
    "for", "from", "in", "is", "it", "its", "of", "on", "or", "that", "the",
    "these", "this", "those", "to", "was", "were", "when", "where", "which", "while",
    "with",
}
_LEXICAL_NEGATIONS = {"no", "not", "never", "without"}


def _lexical_tokens(text: str) -> set[str]:
    raw = (text or "").lower()
    # PDF extraction often splits one word at a line break ("distil- lation").
    # Keep the original pieces and add the joined form for lexical matching.
    dehyphenated = re.sub(r"(?<=[a-z0-9])-\s+(?=[a-z0-9])", "", raw)
    tokens = set()
    for token in re.findall(r"[a-z0-9]+", f"{raw} {dehyphenated}"):
        if token in _LEXICAL_STOPWORDS:
            continue
        if len(token) > 4 and token.endswith("s") and not token.endswith(("ss", "is")):
            token = token[:-1]
        tokens.add(token)
    return tokens


def _find_lexical_support(
    hypothesis: str,
    chunks: list[dict],
    cited_sources: tuple[str, ...],
) -> tuple[str, float] | None:
    claim = hypothesis
    for source in cited_sources:
        claim = re.sub(re.escape(source), " ", claim, flags=re.IGNORECASE)
    claim_tokens = _lexical_tokens(claim)
    if len(claim_tokens) < 6:
        return None

    claim_numbers = {token for token in claim_tokens if any(char.isdigit() for char in token)}
    claim_negated = bool(claim_tokens & _LEXICAL_NEGATIONS)
    best = None
    for chunk in chunks:
        text = re.sub(r"\s+", " ", str(chunk.get("text", ""))).strip()
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s]
        windows = sentences + [
            f"{sentences[i]} {sentences[i + 1]}"
            for i in range(len(sentences) - 1)
        ]
        for source_window in windows:
            source_tokens = _lexical_tokens(source_window)
            if bool(source_tokens & _LEXICAL_NEGATIONS) != claim_negated:
                continue
            if not claim_numbers.issubset(source_tokens):
                continue
            coverage = len(claim_tokens & source_tokens) / len(claim_tokens)
            if coverage >= 0.8 and (best is None or coverage > best[1]):
                best = (str(chunk.get("id", chunk.get("source", ""))), coverage)
    return best


def _batch_translate_to_en(hypotheses: list[str]) -> list[str]:
    """
    將一批中文 hypothesis 一次性翻譯成英文（單次 LLM 呼叫）。
    只在 NLI_TRANSLATE_TO_EN=True 時使用。

    翻譯規則：
    - 數值、單位（wt%, °C, rpm, g, mL）原樣保留
    - 化學式（FeSO4, KBH4, NZVI, G-GEL, GEL）原樣保留
    - 若翻譯結果行數不符，回傳原始清單作為 fallback

    回傳與輸入等長的翻譯後清單。
    """
    import requests as _req, json as _json

    if not hypotheses:
        return hypotheses

    numbered = "\n".join(f"{i+1}. {h}" for i, h in enumerate(hypotheses))
    prompt = (
        "Translate the following sentences from Chinese to English.\n"
        "Rules:\n"
        "- Keep numbers, units (wt%, °C, rpm, g, mL, h), and chemical formulas "
        "(FeSO4, KBH4, NZVI, GEL, G-GEL, TC, DO, OH, etc.) unchanged.\n"
        "- Output ONLY the translations in the same numbered format (1. ... 2. ...).\n"
        "- No extra explanation or blank lines between items.\n\n"
        f"{numbered}"
    )

    try:
        _t = time.perf_counter()
        resp = _req.post(
            f"{cfg.OLLAMA_BASE_URL}/api/generate",
            json={
                "model":  cfg.SYNTHESIS_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 2048, "num_ctx": 4096},
            },
            timeout=120,
        )
        _add_llm_time(time.perf_counter() - _t)
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()

        # 解析 "N. text" 格式，提取翻譯結果
        translated = []
        for line in raw.split("\n"):
            line = line.strip()
            if not line:
                continue
            m = re.match(r'^\d+\.\s+(.+)', line)
            if m:
                translated.append(m.group(1).strip())

        if len(translated) == len(hypotheses):
            print(f"  [NLI-translate] 批次翻譯完成（{len(translated)} 句）")
            return translated
        else:
            print(f"  [NLI-translate] ⚠️ 翻譯行數不符（預期 {len(hypotheses)}，"
                  f"實際 {len(translated)}），使用原始中文 hypothesis")
            return hypotheses

    except Exception as e:
        print(f"  [NLI-translate] ⚠️ 翻譯失敗（{e}），fallback 至原始中文")
        return hypotheses


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

    若 NLI_TRANSLATE_TO_EN=True，所有中文 hypothesis 會在 NLI 前批次翻譯成英文，
    以消除 EN premise vs ZH hypothesis 的跨語言 entailment 落差。
    """
    if not sentences or not chunks:
        return []

    # ── 第一步：預處理所有句子，收集有效的 (sentence, hypothesis) 對 ──────
    valid_pairs: list[tuple[str, str, tuple[str, ...]]] = []
    skipped_sentences: set[int] = set()       # 原始 sentences 中被跳過的 index

    citation_sources = sorted({
        str(chunk.get("source", "")).strip()
        for chunk in chunks
        if chunk.get("source")
    })
    for idx, sentence in enumerate(sentences):
        hypothesis = _preprocess_for_nli(sentence, citation_sources)
        if not hypothesis:
            skipped_sentences.add(idx)
            continue
        cited_sources = _cited_sources_in_sentence(sentence, citation_sources)
        valid_pairs.append((sentence, hypothesis, cited_sources))

    if not valid_pairs:
        return []

    # ── 第二步：若啟用翻譯，批次將 hypothesis 翻譯為英文 ───────────────────
    # EN_DRAFT_PIPELINE=True 時 hypothesis 本身已是英文，跳過翻譯步驟
    if cfg.NLI_TRANSLATE_TO_EN and not cfg.EN_DRAFT_PIPELINE:
        raw_hypotheses = [h for _, h, _ in valid_pairs]
        translated     = _batch_translate_to_en(raw_hypotheses)
        valid_pairs    = [(s, t, sources) for (s, _, sources), t in zip(valid_pairs, translated)]

    # ── 第三步：逐句跑 NLI ────────────────────────────────────────────────
    results = []

    for sentence, hypothesis, cited_sources in valid_pairs:
        best_entail  = 0.0
        best_chunk_id = None
        best_entail_c = 0.0   # contradiction score of the best-entailment chunk
        support_method = "nli"
        scoped_chunks = chunks
        if getattr(cfg, "GROUNDING_CITATION_AWARE_ENABLED", False) and cited_sources:
            scoped_chunks = [c for c in chunks if str(c.get("source", "")) in cited_sources] or chunks

        lexical = None
        if (
            getattr(cfg, "GROUNDING_LEXICAL_SUPPORT_ENABLED", False)
            and cited_sources
        ):
            lexical = _find_lexical_support(hypothesis, scoped_chunks, cited_sources)
        if lexical:
            best_chunk_id, best_entail = lexical
            support_method = "lexical"
        else:
            # 滑動窗：chunk 常比 mDeBERTa 的 512-token 窗大，直接截斷會漏掉落在後段的事實
            # （實測事實常在第 1000~3400 字元，被舊的 [:512] 砍掉 → 誤判 unsupported）。
            # 把每個 chunk 切成重疊窗，全部一次批次 NLI，取最高 entailment。
            win_texts, win_chunk_ids = [], []
            for c in scoped_chunks:
                cid = c.get("id", c.get("source", ""))
                for w in _window_text(c["text"]):
                    win_texts.append(w)
                    win_chunk_ids.append(cid)
            try:
                scores_list = _run_nli_batch(win_texts, [hypothesis] * len(win_texts))
            except Exception as e:
                _report_nli_error(e)
                scores_list = []
            for cid, scores in zip(win_chunk_ids, scores_list):
                e_score = scores["entailment"]
                if e_score > best_entail:
                    best_entail   = e_score
                    best_entail_c = scores["contradiction"]
                    best_chunk_id = cid

        # ── 升級一：多來源聯合驗證（NLI_JOINT_VERIFY_ENABLED）────────────
        # 個別 chunk 都不夠支撐，但 top-3 合併後可以 → INFERENCE_BRIDGE
        is_bridge = False
        if best_entail < 0.5 and cfg.NLI_JOINT_VERIFY_ENABLED:
            joint = joint_verify(hypothesis, scoped_chunks)
            if joint["is_inference_bridge"]:
                is_bridge = True
                print(f"  [NLI-bridge] joint_score={joint['joint_score']:.3f} → INFERENCE_BRIDGE")

        # ── 升級二：子命題拆解驗證（NLI_DECOMPOSE_ENABLED）──────────────
        # 長句整句 NLI 不過，拆成子命題分別驗後全通過 → 視為 SUPPORTED
        if best_entail < 0.5 and not is_bridge and cfg.NLI_DECOMPOSE_ENABLED:
            decomp = decompose_and_verify(hypothesis, scoped_chunks)
            if decomp["chain_complete"] and decomp["sub_claims"]:
                best_entail = 0.7  # 合成分數，標記為有效支撐
                print(f"  [NLI-decomp] chain_complete → upgrade to SUPPORTED")

        # contradiction 只看 best-entailment chunk 自己的 c 分數：
        # 若最支持這句話的 chunk 本身也高度矛盾，才算真正的 CONFLICT；
        # 額外門檻：best_entail < 0.25 時 contradiction 分數是雜訊，不觸發 CONFLICT
        contradiction_detected = (
            cfg.NLI_CONTRADICTION_ENABLED
            and best_entail >= 0.25
            and best_entail_c > 0.7
        )

        if is_bridge:
            status = "INFERENCE_BRIDGE"
        elif best_entail >= 0.5:
            status = "CONFLICT" if contradiction_detected else "SUPPORTED"
        else:
            status = "CONFLICT" if contradiction_detected else "UNSUPPORTED"

        scope = ",".join(cited_sources) if cited_sources else "all"
        print(f"  [NLI-debug] e={best_entail:.3f} c={best_entail_c:.3f} "
              f"status={status} method={support_method} scope={scope} | {sentence[:60]}")
        # 顯示預處理後實際送進 NLI 的 hypothesis，方便對照確認前處理效果
        if hypothesis != sentence[:len(hypothesis)]:
            print(f"  [NLI-hypo]  → {hypothesis[:80]}")

        results.append({
            "sentence": sentence,
            "supported": best_entail >= 0.5 or is_bridge,
            "confidence": round(best_entail, 3),
            "best_chunk": best_chunk_id,
            "contradiction_detected": contradiction_detected,
            "contradiction_source": best_chunk_id if contradiction_detected else None,
            "status": status,
            "citation_sources": list(cited_sources),
            "support_method": support_method,
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
    supported = sum(1 for r in citation_results if r["supported"] or r.get("status") == "INFERENCE_BRIDGE")
    return round(supported / len(citation_results), 3)


def format_grounding_report(citation_results: list, section_scores: dict | None = None) -> str:
    """
    產生答案品質報告。
    section_scores 格式（可選）：
      {
        "direct":      {"score": float, "n_supported": int, "n_total": int},
        "inference":   {"score": float, "n_supported": int, "n_total": int},
        "speculation": {"score": float, "n_supported": int, "n_total": int},
      }
    有 section_scores 時顯示分段依據率；direct_score 作為品質門檻依據。
    沒有時退回整體依據率（向下相容）。
    """
    lines = ["\n\n---", "📋 **答案品質報告**\n"]

    _SECTION_LABELS = {
        "direct":      "【論文直接依據】",
        "inference":   "【跨文獻推論】",
        "speculation": "【知識延伸推測】",
    }
    _SECTION_NOTES = {
        "inference":   "  ← 跨論文推論，低分為預期範圍",
        "speculation": "  ← 知識延伸推測，低分為預期範圍",
    }

    def _score_emoji(score: float) -> str:
        if score >= 0.8: return "✅"
        if score >= 0.5: return "⚠️"
        return "❌"

    if section_scores:
        # ── 分段依據率顯示 ───────────────────────────────────
        direct_info   = section_scores.get("direct")
        primary_score = direct_info["score"] if direct_info else compute_grounding_score(citation_results)

        if primary_score >= 0.8:
            overall_label = "高（直接引用高度忠實於論文）"
        elif primary_score >= 0.5:
            overall_label = "中（部分直引陳述需確認）"
        else:
            overall_label = "低（建議縮小問題範圍）"

        lines.append("📊 **分段論文依據率：**\n")
        for key in ("direct", "inference", "speculation"):
            info = section_scores.get(key)
            if info is None:
                continue
            emoji = _score_emoji(info["score"])
            label = _SECTION_LABELS.get(key, key)
            note  = _SECTION_NOTES.get(key, "")
            lines.append(
                f"  {emoji} {label}：{info['score']:.1%}"
                f"（{info['n_supported']}/{info['n_total']} 句）{note}"
            )
        lines.append(f"\n{_score_emoji(primary_score)} **直引依據率**：{primary_score:.1%}　{overall_label}\n")
    else:
        # ── fallback：無 section 資訊，顯示整體依據率 ────────
        grounding_score = compute_grounding_score(citation_results)
        primary_score   = grounding_score
        if grounding_score >= 0.8:
            emoji, label = "✅", "高（答案高度忠實於論文內容）"
        elif grounding_score >= 0.5:
            emoji, label = "⚠️", "中（部分陳述需要確認）"
        else:
            emoji, label = "❌", "低（建議重新查詢或縮小問題範圍）"
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

    unsupported = [r for r in citation_results if not r["supported"] and r.get("status") not in ("CONFLICT", "INFERENCE_BRIDGE")]
    if not unsupported and not [r for r in citation_results if r.get("status") == "CONFLICT"]:
        lines.append("✅ **所有陳述均有論文依據**\n")
    elif unsupported:
        lines.append(
            f"⚠️  **以下 {len(unsupported)} 個陳述未找到明確論文依據，請謹慎參考：**\n"
        )
        for r in unsupported:
            chunk_info = f"，最近似來源：{r['best_chunk']}" if r.get("best_chunk") else ""
            lines.append(f"- {r['sentence']}（信心度：{r['confidence']:.1%}{chunk_info}）")

    # ★ 附上機器可解析的分數，供 api.py 的品質門檻使用（用直引分數，語義最準確）
    lines.append(f"\n<!-- grounding_score={primary_score:.3f} -->")
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
#  生成自我修正：對 NLI 標記句一次 batched gemma4 裁定（GENERATION_SELFCORRECT_ENABLED）
# ══════════════════════════════════════════════════════════════════

def selfcorrect_flagged(flagged: list[dict], chunks: list[dict], max_items: int = 14) -> list[dict]:
    """
    對 NLI 標記為「不支持」的句子做「一次」batched gemma4 裁定。
    回傳與 flagged 等長對齊的 [{"verdict": "SUPPORTED|CORRECT|UNVERIFIED", "fixed": str}]。
    - SUPPORTED：source 其實支持（NLI 假陰性）→ 保留不動。
    - CORRECT  ：source 矛盾/部分支持 → fixed 給對著 source 改寫後的句子。
    - UNVERIFIED：source 根本沒這資訊。
    呼叫/解析失敗 → 全部回 SUPPORTED（保守：不動答案）。單趟、不 retry。
    """
    import requests as _req, json as _json, re as _re
    if not flagged:
        return []
    flagged = flagged[:max_items]

    cmap = {}
    for c in chunks:
        cmap[c.get("id", c.get("source", ""))] = c.get("text", "")

    # 去重 source（多句常共用同一 chunk），各給字母代號
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    srcs, src_blocks = {}, []
    for r in flagged:
        cid = r.get("best_chunk")
        if cid and cid not in srcs and cid in cmap:
            srcs[cid] = letters[len(srcs)] if len(srcs) < len(letters) else "?"
            src_blocks.append(f"[{srcs[cid]}] {cmap[cid][:2400]}")

    items = [f"[{i}] (source {srcs.get(r.get('best_chunk'), '?')}) {r['sentence']}"
             for i, r in enumerate(flagged, 1)]

    prompt = (
        "An automated NLI check flagged the following answer CLAIMS as possibly NOT supported "
        "by their SOURCE excerpt. The check has MANY FALSE POSITIVES (it often flags claims that "
        "ARE supported but phrased differently, or that state only part of a list). "
        "Your job is mainly to RESCUE those false positives, not to rewrite.\n"
        "STRONGLY PREFER 'SUPPORTED'. For each claim:\n"
        "- SUPPORTED (default): the source supports the claim, even if worded differently, even if "
        "the claim states only one item of a list or omits sub-details. Keep unchanged.\n"
        "- CORRECT: ONLY if the source DIRECTLY CONTRADICTS the claim (e.g. a different number, the "
        "opposite fact). Then rewrite MINIMALLY — fix only the contradicted part, keep everything else.\n"
        "- UNVERIFIED: ONLY if the source clearly contains nothing about this claim.\n"
        "When in doubt, choose SUPPORTED.\n\n"
        'Output ONLY a JSON array, one object per claim IN ORDER:\n'
        '[{"i":1,"verdict":"SUPPORTED","fixed":""}, ...]\n'
        "(fixed = minimally rewritten claim ONLY when verdict is CORRECT, else empty string)\n\n"
        "SOURCES:\n" + "\n\n".join(src_blocks) + "\n\nCLAIMS:\n" + "\n".join(items)
    )

    default = [{"verdict": "SUPPORTED", "fixed": ""} for _ in flagged]
    try:
        _t = time.perf_counter()
        resp = _req.post(
            f"{cfg.OLLAMA_BASE_URL}/api/generate",
            json={"model": cfg.SYNTHESIS_MODEL, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.1, "num_predict": 2048, "num_ctx": 16384}},
            timeout=600,
        )
        _add_llm_time(time.perf_counter() - _t)
        resp.raise_for_status()
        raw = _re.sub(r'```json|```', '', resp.json().get("response", "[]")).strip()
        m = _re.search(r'\[.*\]', raw, _re.S)
        parsed = _json.loads(m.group(0) if m else raw)
    except Exception:
        return default

    out = []
    for i in range(len(flagged)):
        v = parsed[i] if i < len(parsed) and isinstance(parsed[i], dict) else {}
        verdict = str(v.get("verdict", "SUPPORTED")).upper()
        if verdict not in ("SUPPORTED", "CORRECT", "UNVERIFIED"):
            verdict = "SUPPORTED"
        out.append({"verdict": verdict, "fixed": (v.get("fixed") or "").strip()})
    return out


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
    if cfg.EN_DRAFT_PIPELINE:
        prompt = (
            f"Decompose the following conclusion into 2-4 independent sub-claims, "
            f"each of which can be independently supported or refuted by literature.\n"
            f"Output ONLY a JSON array: [\"sub-claim 1\", \"sub-claim 2\", ...]\n\n"
            f"Conclusion: {conclusion}"
        )
    else:
        prompt = (
            f"請將以下結論句子拆解成 2-4 個獨立的子命題，每個子命題應能獨立被文獻支撐或反駁。\n"
            f"只輸出 JSON 陣列，格式：[\"子命題1\", \"子命題2\", ...]\n\n"
            f"結論：{conclusion}"
        )
    # ponytail: 叫 gemma4 前先把 NLI 累積的 torch reserved 快取(~2-3GB)還給驅動，
    # 否則 gemma4:31b(~18-20GB) + 這塊快取會頂破 24GB → gemma4 部分 offload 到 RAM 變慢。
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    try:
        _t = time.perf_counter()
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
        _add_llm_time(time.perf_counter() - _t)
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
    fact_texts = [f["text"][:512] for f in facts]
    for claim in sub_claims_text:
        best_score = 0.0
        best_source = None
        try:
            scores_list = _run_nli_batch(fact_texts, [claim] * len(fact_texts))
        except Exception as e:
            _report_nli_error(e)
            scores_list = []
        for fact, scores in zip(facts, scores_list):
            if scores["entailment"] > best_score:
                best_score = scores["entailment"]
                best_source = fact.get("id", fact.get("source", ""))

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
    fact_texts = [fact["text"][:512] for fact in facts]
    try:
        scores_list = _run_nli_batch(fact_texts, [claim] * len(fact_texts))
    except Exception as e:
        _report_nli_error(e)
        scores_list = []
    scored_facts = []
    for fact, scores in zip(facts, scores_list):
        scored_facts.append({
            "source": fact.get("id", fact.get("source", "")),
            "text": fact["text"][:512],
            "score": scores["entailment"],
        })

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
