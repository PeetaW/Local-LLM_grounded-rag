# rag/query_translation.py
# Pipeline Stage 7: translate the verified English draft to Traditional Chinese.
# Only used when EN_DRAFT_PIPELINE is enabled.

import re

import config as cfg


_OCR_NORMALITY_RE = re.compile(
    r"\b(?:I|l)\s+N(?=\s+(?:hydrochloric|sulfuric|nitric)\s+acid\b|\s+(?:HCl|NaOH|KOH)\b)"
)
_BYTE_FALLBACK_SEQUENCE_RE = re.compile(r"(?:<0[xX][0-9A-Fa-f]{2}>)+")
_BYTE_FALLBACK_TOKEN_RE = re.compile(r"<0[xX]([0-9A-Fa-f]{2})>")
_REDUNDANT_UNDETECTABLE_RE = re.compile(
    r"未(?P<verb>檢測|偵測)到可(?:檢測|偵測)的"
)


def _normalize_ocr_measurements(text: str) -> str:
    return _OCR_NORMALITY_RE.sub("1 N", text or "")


def _decode_utf8_byte_fallbacks(text: str) -> str:
    def decode(match: re.Match) -> str:
        raw = bytes(
            int(value, 16)
            for value in _BYTE_FALLBACK_TOKEN_RE.findall(match.group(0))
        )
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return match.group(0)

    return _BYTE_FALLBACK_SEQUENCE_RE.sub(decode, text or "")


def _normalize_translation_semantics(text: str) -> str:
    return _REDUNDANT_UNDETECTABLE_RE.sub(
        lambda match: f"未{match.group('verb')}到",
        text or "",
    )


def _term_fidelity_rules() -> str:
    if not getattr(cfg, "TERM_FIDELITY_GUARD_ENABLED", False):
        return ""
    return (
        "- Preserve exact English spellings for enzymes, reagents, compounds, method names, model names, and abbreviations; "
        "include the English term in parentheses if you translate the surrounding phrase.\n"
        "- Do not substitute near-synonyms for technical names. For example, chymotrypsin and trypsin are different enzymes; "
        "keep whichever term appears in the source answer.\n"
        "- Preserve route-defining phrases verbatim in English when present, especially "
        "\"chymotrypsin-catalysed enzymatic hydrolysis\"; add Chinese explanation after it if helpful.\n"
        "- For chemical or peptide conjugation, translate \"conjugated to\" as \"偶聯至\" or \"與...偶聯\", "
        "never the generic \"結合\"; preserve that the components are chemically linked.\n"
    )


def _fold_change_rules() -> str:
    if not getattr(cfg, "TRANSLATION_FOLD_CHANGE_GUARD_ENABLED", False):
        return ""
    return (
        "- Preserve the mathematical meaning of fold changes. Translate an N-fold decrease or reduction "
        "as 'decreased to approximately 1/N of the original' (for example, 'a three-fold decrease' → "
        "'降至原來約三分之一'), never as the ambiguous '降低 N 倍'. Translate an N-fold increase as "
        "'increased to N times the original'.\n"
    )


def translate_to_traditional_chinese(text: str, on_status=None) -> str:
    """
    Translate an academic answer from English to Traditional Chinese.
    Section headers are mapped to their Chinese equivalents.
    Returns the translated text, or the original on failure.
    """
    import requests as _req

    def _status(msg):
        if on_status:
            on_status(msg)
        else:
            print(msg)

    text = _normalize_ocr_measurements(text)
    _status("\n  🌏 翻譯英文答案為繁體中文...")
    prompt = (
        "Translate the following academic answer from English to Traditional Chinese (繁體中文).\n"
        "Rules:\n"
        "- Section headers must be translated as follows:\n"
        "  '## [Direct Paper Evidence]' → '## 【論文直接依據】'\n"
        "  '## [Cross-Literature Inference]' → '## 【跨文獻推論】'\n"
        "  '## [Knowledge Extension and Speculation]' → '## 【知識延伸與推測】'\n"
        "- Paper name labels [Paper Name] → 【Paper Name】 (keep the name itself unchanged)\n"
        "- Preserve all numbers, units (wt%, °C, rpm, g, mL, h), and chemical formulas exactly; "
        "for fold changes, preserve the quantitative meaning according to the rule below.\n"
        "- Keep label tags unchanged: [Fact N], [Insufficient Evidence], [Unverified], VERIFY_PASS, VERIFY_FAIL.\n"
        f"{_term_fidelity_rules()}"
        f"{_fold_change_rules()}"
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
                # ponytail: 翻譯輸入只有單篇答案，不需 64k KV cache。降到 16384
                # 省 VRAM、加速 token，又留足邊際避免長答案(輸入+中文輸出)被截斷掉字。
                "options": {"temperature": 0.1, "num_predict": -1, "num_ctx": 16384},
            },
            timeout=cfg.LLM_TIMEOUT,
        )
        if resp.ok:
            translated = _normalize_translation_semantics(
                _decode_utf8_byte_fallbacks(resp.json().get("response", ""))
            ).strip()
            if translated:
                _status(f"  ✅ 翻譯完成（{len(translated):,} 字元）")
                return translated
    except Exception as e:
        _status(f"  ⚠️  翻譯失敗，保留英文版本：{e}")
    return text
