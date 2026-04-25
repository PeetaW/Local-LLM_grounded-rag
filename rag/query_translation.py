# rag/query_translation.py
# Pipeline Stage 7: translate the verified English draft to Traditional Chinese.
# Only used when EN_DRAFT_PIPELINE is enabled.

import config as cfg


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
