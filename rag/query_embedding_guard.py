# rag/query_embedding_guard.py
# Embedding safety layer: text cleaning, NaN detection, retry/truncation logic.
# Thread-safe (pure HTTP + stateless transforms). No LLM calls.

import re
import unicodedata

import config as cfg


def _clean_for_embed(text: str) -> str:
    """
    Remove characters that cause bge-m3 to produce NaN embeddings.
    Preserves as much semantic content as possible.
    """
    removed_ctrl = [
        f"U+{ord(c):04X}({unicodedata.name(c, '?')})"
        for c in text if unicodedata.category(c).startswith('C')
    ]
    if removed_ctrl:
        print(f"  🔬 [embed-debug] 移除控制字元：{removed_ctrl[:5]}")

    text = ''.join(c for c in text if not unicodedata.category(c).startswith('C'))
    text = text.replace('（', '(').replace('）', ')')

    angle_matches = re.findall(r'[<>]\s*\d+\s*\w+', text)
    if angle_matches:
        print(f"  🔬 [embed-debug] 角括號數值（替換）：{angle_matches}")
    text = re.sub(r'<\s*(\d+\s*\w+)', r'less than \1', text)
    text = re.sub(r'>\s*(\d+\s*\w+)', r'greater than \1', text)

    long_parens = re.findall(r'\([^)]{25,}\)', text)
    if long_parens:
        print(f"  🔬 [embed-debug] 移除長括號內容：{[p[:30] for p in long_parens]}")
    text = re.sub(r'\([^)]{25,}\)', '', text)

    return text.strip()


def _test_embed(text: str, label: str = "") -> str:
    """
    Run a single embedding preflight check.
    Returns 'ok' / 'nan' / 'timeout' / 'error'.
    Thread-safe: read-only HTTP, no shared state.
    """
    import requests
    try:
        r = requests.post(
            f"{cfg.OLLAMA_BASE_URL}/api/embeddings",
            json={"model": cfg.EMBED_MODEL, "prompt": text},
            timeout=120,
        )
        if r.status_code != 200:
            # HTTP 500 + NaN message means bge-m3 produced NaN; Ollama cannot serialize it.
            # This is a text content issue — return "nan" to trigger cleaning/truncation.
            if r.status_code == 500 and "NaN" in r.text:
                print(f"  🔬 [embed-debug] HTTP 500 NaN（bge-m3 輸出 NaN），text={text[:60]!r}")
                return "nan"
            print(f"  🔬 [embed-debug] HTTP {r.status_code}，text={text[:60]!r}")
            return "error"
        embedding = r.json().get("embedding", [])
        if not embedding:
            print(f"  🔬 [embed-debug] embedding 為空，text={text[:60]!r}")
            return "error"
        nan_count = sum(1 for x in embedding if x != x)
        if nan_count:
            print(f"  🔬 [embed-debug] NaN {nan_count}/{len(embedding)} 維，"
                  f"label={label!r}，text={text[:80]!r}")
            return "nan"
        return "ok"
    except requests.exceptions.Timeout:
        print(f"  🔬 [embed-debug] Timeout（Ollama 忙碌），text={text[:60]!r}")
        return "timeout"
    except Exception as e:
        print(f"  🔬 [embed-debug] 例外：{e}，text={text[:60]!r}")
        return "error"


def _embed_with_retry(text: str, label: str = "", max_retries: int = 5) -> bool:
    """
    Retry on timeout; return False on NaN so caller can truncate.
    Thread-safe.
    """
    import time
    for i in range(max_retries):
        result = _test_embed(text, label=label)
        if result == "ok":
            return True
        if result == "nan":
            return False
        wait = 15 * (i + 1)
        print(f"  ⏳ [embed] Ollama 忙碌，{wait}s 後重試（第{i+1}/{max_retries}次）...")
        time.sleep(wait)
    return False


def prepare_query_text(query_str: str) -> str:
    """
    Phase A-1: clean text + embedding preflight.
    Returns a query string safe to pass to the retriever.
    Uses only bge-m3; no LLM call. Thread-safe.
    """
    cleaned = _clean_for_embed(query_str)
    if cleaned != query_str:
        print(f"  🧹 偵測到特殊字元，清洗後重試...")
        if _embed_with_retry(cleaned, label="cleaned"):
            return cleaned

    if _embed_with_retry(query_str, label="original"):
        return query_str

    # Last resort: progressively truncate
    current = cleaned if cleaned else query_str
    for attempt in range(3):
        cut = int(len(current) * 2 / 3)
        current = current[:cut].strip()
        print(f"  ⚠️  embedding NaN，截短至 {len(current)} 字元後重試（第{attempt+1}次）")
        print(f"  🔬 [embed-debug] 截短後內容：{current!r}")
        if _embed_with_retry(current, label=f"truncated-{attempt+1}"):
            return current

    print(f"  ❌ embedding 反覆失敗，使用截短版本強制查詢")
    return current
