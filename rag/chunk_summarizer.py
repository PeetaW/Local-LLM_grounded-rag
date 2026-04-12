# rag/chunk_summarizer.py
# 負責在建索引時對每個 chunk 生成摘要標頭
# 使用 deepseek-r1:32b 確保學術理解深度

import os
import json
import hashlib
import config as cfg


def get_summary_cache_path(paper_name: str) -> str:
    """每篇論文的摘要快取路徑"""
    cache_dir = os.path.join(cfg.INDEX_BASE_DIR, paper_name)
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, "chunk_summaries.json")


def load_summary_cache(paper_name: str) -> dict:
    """載入既有的摘要快取"""
    cache_path = get_summary_cache_path(paper_name)
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_summary_cache(paper_name: str, cache: dict):
    """儲存摘要快取"""
    cache_path = get_summary_cache_path(paper_name)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def chunk_hash(text: str) -> str:
    """用 hash 作為 chunk 的唯一 ID，避免重複生成"""
    return hashlib.md5(text.encode()).hexdigest()[:12]


def summarize_chunk(chunk_text: str) -> str:
    """
    用 deepseek-r1:32b 對單個 chunk 生成一行摘要標頭。
    摘要應該包含：這段在說什麼、關鍵數值、科學意義。
    """
    from llama_index.core import Settings

    prompt = (
        "以下是一段學術論文的內容：\n\n"
        f"{chunk_text[:1500]}\n\n"
        "請用一句話（繁體中文，30字以內）概括這段內容的核心資訊。\n"
        "重點保留：具體數值、化學物質名稱、實驗條件、關鍵發現。\n"
        "只輸出摘要句子，不要其他文字。"
    )

    try:
        response = Settings.llm.complete(prompt)
        # 清除 deepseek-r1 的 <think> token
        import re
        summary = re.sub(
            r'<think>.*?</think>', '', response.text, flags=re.DOTALL
        ).strip()
        return summary
    except Exception as e:
        print(f"  ⚠️  摘要生成失敗：{e}")
        return ""


def add_summaries_to_nodes(nodes: list, paper_name: str) -> list:
    """
    對一篇論文的所有 chunks 生成摘要，
    把摘要加進每個 node 的 metadata 和文字開頭。
    有快取則跳過，節省時間。
    回傳加上摘要的 nodes。
    """
    if not cfg.CONTEXT_SUMMARY_ENABLED:
        return nodes

    cache = load_summary_cache(paper_name)
    updated_nodes = []
    new_summaries = 0

    print(f"  📝 生成 chunk 摘要（共 {len(nodes)} 個）...")

    for i, node in enumerate(nodes):
        text = node.text
        h = chunk_hash(text)

        if h in cache:
            # 快取命中，直接用
            summary = cache[h]
        else:
            # 需要生成新摘要
            print(f"  ✍️  [{i+1}/{len(nodes)}] 生成摘要中...")
            summary = summarize_chunk(text)
            cache[h] = summary
            new_summaries += 1

        # 把摘要加進 node
        if summary:
            node.metadata["chunk_summary"] = summary
            # 在文字開頭加上摘要標頭，讓 embedding 和 LLM 都能看到
            node.text = f"[摘要：{summary}]\n\n{text}"

        updated_nodes.append(node)

    # 儲存快取
    if new_summaries > 0:
        save_summary_cache(paper_name, cache)
        print(f"  ✅ 新生成 {new_summaries} 個摘要，已快取")
    else:
        print(f"  ✅ 全部使用快取摘要")

    return updated_nodes