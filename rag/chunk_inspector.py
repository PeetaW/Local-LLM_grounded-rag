# rag/chunk_inspector.py
# Chunk 品質檢查工具
# 執行方式：python main.py --test-chunks [--paper 檔名] [--n 數量]

from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter

import config as cfg
from rag.pdf_loader import load_pdf_with_pymupdf


def inspect_chunks(pdf_path: str, num_chunks: int = 10):
    """
    載入單篇 PDF，切成 chunks 後逐一印出，供肉眼檢查。
    重點確認：
      1. 句子/段落有沒有被截斷到一半
      2. 化學式、數字、單位有沒有跟說明分開
      3. 表格內容有沒有被切壞
      4. chunk 之間的 overlap 是否有正確重疊
    """
    print(f"\n{'='*65}")
    print(f"  Chunk 品質檢查")
    print(f"  PDF：{pdf_path}")
    print(f"  chunk_size={Settings.chunk_size}, chunk_overlap={Settings.chunk_overlap}")
    print(f"{'='*65}\n")

    docs = load_pdf_with_pymupdf(pdf_path)
    text_docs = [d for d in docs if d.metadata.get("source_type") == "pdf_text"]

    if not text_docs:
        print("⚠️  找不到 PDF 文字內容，請確認路徑正確")
        return

    parser = SentenceSplitter(
        chunk_size=Settings.chunk_size,
        chunk_overlap=Settings.chunk_overlap,
    )
    nodes = parser.get_nodes_from_documents(text_docs)
    total = len(nodes)

    print(f"  → 共切出 {total} 個 chunks，顯示前 {min(num_chunks, total)} 個\n")

    for i, node in enumerate(nodes[:num_chunks]):
        text = node.text
        char_count = len(text)
        token_estimate = char_count // 3

        print(f"┌─ Chunk {i+1:02d}/{total} {'─'*45}")
        print(f"│  字元數：{char_count:,}　　估計 token：{token_estimate:,}")
        print(f"│  {'─'*52}")

        head = text[:200].replace("\n", "↵")
        print(f"│  【開頭】{head}")

        if char_count > 400:
            tail = text[-200:].replace("\n", "↵")
            print(f"│  【結尾】...{tail}")

        last_line = text.strip().split("\n")[-1]
        incomplete_end_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        if last_line and last_line[-1] in incomplete_end_chars:
            print(f"│  ⚠️  【結尾疑似截斷】最後字元：'{last_line[-1]}'")

        print(f"└{'─'*54}\n")

    if total >= 2:
        print(f"\n{'='*65}")
        print(f"  Overlap 檢查（Chunk 1 結尾 vs Chunk 2 開頭）")
        print(f"{'='*65}")
        tail_1 = nodes[0].text[-Settings.chunk_overlap:]
        head_2 = nodes[1].text[:Settings.chunk_overlap]
        print(f"\n【Chunk 1 結尾 {Settings.chunk_overlap} 字元】")
        print(tail_1.replace("\n", "↵"))
        print(f"\n【Chunk 2 開頭 {Settings.chunk_overlap} 字元】")
        print(head_2.replace("\n", "↵"))

        overlap_found = 0
        for size in range(min(len(tail_1), len(head_2)), 0, -1):
            if tail_1[-size:] == head_2[:size]:
                overlap_found = size
                break
        if overlap_found > 0:
            print(f"\n  ✅ 實際重疊：{overlap_found} 字元")
        else:
            print(f"\n  ℹ️  未偵測到精確重疊（SentenceSplitter 以句子為單位切割，屬正常現象）")

    print(f"\n{'='*65}")
    print(f"  檢查完成！請人工確認上方 chunks 內容是否合理")
    print(f"  若有大量截斷或亂碼，考慮調整 chunk_size / chunk_overlap")
    print(f"{'='*65}\n")