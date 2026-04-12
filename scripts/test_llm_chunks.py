"""
test_llm_chunks.py
==================
目的：
    確認 reranker 拿到的 4 個 chunks 直接送給 LLM 後，
    LLM 能不能正確回答合成步驟問題。

    同時測試兩種送法：
    A) 模擬主腳本的方式：用 RetrieverQueryEngine（response_mode="compact"）
    B) 直接把 4 個 chunks 拼成字串送進 prompt（繞過 LlamaIndex 的處理）

    比較 A vs B 的結果，確認是 LlamaIndex 的 compact 壓縮造成問題，
    還是 LLM 本身就看不懂這些 chunks。

使用方式：
    conda activate llm_env
    python test_llm_chunks.py
"""

import os
import sys
import time
import httpx

from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.core.retrievers import VectorIndexRetriever, QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import QueryBundle

# ── 設定（與主腳本完全相同）─────────────────────────────
http_client = httpx.Client(timeout=httpx.Timeout(28800.0, connect=30.0))

Settings.llm = OpenAILike(
    model="deepseek-r1:32b",
    api_base="http://localhost:11434/v1",
    api_key="ollama",
    is_chat_model=True,
    timeout=28800.0,
    http_client=http_client,
    context_window=16384,
    system_prompt=(
        "你是一個學術論文分析助手。"
        "請只根據提供的論文內容回答問題，使用繁體中文。"
        "請務必使用繁體中文，絕對不可使用簡體中文。"
        "如果論文中沒有相關資訊，請直接說明「此論文未涉及此議題」，"
        "不要自行推測或補充論文以外的內容。"
        "回答時請盡量引用論文中的具體數據、步驟與條件。"
    ),
)

Settings.embed_model = OllamaEmbedding(
    model_name="bge-m3",
    base_url="http://localhost:11434",
)

Settings.chunk_size = 1024
Settings.chunk_overlap = 256

# ── 載入索引 ─────────────────────────────────────────────
INDEX_BASE_DIR = "index_storage"
target_paper = "1-s2.0-S1878029613002417-main"
index_dir = os.path.join(INDEX_BASE_DIR, target_paper)

print(f"\n{'='*65}")
print(f"  LLM + Chunks 直接診斷測試")
print(f"  論文：{target_paper}")
print(f"{'='*65}\n")

print("📂 載入索引中...")
storage_context = StorageContext.from_defaults(persist_dir=index_dir)
index = load_index_from_storage(storage_context)
all_nodes = list(index.docstore.docs.values())
print(f"  → 共 {len(all_nodes)} 個 chunks\n")

# ── 查詢問題 ─────────────────────────────────────────────
query_text = (
    "What is the synthesis procedure for NZVI? "
    "What amounts of FeSO4 and NaBH4 were used? "
    "What were the stirring speed and experimental conditions?"
)

# ── 取得 reranker 後的 8 個 chunks（重現主腳本的 retrieval）
print("🔍 執行 retrieval pipeline，取得 reranker 後的 8 個 chunks...")

vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=8)
bm25_retriever = BM25Retriever.from_defaults(nodes=all_nodes, similarity_top_k=8)
hybrid_retriever = QueryFusionRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    similarity_top_k=8,
    num_queries=1,
    mode="reciprocal_rerank",
    use_async=False,
)
reranker = SentenceTransformerRerank(
    model="BAAI/bge-reranker-v2-m3",
    top_n=8,
)

hybrid_results = hybrid_retriever.retrieve(query_text)
query_bundle = QueryBundle(query_str=query_text)
final_chunks = reranker.postprocess_nodes(hybrid_results, query_bundle)

print(f"  → 取得 {len(final_chunks)} 個 chunks\n")

# 印出 chunks 摘要
print("📄 Reranker 後的 8 個 chunks 內容摘要：")
print("─" * 65)
for i, node in enumerate(final_chunks, 1):
    text = node.get_content()
    score = node.score
    print(f"\n  Chunk [{i}] reranker_score={score:.4f}，字元數={len(text)}")
    print(f"  前200字：{text[:200].strip()!r}")
    print(f"  後100字：...{text[-100:].strip()!r}")
print()

# ════════════════════════════════════════════════════════
# 測試 A：模擬主腳本的 RetrieverQueryEngine（refine 模式）
# ════════════════════════════════════════════════════════
print(f"{'='*65}")
print(f"【測試 A】RetrieverQueryEngine（response_mode='refine'）")
print(f"  這是主腳本目前的做法")
print(f"{'='*65}\n")

engine = RetrieverQueryEngine.from_args(
    retriever=hybrid_retriever,
    response_mode="refine",
    node_postprocessors=[reranker],
)

# 用中文問（模擬主腳本的子問題）
chinese_query = (
    "What is the synthesis procedure for NZVI mediated by L-amino acid? "
    "List all reagents and their exact amounts in grams and mL. "
    "Include stirring speed in rpm and other experimental conditions."
)

print(f"查詢：{chinese_query}\n")
print("🤖 LLM 回答（refine 模式）：\n")

start = time.time()
response_a = engine.query(chinese_query)
elapsed_a = time.time() - start

print(str(response_a))
print(f"\n⏱ 耗時：{int(elapsed_a//60)}分{int(elapsed_a%60)}秒")

# ════════════════════════════════════════════════════════
# 測試 B：直接把 8 個 chunks 拼成字串送進 prompt
# ════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"【測試 B】直接拼接 chunks → prompt（繞過 LlamaIndex 壓縮）")
print(f"  這是確認 compact 模式是否造成問題的對照組")
print(f"{'='*65}\n")

# 把 8 個 chunks 完整拼接
chunks_text = ""
for i, node in enumerate(final_chunks, 1):
    text = node.get_content()
    chunks_text += f"\n--- Chunk {i} ---\n{text}\n"

direct_prompt = f"""以下是從論文中檢索到的相關段落：

{chunks_text}

---
問題：請根據以上段落，詳細回答：
1. ZVI（零價鐵奈米粒子）的完整合成步驟是什麼？
2. 使用了哪些試劑？每種試劑的用量是多少？（請列出具體數字：克、mL、mM）
3. 操作條件是什麼？（攪拌速度 rpm、溫度、時間）
請直接引用段落中的具體數字，不要推測或補充。
"""

total_chars = len(direct_prompt)
print(f"Prompt 總字元數：{total_chars}（估計 {total_chars//4} tokens）\n")
print("🤖 LLM 回答（直接 prompt）：\n")

start = time.time()
response_b = ""
for chunk in Settings.llm.stream_complete(direct_prompt):
    print(chunk.delta, end="", flush=True)
    response_b += chunk.delta
elapsed_b = time.time() - start

print(f"\n\n⏱ 耗時：{int(elapsed_b//60)}分{int(elapsed_b%60)}秒")

# ════════════════════════════════════════════════════════
# 最終比較
# ════════════════════════════════════════════════════════
TARGET_KEYWORDS = ["0.27785", "0.46", "500 rpm", "900 rpm", "20 mM", "NaBH", "FeSO4", "ice"]

def check_keywords(text: str) -> tuple:
    found = [kw for kw in TARGET_KEYWORDS if kw.lower() in text.lower()]
    missing = [kw for kw in TARGET_KEYWORDS if kw.lower() not in text.lower()]
    return found, missing

found_a, missing_a = check_keywords(str(response_a))
found_b, missing_b = check_keywords(response_b)

print(f"\n{'='*65}")
print(f"  最終比較報告")
print(f"{'='*65}")
print(f"\n  【測試 A】refine 模式")
print(f"  命中關鍵字（{len(found_a)}/{len(TARGET_KEYWORDS)}）：{', '.join(found_a) if found_a else '無'}")
print(f"  遺漏關鍵字：{', '.join(missing_a) if missing_a else '無'}")

print(f"\n  【測試 B】直接 prompt")
print(f"  命中關鍵字（{len(found_b)}/{len(TARGET_KEYWORDS)}）：{', '.join(found_b) if found_b else '無'}")
print(f"  遺漏關鍵字：{', '.join(missing_b) if missing_b else '無'}")

print(f"\n  → 結論：", end="")
if len(found_b) > len(found_a):
    print("refine 模式有問題，直接 prompt 更好")
    print("         建議把主腳本的 response_mode 從 refine 改為直接 prompt 方式")
elif len(found_a) >= len(found_b):
    print("兩者差不多，問題不在 refine或直接prompt 模式")
    print("         需要進一步排查子問題的查詢字串本身")
print(f"{'='*65}\n")
