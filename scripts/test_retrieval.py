"""
test_retrieval.py
=================
目的：
    診斷 retrieval 三層（BM25 / 向量搜尋 / reranker）各自拿到哪些 chunks，
    找出合成步驟 chunk 在哪一層被漏掉。

使用方式：
    conda activate llm_env
    python test_retrieval.py

可選參數：
    --paper   指定論文名稱（預設 1-s2.0-S1878029613002417-main）
    --query   指定查詢問題（預設：合成步驟相關問題）

    python test_retrieval.py --paper 1-s2.0-S1878029613002417-main --query "synthesis procedure amounts"
"""

import os
import sys
import httpx

from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.core.retrievers import VectorIndexRetriever, QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.postprocessor import SentenceTransformerRerank
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
)

Settings.embed_model = OllamaEmbedding(
    model_name="bge-m3",
    base_url="http://localhost:11434",
)

Settings.chunk_size = 1024
Settings.chunk_overlap = 256

# ── 參數解析 ─────────────────────────────────────────────
INDEX_BASE_DIR = "index_storage"
target_paper = "1-s2.0-S1878029613002417-main"
query_text = "What is the synthesis procedure for NZVI? What amounts of FeSO4 and NaBH4 were used? What were the stirring speed and experimental conditions?"

if "--paper" in sys.argv:
    idx = sys.argv.index("--paper")
    if idx + 1 < len(sys.argv):
        target_paper = sys.argv[idx + 1]

if "--query" in sys.argv:
    idx = sys.argv.index("--query")
    if idx + 1 < len(sys.argv):
        query_text = sys.argv[idx + 1]

index_dir = os.path.join(INDEX_BASE_DIR, target_paper)
if not os.path.exists(index_dir):
    print(f"❌ 找不到索引：{index_dir}")
    print(f"   請先用主腳本建立索引後再執行此測試")
    sys.exit(1)

# ── 載入索引 ─────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  Retrieval 診斷測試")
print(f"  論文：{target_paper}")
print(f"{'='*65}")
print(f"\n📂 載入索引中...")
storage_context = StorageContext.from_defaults(persist_dir=index_dir)
index = load_index_from_storage(storage_context)

# 取出所有 nodes
all_nodes = list(index.docstore.docs.values())
print(f"  → 此論文共有 {len(all_nodes)} 個 chunks\n")

# ── 關鍵字：這些字詞代表「正確的合成步驟 chunk」─────────
TARGET_KEYWORDS = [
    "0.27785", "0.46", "NaBH4", "NaBH₄", "FeSO4", "FeSO₄",
    "500 rpm", "900 rpm", "ice-bath", "ice bath", "20 mM",
    "Briefly", "freshly prepared"
]

def check_is_target(text: str) -> list:
    """回傳命中的關鍵字清單"""
    return [kw for kw in TARGET_KEYWORDS if kw.lower() in text.lower()]

def print_chunk(i: int, node, prefix: str = ""):
    """格式化印出一個 chunk"""
    text = node.get_content() if hasattr(node, 'get_content') else str(node.text)
    hits = check_is_target(text)
    flag = "🎯 ← 目標chunk！" if hits else ""
    print(f"\n  [{i}] {prefix} {flag}")
    if hits:
        print(f"       命中關鍵字：{', '.join(hits)}")
    print(f"       字元數：{len(text)}")
    print(f"       內容前150字：{text[:150].strip()!r}")
    print(f"       內容後100字：...{text[-100:].strip()!r}")

# ════════════════════════════════════════════════════════
# 步驟 0：先列出所有 chunks，找目標 chunk 在哪裡
# ════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print(f"【步驟 0】列出全部 {len(all_nodes)} 個 chunks，標示目標 chunk")
print(f"{'─'*65}")

target_chunk_ids = []
for i, (node_id, node) in enumerate(index.docstore.docs.items()):
    text = node.get_content() if hasattr(node, 'get_content') else str(node.text)
    hits = check_is_target(text)
    if hits:
        target_chunk_ids.append(node_id)
        print(f"\n  🎯 Chunk #{i} 是目標 chunk！")
        print(f"     node_id: {node_id[:40]}...")
        print(f"     命中關鍵字：{', '.join(hits)}")
        print(f"     字元數：{len(text)}")
        print(f"     完整內容：\n")
        # 印出完整內容
        for line in text.strip().split('\n'):
            print(f"     {line}")

if not target_chunk_ids:
    print("\n  ❌ 警告：所有 chunks 裡都找不到目標關鍵字！")
    print("     這表示 PDF 解析或 chunking 有問題，chunk 根本不存在")
    sys.exit(1)

print(f"\n  → 共找到 {len(target_chunk_ids)} 個目標 chunk")

# ════════════════════════════════════════════════════════
# 步驟 1：BM25 單獨測試
# ════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print(f"【步驟 1】BM25 檢索結果（top_k=8）")
print(f"  查詢：{query_text[:80]}...")
print(f"{'─'*65}")

bm25_retriever = BM25Retriever.from_defaults(
    nodes=all_nodes,
    similarity_top_k=8,
)

bm25_results = bm25_retriever.retrieve(query_text)
print(f"  → BM25 返回 {len(bm25_results)} 個 chunks")

bm25_found_target = False
for i, node in enumerate(bm25_results, 1):
    text = node.get_content()
    hits = check_is_target(text)
    if hits:
        bm25_found_target = True
    print_chunk(i, node, f"score={node.score:.4f}" if node.score else "")

if bm25_found_target:
    print(f"\n  ✅ BM25 有找到目標 chunk")
else:
    print(f"\n  ❌ BM25 沒有找到目標 chunk")

# ════════════════════════════════════════════════════════
# 步驟 2：向量搜尋單獨測試
# ════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print(f"【步驟 2】向量搜尋結果（top_k=8）")
print(f"  查詢：{query_text[:80]}...")
print(f"{'─'*65}")

vector_retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=8,
)

vector_results = vector_retriever.retrieve(query_text)
print(f"  → 向量搜尋返回 {len(vector_results)} 個 chunks")

vector_found_target = False
for i, node in enumerate(vector_results, 1):
    text = node.get_content()
    hits = check_is_target(text)
    if hits:
        vector_found_target = True
    print_chunk(i, node, f"score={node.score:.4f}" if node.score else "")

if vector_found_target:
    print(f"\n  ✅ 向量搜尋有找到目標 chunk")
else:
    print(f"\n  ❌ 向量搜尋沒有找到目標 chunk")

# ════════════════════════════════════════════════════════
# 步驟 3：Hybrid（融合後）結果
# ════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print(f"【步驟 3】Hybrid 融合結果（RRF 後，top_k=8）")
print(f"{'─'*65}")

hybrid_retriever = QueryFusionRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    similarity_top_k=8,
    num_queries=1,
    mode="reciprocal_rerank",
    use_async=False,
)

hybrid_results = hybrid_retriever.retrieve(query_text)
print(f"  → Hybrid 返回 {len(hybrid_results)} 個 chunks")

hybrid_found_target = False
for i, node in enumerate(hybrid_results, 1):
    text = node.get_content()
    hits = check_is_target(text)
    if hits:
        hybrid_found_target = True
    print_chunk(i, node, f"score={node.score:.4f}" if node.score else "")

if hybrid_found_target:
    print(f"\n  ✅ Hybrid 有找到目標 chunk")
else:
    print(f"\n  ❌ Hybrid 沒有找到目標 chunk")

# ════════════════════════════════════════════════════════
# 步驟 4：Reranker 後結果
# ════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print(f"【步驟 4】Reranker 後結果（top_n=4）")
print(f"{'─'*65}")

reranker = SentenceTransformerRerank(
    model="BAAI/bge-reranker-v2-m3",
    top_n=4,
)

query_bundle = QueryBundle(query_str=query_text)
reranked_results = reranker.postprocess_nodes(hybrid_results, query_bundle)
print(f"  → Reranker 返回 {len(reranked_results)} 個 chunks")

reranker_found_target = False
for i, node in enumerate(reranked_results, 1):
    text = node.get_content()
    hits = check_is_target(text)
    if hits:
        reranker_found_target = True
    print_chunk(i, node, f"score={node.score:.4f}" if node.score else "")

if reranker_found_target:
    print(f"\n  ✅ Reranker 後有保留目標 chunk")
else:
    print(f"\n  ❌ Reranker 後沒有目標 chunk（被排掉了）")

# ════════════════════════════════════════════════════════
# 最終診斷報告
# ════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"  最終診斷報告")
print(f"{'='*65}")
print(f"  目標 chunk 存在於索引中：✅ 是")
print(f"  BM25 能找到：          {'✅ 是' if bm25_found_target else '❌ 否'}")
print(f"  向量搜尋能找到：        {'✅ 是' if vector_found_target else '❌ 否'}")
print(f"  Hybrid 融合後能找到：   {'✅ 是' if hybrid_found_target else '❌ 否'}")
print(f"  Reranker 後能找到：     {'✅ 是' if reranker_found_target else '❌ 否'}")

print(f"\n  → 問題出在：", end="")
if not bm25_found_target and not vector_found_target:
    print("BM25 和向量搜尋都找不到，查詢和 chunk 語意/關鍵字差距太大")
elif bm25_found_target and not vector_found_target:
    print("向量搜尋找不到，embedding 語意距離問題")
elif not bm25_found_target and vector_found_target:
    print("BM25 找不到，關鍵字不重疊問題")
elif hybrid_found_target and not reranker_found_target:
    print("Reranker 把目標 chunk 排掉了，reranker 判斷有誤")
elif reranker_found_target:
    print("✅ 整個 retrieval pipeline 正常，問題在 LLM 回答階段")
else:
    print("Hybrid 融合階段出問題，RRF 分數計算異常")
print(f"{'='*65}\n")
