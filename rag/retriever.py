# rag/retriever.py
# 負責建立 Hybrid Retriever（BM25 + 向量搜尋融合）

from llama_index.core.retrievers import VectorIndexRetriever, QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever

import config as cfg


def build_hybrid_retriever(index):
    """
    建立 BM25 + 向量搜尋的混合檢索器。
    """
    nodes = list(index.docstore.docs.values())
    # rerank 開啟時多檢索一些候選（RERANK_CANDIDATE_K）讓 cross-encoder 精選；
    # 關閉時直接檢索最終數量（RERANKER_TOP_N）。
    # 夾住 top_k：小論文 chunk 數可能 < target，bm25s 在 k > corpus 時會報錯（P1 根因），
    # 夾到該篇實際 chunk 數即可徹底消除。
    target_k = cfg.RERANK_CANDIDATE_K if cfg.RERANK_ENABLED else cfg.RERANKER_TOP_N
    top_k = min(target_k, len(nodes)) if nodes else target_k

    vector_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k,
    )

    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=top_k,
    )

    hybrid_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=top_k,
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=False,
    )

    return hybrid_retriever