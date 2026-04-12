# rag/retriever.py
# 負責建立 Hybrid Retriever（BM25 + 向量搜尋融合）

from llama_index.core.retrievers import VectorIndexRetriever, QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever

import config as cfg


def build_hybrid_retriever(index):
    """
    建立 BM25 + 向量搜尋的混合檢索器。
    """
    vector_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=cfg.SIMILARITY_TOP_K,
    )

    nodes = list(index.docstore.docs.values())
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=cfg.SIMILARITY_TOP_K,
    )

    hybrid_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=cfg.SIMILARITY_TOP_K,
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=False,
    )

    return hybrid_retriever