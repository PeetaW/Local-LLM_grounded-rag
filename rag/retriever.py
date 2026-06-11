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
    # 夾住 top_k：小論文 chunk 數可能 < RERANK_CANDIDATE_K，
    # bm25s 在 k > corpus size 時會直接報錯（時好時壞，視平行競爭而定），
    # 是 P1「間歇性空白」的根因。夾到該篇實際 chunk 數即可徹底消除。
    top_k = min(cfg.RERANK_CANDIDATE_K, len(nodes)) if nodes else cfg.RERANK_CANDIDATE_K

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