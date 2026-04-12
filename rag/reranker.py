# rag/reranker.py
# 負責建立 Reranker（bge-reranker-v2-m3）

from llama_index.core.postprocessor import SentenceTransformerRerank

import config as cfg


def build_reranker():
    """
    建立 bge-reranker-v2-m3 reranker。
    """
    return SentenceTransformerRerank(
        model=cfg.RERANKER_MODEL,
        top_n=cfg.RERANKER_TOP_N,
    )