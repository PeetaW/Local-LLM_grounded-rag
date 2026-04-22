# rag/indexer.py
# 負責建立或載入每篇論文的向量索引

import os
import sys
import shutil
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)

import config as cfg
from rag.pdf_loader import load_pdf_with_pymupdf
from rag.vl_processor import needs_vl_analysis, run_vl_analysis
from rag.metadata_manager import ensure_metadata
from rag.chunk_summarizer import add_summaries_to_nodes
from llama_index.core.node_parser import SentenceSplitter

def load_or_build_index(pdf_file: str):
    """
    若索引已存在則直接載入，否則解析 PDF 後建立新索引。
    回傳 index 物件。
    """
    index_dir = os.path.join(cfg.INDEX_BASE_DIR, pdf_file.replace(".pdf", ""))
    pdf_path = os.path.join(cfg.PAPERS_DIR, pdf_file)
    paper_name = pdf_file.replace(".pdf", "")

    # ── 自動 VL 分析（若尚未分析過）──────────────────
    vl_was_missing = needs_vl_analysis(paper_name)
    if vl_was_missing:
        print(f"  🔄 偵測到 {paper_name} 尚未進行 VL 分析，自動觸發...")
        run_vl_analysis(pdf_path)

    # ── 自動生成 metadata ─────────────────────────────
    ensure_metadata(pdf_path)

    # 如果 VL 剛剛才分析完，即使索引存在也要強制重建（舊索引沒有 VL 內容）
    if os.path.exists(index_dir) and not vl_was_missing:
        print(f"  載入既有索引：{pdf_file}")
        storage_context = StorageContext.from_defaults(persist_dir=index_dir)
        index = load_index_from_storage(storage_context)
    else:
        if vl_was_missing:
            print(f"  VL 分析剛完成，強制重建索引以納入圖片描述...")
        else:
            print(f"  建立新索引：{pdf_file}")
        docs = load_pdf_with_pymupdf(pdf_path)

        # 先切 chunks
        splitter = SentenceSplitter(
            chunk_size=cfg.CHUNK_SIZE,
            chunk_overlap=cfg.CHUNK_OVERLAP,
        )
        nodes = splitter.get_nodes_from_documents(docs)

        # 對每個 chunk 生成摘要標頭
        nodes = add_summaries_to_nodes(nodes, paper_name)

        # 用加了摘要的 nodes 建立索引
        index = VectorStoreIndex(nodes)
        index.storage_context.persist(persist_dir=index_dir)
        print(f"  ✓ 索引已儲存：{index_dir}")

    return index


def _cleanup_orphan_indexes(existing_paper_names: set):
    """
    刪除 index_storage/ 中沒有對應 PDF 的孤兒索引目錄。
    """
    if not os.path.isdir(cfg.INDEX_BASE_DIR):
        return
    skip = {os.path.basename(cfg.INDEX_CONFIG_PATH).replace(".json", ""), ".gitkeep"}
    for entry in os.listdir(cfg.INDEX_BASE_DIR):
        entry_path = os.path.join(cfg.INDEX_BASE_DIR, entry)
        if not os.path.isdir(entry_path):
            continue
        if entry in existing_paper_names or entry in skip:
            continue
        print(f"  🗑️  刪除孤兒索引（對應 PDF 已移除）：{entry}")
        shutil.rmtree(entry_path)


def load_all_papers():
    """
    掃描 papers/ 資料夾，對每篇 PDF 建立或載入索引。
    回傳 pdf_files 列表 和 paper_name -> index 的 dict。
    """
    pdf_files = sorted([
        f for f in os.listdir(cfg.PAPERS_DIR) if f.endswith(".pdf")
    ])
    paper_names = {f.replace(".pdf", "") for f in pdf_files}
    _cleanup_orphan_indexes(paper_names)
    print(f"找到 {len(pdf_files)} 篇論文，開始建立索引...\n")

    indexes = {}
    for pdf_file in pdf_files:
        index = load_or_build_index(pdf_file)
        paper_name = pdf_file.replace(".pdf", "")
        indexes[paper_name] = index
        print(f"  ✓ 完成：{pdf_file}")

    return pdf_files, indexes


# ── 新增在 indexer.py 最底部 ──────────────────────────

import json
from llama_index.core.query_engine import RetrieverQueryEngine
from rag.retriever import build_hybrid_retriever
from rag.reranker import build_reranker


def check_index_config():
    """
    檢查 index_storage/config.json 與目前設定是否一致。
    不一致時直接 exit()，一致或首次執行時正常繼續。
    """
    current = cfg.INDEX_BUILD_CONFIG
    config_path = cfg.INDEX_CONFIG_PATH

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            saved = json.load(f)
        if saved != current:
            print("⚠️  chunk參數已變更！")
            print(f"   舊設定：{saved}")
            print(f"   新設定：{current}")
            print("   請手動刪除 index_storage 資料夾後重新執行")
            sys.exit(1)
        else:
            print("chunk設定與索引一致，直接載入索引")
    else:
        os.makedirs(cfg.INDEX_BASE_DIR, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(current, f, ensure_ascii=False, indent=2)
        print("✓ 首次執行，已儲存chunk設定\n")


def build_hybrid_query_engine(index):
    """
    組裝完整的查詢引擎：Hybrid Retriever + Reranker。
    """
    hybrid_retriever = build_hybrid_retriever(index)
    reranker = build_reranker()
    return RetrieverQueryEngine.from_args(
        retriever=hybrid_retriever,
        response_mode="compact",
        node_postprocessors=[reranker],
    )