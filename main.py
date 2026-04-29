"""
main.py
==================
目的：
    針對 ZVI（零價鐵奈米粒子）相關學術論文進行 RAG（檢索增強生成）問答。
    每篇論文建立獨立的向量索引，透過 StructuredPlanning 將複雜問題
    自動拆解成子問題，分別查詢各論文後綜合回答。

使用方式：
    1. 確認 Ollama 已啟動，且 deepseek-r1:32b 與 bge-m3 已下載
    2. 將 PDF 論文放入 papers/ 資料夾
    3. 啟動虛擬環境：conda activate llm_env
    4. 執行：python main.py

    首次執行：自動建立索引並儲存至 index_storage/
    後續執行：直接載入既有索引，不重新建立

    ⚠️  若修改 config.py 中任何 INDEX_BUILD_CONFIG 相關參數
       （chunk_size / chunk_overlap / embed_model / parser / include_vl），
        需手動刪除 index_storage/ 資料夾後重新執行

    附加模式：
    --test-chunks            檢查第一篇論文的 chunk 切割品質
    --test-chunks --paper <檔名>  指定論文檔名
    --test-chunks --n <數量>     指定顯示幾個 chunks（預設 10）

專案架構：
    main.py                  主程式入口（本檔案）
    config.py                所有參數集中管理
    rag/
        llm_client.py        LLM & Embedding 初始化
        pdf_loader.py        PDF 解析 + VL 圖片描述融合
        indexer.py           索引建立 / 載入 / config 檢查
        retriever.py         Hybrid Retriever（BM25 + 向量）
        reranker.py          Reranker（bge-reranker-v2-m3）
        query_engine.py      StructuredPlanning 子問題拆解與綜合
        memory.py            ChromaDB 跨 session 記憶
        chunk_inspector.py   Chunk 品質檢查工具

技術備註：
    - LLM：deepseek-r1:32b，直連 Ollama（port 11434）
    - Embedding：bge-m3，透過 Ollama 直連（port 11434）
    - PDF 解析：PyMuPDF，含參考文獻自動過濾
    - Chunking：SentenceSplitter，chunk_size=1024 / overlap=256
    - 檢索：BM25 + 向量 Hybrid Search，similarity_top_k=8
    - Reranker：bge-reranker-v2-m3，top_n=8
    - 查詢規劃：StructuredPlanning（子問題在記憶體內拆解，不寫檔）
    - 記憶：ChromaDB PersistentClient，跨 session 保留
    - Timeout：28800 秒（8 小時），支援長時間推理
    - ⚠️  不要隨意 upgrade 任何套件！
"""

import config as cfg
import os
import sys


# ── --rerun-vl 模式：在所有重度初始化之前攔截 ──────────────
if "--rerun-vl" in sys.argv:
    from rag.vl_processor import rerun_failed_vl, get_failed_vl_images
    from rag.indexer import reindex_paper

    idx = sys.argv.index("--rerun-vl")
    if idx + 1 >= len(sys.argv):
        print("⚠️  請指定論文名稱，例如：python main.py --rerun-vl 論文名稱")
        sys.exit(1)

    target_name = sys.argv[idx + 1]
    target_pdf  = target_name if target_name.endswith(".pdf") else f"{target_name}.pdf"
    pdf_path    = os.path.join(cfg.PAPERS_DIR, target_pdf)

    if not os.path.exists(pdf_path):
        print(f"⚠️  找不到 PDF：{pdf_path}")
        sys.exit(1)

    print(f"\n🔄 重新掃描失敗圖片：{target_name}")
    fixed = rerun_failed_vl(pdf_path)
    if fixed > 0:
        print(f"\n✅ {fixed} 張圖片修復成功，重建索引中...")
        reindex_paper(target_pdf)
        print(f"✅ 索引重建完成：{target_name}")
    else:
        remaining = get_failed_vl_images(target_name)
        if remaining:
            print(f"\n⚠️  {len(remaining)} 張圖片仍需人工審查，索引未重建")
        else:
            print(f"✅ {target_name} 沒有待審查的圖片")
    sys.exit(0)


# ── LlamaIndex 核心 ────────────────────────────────────
from llama_index.core import Settings


# ── LLM & Embedding 初始化 ─────────────────────────────
from rag.llm_client import init_llm_and_embedding
init_llm_and_embedding()


from rag.indexer import check_index_config, load_all_papers, build_hybrid_query_engine
check_index_config()


# ── PDF 載入 ───────────────────────────────────────────
from rag.pdf_loader import load_pdf_with_pymupdf


# ── 載入所有論文索引 ───────────────────────────────────
pdf_files, indexes = load_all_papers()

paper_engines = {}

for pdf_file in pdf_files:
    paper_name = pdf_file.replace(".pdf", "")
    index = indexes[paper_name]
    engine = build_hybrid_query_engine(index)
    paper_engines[paper_name] = engine


# ── Query Engine ───────────────────────────────────────
from rag.query_pipeline import execute_structured_query


# ── 記憶系統初始化 ─────────────────────────────────────
from rag.memory import init_memory

episodic_collection, preference_collection = init_memory()


# ══════════════════════════════════════════════════════
# 注意：查詢主流程已移至 api.py 的 /query endpoint
# main.py 只負責初始化 paper_engines 和 memory_collection
# ══════════════════════════════════════════════════════

from rag.chunk_inspector import inspect_chunks


# ── 主程式 ─────────────────────────────────────────────

# --test-chunks 模式：只做 chunk 檢查，不執行問答
if "--test-chunks" in sys.argv:
    pdf_files_all = sorted([f for f in os.listdir(cfg.PAPERS_DIR) if f.endswith(".pdf")])
    if not pdf_files_all:
        print("⚠️  papers/ 資料夾內找不到 PDF")
        sys.exit(1)

    # 預設檢查第一篇；可用 --paper 指定論文檔名
    target_pdf = pdf_files_all[0]
    if "--paper" in sys.argv:
        idx = sys.argv.index("--paper")
        if idx + 1 < len(sys.argv):
            target_pdf = sys.argv[idx + 1]

    # 可用 --n 指定要看幾個 chunks（預設 10）
    num_to_show = 10
    if "--n" in sys.argv:
        idx = sys.argv.index("--n")
        if idx + 1 < len(sys.argv):
            try:
                num_to_show = int(sys.argv[idx + 1])
            except ValueError:
                print(f"⚠️  --n 參數必須是整數，收到：{sys.argv[idx + 1]}，使用預設值 {num_to_show}")

    inspect_chunks(os.path.join(cfg.PAPERS_DIR, target_pdf), num_chunks=num_to_show)
    sys.exit(0)
