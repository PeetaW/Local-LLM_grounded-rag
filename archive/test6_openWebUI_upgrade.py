"""
test6_openWebUI_upgrade_fixed.py
==================
目的：
    針對ZVI（零價鐵奈米粒子）相關學術論文進行RAG（檢索增強生成）問答。
    每篇論文建立獨立的向量索引，透過StructuredPlanning將複雜問題
    自動拆解成子問題，分別查詢各論文後綜合回答。

使用方式：
    1. 確認 Ollama 已啟動，且 deepseek-r1:32b 與 bge-m3 已下載
    2. 確認 Open WebUI 正在執行（預設 http://localhost:8080）
    3. 將PDF論文放入 papers/ 資料夾
    4. 啟動虛擬環境：conda activate llm_env
    5. 執行：python test6_openWebUI_upgrade_fixed.py

    首次執行：自動建立索引並儲存至 index_storage/
    後續執行：直接載入既有索引，不重新建立

    ⚠️  若修改 chunk_size / chunk_overlap / embed_model / parser / include_vl 任一參數，
        需手動刪除 index_storage/ 資料夾後重新執行

修正紀錄（對比 test6_openWebUI_upgrade.py）：
    ✅ memsearch + milvus-lite → ChromaDB（Windows 不支援 milvus-lite）
    ✅ recall_memories 改為同步函數（移除 async / await）
    ✅ 移除 await ms.index()
    ✅ 新增 streaming 輸出（最終綜合回答字出來就印）
    ── api_base 保留 Open WebUI (port 8080)（不動）
    ── api_key 保留原本的 JWT token（不動）
    ── context_window 保留 16384（不動）
"""

import os
import time
import httpx
import json
import asyncio
import uuid
import fitz  # PyMuPDF
from datetime import date
from pathlib import Path

# ── LlamaIndex 核心 ────────────────────────────────────
from llama_index.core import (
    VectorStoreIndex,
    Settings, StorageContext,
    load_index_from_storage,
)
from llama_index.core.schema import Document
from llama_index.core.retrievers import VectorIndexRetriever, QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata

# ── ChromaDB（跨 session 記憶，取代 memsearch）─────────
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

# ── 連線設定 ───────────────────────────────────────────
http_client = httpx.Client(timeout=httpx.Timeout(14400.0, connect=30.0))

Settings.llm = OpenAILike(
    model="deepseek-r1:32b",
    api_base="http://localhost:8080/api/v1",        # ← Open WebUI，保留不動
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6ImFmYTlkZDc2LTNkYTctNGZiNy1iODFjLWYwNmM2MDgzMDc2YyIsImV4cCI6MTc3NDYyMDEyOSwianRpIjoiZmYzNzU3ZGYtYzQ1ZS00YWExLTg3MGUtODdiNDAxMTgzYjE0In0.NTQPKncbuvtN4--fBLVMuNmxRlOYtfeSHGU7GdGg22",  # ← 保留不動
    is_chat_model=True,
    timeout=14400.0,
    http_client=http_client,
    context_window=16384,
    additional_kwargs={"options": {"num_ctx": 16384}},
    system_prompt=(
        "你是一個學術論文分析助手。"
        "請只根據提供的論文內容回答問題，使用繁體中文。"
        "請務必使用繁體中文，絕對不可使用簡體中文。"
        "如果論文中沒有相關資訊，請直接說明「此論文未涉及此議題」，"
        "不要自行推測或補充論文以外的內容。"
        "回答時請盡量引用論文中的具體數據、步驟與條件。"
    ),
)

# ── bge-m3 embedding ──────────────────────────────────
Settings.embed_model = OllamaEmbedding(
    model_name="bge-m3",
    base_url="http://localhost:11434",
)

Settings.chunk_size = 1024
Settings.chunk_overlap = 256

# 儲存當前chunk設定
config = {
    "chunk_size": 1024,
    "chunk_overlap": 200,
    "embed_model": "bge-m3",
    "parser": "pymupdf",
    "include_vl": True,
}
INDEX_BASE_DIR = "index_storage"
config_path = "index_storage/config.json"

# 判斷設定有沒有改變
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        saved_config = json.load(f)
    if saved_config != config:
        print("⚠️  chunk參數已變更！")
        print(f"   舊設定：{saved_config}")
        print(f"   新設定：{config}")
        print("   請手動刪除 index_storage 資料夾後重新執行")
        exit()
    else:
        print("chunk設定與索引一致，直接載入索引")
else:
    os.makedirs(INDEX_BASE_DIR, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print("✓ 首次執行，已儲存chunk設定\n")


# ── PyMuPDF解析PDF + VL圖片描述融合 ──────────────────
def load_pdf_with_pymupdf(pdf_path, vl_output_dir="vl_test_output"):
    pdf_filename = os.path.basename(pdf_path)
    paper_name = pdf_filename.replace(".pdf", "")

    doc = fitz.open(pdf_path)
    full_text = ""
    ref_section_started = False

    for page_num, page in enumerate(doc, 1):
        text = page.get_text("text")
        if not text.strip():
            continue

        lines = text.strip().split("\n")
        has_ref_header = any(
            line.strip() in ("References", "參考文獻", "REFERENCES")
            for line in lines
        )
        doi_count = text.lower().count("doi:")

        if has_ref_header and doi_count >= 3:
            ref_section_started = True
            print(f"  🔍 第{page_num}頁偵測到參考文獻區塊，後續頁面略過")

        if ref_section_started:
            continue

        full_text += f"\n[第{page_num}頁]\n{text}"

    doc.close()

    if not full_text.strip():
        print(f"  ⚠️  參考文獻過濾後內容為空，退回完整文字")
        doc = fitz.open(pdf_path)
        full_text = ""
        for page_num, page in enumerate(doc, 1):
            text = page.get_text("text")
            if text.strip():
                full_text += f"\n[第{page_num}頁]\n{text}"
        doc.close()

    documents = [Document(
        text=full_text,
        metadata={
            "file_name": pdf_filename,
            "file_path": pdf_path,
            "source_type": "pdf_text",
            "parser": "pymupdf",
        }
    )]

    vl_result_path = os.path.join(vl_output_dir, paper_name, "vl_test_result.json")

    if not os.path.exists(vl_result_path):
        print(f"  ℹ️  找不到VL描述：{paper_name}，僅使用PDF文字")
        return documents

    with open(vl_result_path, "r", encoding="utf-8") as f:
        vl_data = json.load(f)

    img_count = 0
    skipped_count = 0
    for img in vl_data.get("images", []):
        if not img.get("success", False):
            continue
        if img.get("needs_review", False):
            skipped_count += 1
            continue

        description = img.get("description", "").strip()
        if not description:
            continue

        img_text = (
            f"【圖片描述】\n"
            f"來源論文：{pdf_filename}\n"
            f"圖片檔名：{img['filename']}（第{img['page']}頁）\n\n"
            f"{description}"
        )

        documents.append(Document(
            text=img_text,
            metadata={
                "file_name": pdf_filename,
                "source_type": "image_description",
                "image_filename": img["filename"],
                "page": img["page"],
            }
        ))
        img_count += 1

    print(f"  ✅ 載入 {img_count} 張圖片描述", end="")
    if skipped_count > 0:
        print(f"（跳過 {skipped_count} 張 needs_review）", end="")
    print()

    return documents


# ── 建立 Hybrid 查詢引擎（向量 + BM25）───────────────
def build_hybrid_query_engine(index):
    vector_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=8,
    )

    nodes = list(index.docstore.docs.values())
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=8,
    )

    hybrid_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=8,
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=False,
    )

    return RetrieverQueryEngine.from_args(
        retriever=hybrid_retriever,
        response_mode="compact",
    )


# ── 每篇paper建立獨立索引 ──────────────────────────────
papers_dir = "papers"
pdf_files = sorted([f for f in os.listdir(papers_dir) if f.endswith(".pdf")])

print(f"找到 {len(pdf_files)} 篇論文，開始建立索引...\n")

query_engine_tools = []
paper_engines = {}

for pdf_file in pdf_files:
    index_dir = os.path.join(INDEX_BASE_DIR, pdf_file.replace(".pdf", ""))

    if os.path.exists(index_dir):
        print(f"  載入既有索引：{pdf_file}")
        storage_context = StorageContext.from_defaults(persist_dir=index_dir)
        index = load_index_from_storage(storage_context)
    else:
        print(f"  建立新索引：{pdf_file}")
        docs = load_pdf_with_pymupdf(os.path.join(papers_dir, pdf_file))
        index = VectorStoreIndex.from_documents(docs)
        index.storage_context.persist(persist_dir=index_dir)
        print(f"  ✓ 索引已儲存：{index_dir}")

    engine = build_hybrid_query_engine(index)

    paper_name = pdf_file.replace(".pdf", "")
    paper_engines[paper_name] = engine

    tool = QueryEngineTool(
        query_engine=engine,
        metadata=ToolMetadata(
            name=paper_name.replace(" ", "_").replace("-", "_"),
            description=(
                f"包含論文 {pdf_file} 的完整內容，"
                f"涵蓋合成方法、實驗結果與討論"
            ),
        ),
    )
    query_engine_tools.append(tool)
    print(f"  ✓ 完成：{pdf_file}")


# ── StructuredPlanning 子問題拆解 ─────────────────────
def plan_sub_questions(question: str, paper_names: list) -> list:
    paper_list_str = "\n".join(f"- {p}" for p in paper_names)

    prompt = f"""你是一個查詢規劃助手。
    
可用的論文清單：
{paper_list_str}

使用者的複合問題：
{question}

請將這個問題拆解成多個子問題，每個子問題針對一篇或所有論文。
以 JSON 陣列回傳，格式如下，只輸出 JSON，不要其他文字：
[
  {{"paper": "論文檔名（不含.pdf）或 ALL（代表所有論文）", "sub_q": "子問題內容"}},
  ...
]
"""
    response = Settings.llm.complete(prompt)

    try:
        raw = response.text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        sub_questions = json.loads(raw)
        return sub_questions
    except json.JSONDecodeError:
        print("  ⚠️  子問題拆解失敗，改為對所有論文問同一問題")
        return [{"paper": "ALL", "sub_q": question}]


def execute_structured_query(question: str) -> str:
    paper_names = list(paper_engines.keys())

    print("\n  📋 拆解子問題中...")
    sub_questions = plan_sub_questions(question, paper_names)
    print(f"  → 拆出 {len(sub_questions)} 個子問題")

    sub_answers = []
    for i, sq in enumerate(sub_questions, 1):
        paper = sq.get("paper", "ALL")
        sub_q = sq.get("sub_q", "")
        print(f"\n  [{i}/{len(sub_questions)}] 查詢：{sub_q[:50]}...")

        if paper == "ALL":
            for name, engine in paper_engines.items():
                try:
                    result = engine.query(sub_q)
                    print(f"\n  ── 【{name}】回覆 ──\n  {result}")
                    sub_answers.append(f"【{name}】\n{result}")
                except Exception as e:
                    sub_answers.append(f"【{name}】查詢失敗：{e}")
        else:
            engine = paper_engines.get(paper)
            if engine is None:
                matched = next(
                    (k for k in paper_engines if paper in k or k in paper),
                    None
                )
                engine = paper_engines.get(matched) if matched else None

            if engine:
                try:
                    result = engine.query(sub_q)
                    print(f"\n  ── 【{paper}】回覆 ──\n  {result}")
                    sub_answers.append(f"【{paper}】\n{result}")
                except Exception as e:
                    sub_answers.append(f"【{paper}】查詢失敗：{e}")
            else:
                sub_answers.append(f"【{paper}】找不到對應論文")

    print("\n  🔗 綜合所有子答案中...")
    combined = "\n\n".join(sub_answers)

    synthesis_prompt = f"""
以下是針對各子問題的查詢結果：

{combined}

---
原始問題：{question}

請根據以上資料，用繁體中文撰寫一份完整、有條理的綜合回答。
如果各論文有差異，請明確比較。
只使用上述資料中的內容，不要自行補充。
"""

    # streaming 輸出，字出來就印，不等全部完成
    print("\n 最終綜合回答：")
    full_text = ""
    for chunk in Settings.llm.stream_complete(synthesis_prompt):
        print(chunk.delta, end="", flush=True)
        full_text += chunk.delta
    print("\n")
    return full_text


# ══════════════════════════════════════════════════════
# ChromaDB 跨 session 記憶系統（取代 memsearch）
# 資料永久存在 ./memory_db/ 資料夾，重啟後仍可讀取
# ══════════════════════════════════════════════════════
MEMORY_DB_DIR = "./memory_db"

chroma_client = chromadb.PersistentClient(path=MEMORY_DB_DIR)

ollama_ef = OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name="bge-m3",
)

memory_collection = chroma_client.get_or_create_collection(
    name="rag_memory",
    metadata={"description": "RAG問答的跨session記憶"},
    embedding_function=ollama_ef,
)

print(f"✓ 記憶系統啟動（ChromaDB），目前已有 {memory_collection.count()} 筆記憶")


def save_memory(question: str, answer: str):
    """把這次問答存入 ChromaDB 記憶"""
    summary = answer[:500] + "..." if len(answer) > 500 else answer
    memory_collection.add(
        documents=[f"問：{question}\n答：{summary}"],
        metadatas=[{
            "date": str(date.today()),
            "question": question[:200],
        }],
        ids=[str(uuid.uuid4())]
    )


def recall_memories(question: str) -> str:
    """
    用語意搜尋找出過去最相關的 3 筆記憶
    ⚠️  改為同步函數，不需要 async / await
    """
    if memory_collection.count() == 0:
        return ""

    try:
        results = memory_collection.query(
            query_texts=[question],
            n_results=min(3, memory_collection.count()),
        )

        documents = results.get("documents", [[]])[0]
        if not documents:
            return ""

        context = "\n".join(f"- {doc[:200]}" for doc in documents)
        return f"\n【相關歷史問答記憶】\n{context}\n"

    except Exception as e:
        print(f"  ⚠️  記憶搜尋失敗（不影響主流程）：{e}")
        return ""


# ══════════════════════════════════════════════════════
# 主查詢流程（整合記憶）
# ══════════════════════════════════════════════════════
async def query_with_memory(question: str) -> str:
    # 1. 搜尋過去記憶（同步呼叫，不需要 await）
    memory_context = recall_memories(question)
    if memory_context:
        print(f"  💭 找到相關記憶，注入 context")
        augmented_question = f"{memory_context}\n\n當前問題：{question}"
    else:
        augmented_question = question

    # 2. 執行 StructuredPlanning 查詢
    answer = execute_structured_query(augmented_question)

    # 3. 把這次問答存入記憶（移除舊版的 await ms.index()）
    save_memory(question, answer)
    print(f"  💾 記憶已儲存（目前共 {memory_collection.count()} 筆）")

    return answer


# ── 主程式 ─────────────────────────────────────────────
questions = [
    "請摘要每篇論文的核心合成方法與主要發現，請用繁體中文回答",
]

total_start = time.time()

for i, question in enumerate(questions, 1):
    print(f"\n{'='*65}")
    print(f"[問題 {i}/{len(questions)}] {question}")
    print(f"{'='*65}")

    q_start = time.time()
    response = asyncio.run(query_with_memory(question))
    q_elapsed = time.time() - q_start

    minutes = int(q_elapsed // 60)
    seconds = int(q_elapsed % 60)

    print(f"\n⏱ 本題耗時：{minutes}分{seconds}秒")

total_elapsed = time.time() - total_start
total_min = int(total_elapsed // 60)
total_sec = int(total_elapsed % 60)

print(f"\n{'='*65}")
print(f"✓ 完成！總耗時：{total_min}分{total_sec}秒")
print(f"{'='*65}")