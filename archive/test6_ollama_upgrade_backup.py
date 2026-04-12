"""
test6_ollama_upgrade.py
==================
目的：
    針對ZVI（零價鐵奈米粒子）相關學術論文進行RAG（檢索增強生成）問答。
    每篇論文建立獨立的向量索引，透過StructuredPlanning將複雜問題
    自動拆解成子問題，分別查詢各論文後綜合回答。

使用方式：
    1. 確認 Ollama 已啟動，且 deepseek-r1:32b 與 bge-m3 已下載
    2. 將PDF論文放入 papers/ 資料夾
    3. 啟動虛擬環境：conda activate llm_env
    4. 執行：python test6_ollama_upgrade.py
    
    首次執行：自動建立索引並儲存至 index_storage/
    後續執行：直接載入既有索引，不重新建立
    
    ⚠️  若修改 chunk_size / chunk_overlap / embed_model / parser / include_vl 任一參數，
        需手動刪除 index_storage/ 資料夾後重新執行

作者備註：
    - LLM：deepseek-r1:32b，直連 Ollama（port 11434）
    - Embedding：bge-m3，透過 Ollama 直連（port 11434）
    - PDF解析器：PyMuPDF（fitz），取代原本的 pypdf，解析品質更佳
    - chunk_size=1024 / chunk_overlap=256，經測試後的穩定設定
    - similarity_top_k=8，每次檢索取最相關的8個chunk
    - StructuredPlanning：大問題自動拆解成子問題，每篇論文獨立查詢
    - timeout設定28800秒（8小時），避免長時間推理被中斷
    - 跨 session 記憶：ChromaDB（取代 memsearch，原生支援 Windows）
    - ⚠️  不要隨意 upgrade 任何套件！

升級重點：
    1. Embedding：nomic-embed-text → bge-m3（中英混合更強）
    2. 檢索：純向量 → BM25 + 向量 Hybrid Search
    3. 查詢引擎：SubQuestionQueryEngine → StructuredPlanning
       （LLM 在記憶體內拆解子問題，不寫檔，結束即消失）
    4. 跨 session 記憶：memsearch → ChromaDB（Windows 相容）
    5. Context window：4096 → 32768
    6. 連線：Open WebUI (8080) → 直連 Ollama (11434)
"""

import os
import sys
import time
import httpx
import json
import asyncio
import re
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
from llama_index.core.postprocessor import SentenceTransformerRerank


# ── ChromaDB（跨 session 記憶，取代 memsearch）─────────
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

# ── 連線設定 ───────────────────────────────────────────
http_client = httpx.Client(timeout=httpx.Timeout(28800.0, connect=30.0))

Settings.llm = OpenAILike(
    model="deepseek-r1:32b",
    api_base="http://localhost:11434/v1",   # 直連 Ollama，不經過 Open WebUI
    api_key="ollama",                        # Ollama 不需要真實 key
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
    "chunk_overlap": 256,
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
    """
    用PyMuPDF解析PDF，抽取文字內容
    同時載入對應的VL圖片描述（若存在）
    回傳多個 Document：1個文字 + N個圖片描述
    """
    pdf_filename = os.path.basename(pdf_path)
    paper_name = pdf_filename.replace(".pdf", "")

    doc = fitz.open(pdf_path)
    full_text = ""
    ref_section_started = False

    for page_num, page in enumerate(doc, 1):
        text = page.get_text("text")
        text = text.replace("\r\n", " ").replace("\r", " ")
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

        full_text += f"\n{text}"

    doc.close()

    if not full_text.strip():
        print(f"  ⚠️  參考文獻過濾後內容為空，退回完整文字")
        doc = fitz.open(pdf_path)
        full_text = ""
        for page_num, page in enumerate(doc, 1):
            text = page.get_text("text")
            text = text.replace("\r\n", " ").replace("\r", " ")
            if text.strip():
                full_text += f"\n{text}"
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

    reranker = SentenceTransformerRerank(
        model="BAAI/bge-reranker-v2-m3",
        top_n=4,
    )

    return RetrieverQueryEngine.from_args(
        retriever=hybrid_retriever,
        response_mode="refine",
        node_postprocessors=[reranker],
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
    """
    讓 LLM 把複合問題拆成子問題清單（純記憶體操作，不寫檔）
    回傳格式：[{"paper": "論文名稱或ALL", "sub_q": "子問題"}]
    """
    paper_list_str = "\n".join(f"- {p}" for p in paper_names)

    prompt = f"""你是一個查詢規劃助手。
    
可用的論文清單：
{paper_list_str}

使用者的複合問題：
{question}

請自行判斷是否需要將這個問題拆解成多個仔問題。若需要將這個問題拆解，就自動拆成多個子問題，把每個子問題針對一篇或所有論文。
子問題請用英文撰寫，使用學術論文常見的詞彙（例如 synthesis procedure, preparation method, reagents used, experimental conditions）
若問題涉及合成或實驗方法，必須額外拆出一個子問題專門詢問具體操作參數，例如：amounts, concentrations, temperature, stirring speed, reaction time。
以 JSON 陣列回傳，格式如下，只輸出 JSON，不要其他文字：
[
  {{"paper": "論文檔名（不含.pdf）或 ALL（代表所有論文）", "sub_q": "子問題內容"}},
  ...
]
"""
    response = Settings.llm.complete(prompt)

    try:
        raw = response.text.strip()
        # 清除 deepseek-r1 的 <think>...</think> 推理內容，避免污染 JSON 解析
        raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        sub_questions = json.loads(raw)
        print(f"  → 子問題內容：{[sq.get('sub_q', '')[:60] for sq in sub_questions]}")
        return sub_questions
    except json.JSONDecodeError:
        print(f"  ⚠️  子問題拆解失敗，raw 內容：{raw[:200]}")
        print("       改為對所有論文問同一問題")
        return [{"paper": "ALL", "sub_q": question}]


def execute_structured_query(question: str, memory_context: str = "") -> str:
    """
    StructuredPlanning 主流程：
    1. LLM 拆解子問題（記憶體內）
    2. 逐一查詢（線性，避免 VRAM 爆炸）
    3. 合併所有子答案，LLM 做最終綜合
    """
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
                    # 印出子問題回覆
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
                    # 印出子問題回覆
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

{f"---{chr(10)}【相關歷史問答記憶，僅供參考】{memory_context}" if memory_context else ""}

---
原始問題：{question}

請根據以上資料，用繁體中文撰寫一份完整、有條理的綜合回答。
如果各論文有差異，請明確比較。
只使用上述資料中的內容，不要自行補充。
"""
    
# Streaming 輸出最終答案，字出來就印，不等全部完成
    print("\n 最終綜合回答：")
    full_text = ""
    for chunk in Settings.llm.stream_complete(synthesis_prompt):
        print(chunk.delta, end="", flush=True)
        full_text += chunk.delta
    print("\n")  # 換行
    return full_text


# ══════════════════════════════════════════════════════
# ChromaDB 跨 session 記憶系統
# 資料永久存在 ./memory_db/ 資料夾，重啟後仍可讀取
# ══════════════════════════════════════════════════════
MEMORY_DB_DIR = "./memory_db"

# PersistentClient：資料存在硬碟，跨 session 保留
# 第一次執行自動建立資料夾，之後自動讀取
chroma_client = chromadb.PersistentClient(path=MEMORY_DB_DIR)

# 指定用 Ollama 的 bge-m3 做 embedding，不讓 ChromaDB 自己下載模型
ollama_ef = OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name="bge-m3",
)

# collection = 記憶的「資料表」
# get_or_create：已存在就讀取，不存在就新建，不會重複
memory_collection = chroma_client.get_or_create_collection(
    name="rag_memory",
    metadata={"description": "RAG問答的跨session記憶"},
    embedding_function=ollama_ef,
)

print(f"✓ 記憶系統啟動（ChromaDB），目前已有 {memory_collection.count()} 筆記憶")


def save_memory(question: str, answer: str):
    """
    把這次問答存入 ChromaDB 記憶
    - 用時間戳作為唯一 ID，避免重複
    - 只存前500字的摘要，避免塞太多
    """
    import uuid
    summary = answer[:500] + "..." if len(answer) > 500 else answer
    
    # 存入 ChromaDB
    # document = 實際存的文字內容（用來做向量搜尋）
    # metadata = 額外資訊（日期、原始問題）
    # id = 唯一識別碼
    memory_collection.add(
        documents=[f"問：{question}\n答：{summary}"],
        metadatas=[{
            "date": str(date.today()),
            "question": question[:200],  # metadata 不能太長
        }],
        ids=[str(uuid.uuid4())]  # 每筆記憶給一個唯一ID
    )


def recall_memories(question: str) -> str:
    """
    用語意搜尋找出過去最相關的 3 筆記憶
    ChromaDB 內建向量搜尋，不需要額外呼叫 Ollama
    """
    # 如果記憶庫是空的，直接回傳空字串
    if memory_collection.count() == 0:
        return ""

    try:
        results = memory_collection.query(
            query_texts=[question],
            n_results=min(3, memory_collection.count()),  # 最多3筆，但不超過總數
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
# 注意：recall_memories 已改為同步函數，不需要 async
# ══════════════════════════════════════════════════════
async def query_with_memory(question: str) -> str:
    # 1. 搜尋過去記憶
    memory_context = recall_memories(question)  # 同步呼叫，不需要 await
    if memory_context:
        print(f"  💭 找到相關記憶，注入 context")

    # 2. 執行 StructuredPlanning 查詢
    answer = execute_structured_query(question, memory_context)

    # 3. 把這次問答存入記憶
    save_memory(question, answer)
    print(f"  💾 記憶已儲存（目前共 {memory_collection.count()} 筆）")

    return answer


# ══════════════════════════════════════════════════════
# Chunk 品質檢查工具（執行：python test6_ollama_upgrade.py --test-chunks）
# ══════════════════════════════════════════════════════
def inspect_chunks(pdf_path: str, num_chunks: int = 10):
    """
    載入單篇 PDF，切成 chunks 後逐一印出，供肉眼檢查。
    重點確認：
      1. 句子/段落有沒有被截斷到一半
      2. 化學式、數字、單位有沒有跟說明分開
      3. 表格內容有沒有被切壞
      4. chunk 之間的 overlap 是否有正確重疊
    """
    from llama_index.core.node_parser import SentenceSplitter

    print(f"\n{'='*65}")
    print(f"  Chunk 品質檢查")
    print(f"  PDF：{pdf_path}")
    print(f"  chunk_size={Settings.chunk_size}, chunk_overlap={Settings.chunk_overlap}")
    print(f"{'='*65}\n")

    # 載入 PDF（走你現有的 PyMuPDF 解析器，結果跟正式流程完全一樣）
    docs = load_pdf_with_pymupdf(pdf_path)
    text_docs = [d for d in docs if d.metadata.get("source_type") == "pdf_text"]

    if not text_docs:
        print("⚠️  找不到 PDF 文字內容，請確認路徑正確")
        return

    # 用跟 LlamaIndex 一樣的 SentenceSplitter 切割
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
        # 粗估 token 數：英文約 4 字元/token，中文約 1.5 字元/token
        token_estimate = char_count // 3

        print(f"┌─ Chunk {i+1:02d}/{total} {'─'*45}")
        print(f"│  字元數：{char_count:,}　　估計 token：{token_estimate:,}")
        print(f"│  {'─'*52}")

        # 印出開頭（前 200 字元）
        head = text[:200].replace("\n", "↵")
        print(f"│  【開頭】{head}")

        # 如果 chunk 夠長，也印出結尾（後 200 字元）
        if char_count > 400:
            tail = text[-200:].replace("\n", "↵")
            print(f"│  【結尾】...{tail}")

        # 警告：開頭或結尾是不完整的句子（沒有標點結束）
        first_line = text.strip().split("\n")[0]
        last_line = text.strip().split("\n")[-1]
        incomplete_end_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        if last_line and last_line[-1] in incomplete_end_chars:
            print(f"│  ⚠️  【結尾疑似截斷】最後字元：'{last_line[-1]}'")

        print(f"└{'─'*54}\n")

    # 額外：印出相鄰兩個 chunk 的 overlap 區域，確認重疊正確
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

        # 計算實際重疊字元數
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


# ── 主程式 ─────────────────────────────────────────────

# --test-chunks 模式：只做 chunk 檢查，不執行問答
if "--test-chunks" in sys.argv:
    pdf_files_all = sorted([f for f in os.listdir(papers_dir) if f.endswith(".pdf")])
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
                pass

    inspect_chunks(os.path.join(papers_dir, target_pdf), num_chunks=num_to_show)
    sys.exit(0)

questions = [
    "請幫我再仔細分析1-s2.0-S1878029613002417-main這篇paper，並跟我詳細講解這篇的合成步驟ZVI為何?是否有秤取藥物?若有，各秤取多少重量?使用了哪些reagent?請務必詳盡列出。",
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

    # print(f"\n回答：\n{response}")
    print(f"\n⏱ 本題耗時：{minutes}分{seconds}秒")

total_elapsed = time.time() - total_start
total_min = int(total_elapsed // 60)
total_sec = int(total_elapsed % 60)

print(f"\n{'='*65}")
print(f"✓ 完成！總耗時：{total_min}分{total_sec}秒")
print(f"{'='*65}")