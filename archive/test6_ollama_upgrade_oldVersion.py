"""
test6_openWebUI_upgrade.py
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
    5. 執行：python test5_openWebUI.py
    
    首次執行：自動建立索引並儲存至 index_storage/
    後續執行：直接載入既有索引，不重新建立
    
    ⚠️  若修改 chunk_size / chunk_overlap / embed_model / parser / include_vl 任一參數，
        需手動刪除 index_storage/ 資料夾後重新執行

作者備註：
    - LLM：deepseek-r1:32b，透過 Open WebUI API（port 8080）呼叫
    - Embedding：nomic-embed-text，透過 Ollama 直連（port 11434）
    - PDF解析器：PyMuPDF（fitz），取代原本的 pypdf，解析品質更佳
    - chunk_size=1024 / chunk_overlap=200，經測試後的穩定設定
    - similarity_top_k=8，每次檢索取最相關的8個chunk
    - SubQuestionQueryEngine：大問題自動拆解成子問題，每篇論文獨立查詢
    - timeout設定14400秒（4小時），避免長時間推理被中斷
    - ⚠️  此環境與 Open WebUI 共用同一虛擬環境（llm_env），
          套件版本牽一髮動全身，不要隨意 upgrade 任何套件！
    - 後續將再加入VL模型對圖片的理解輸出結果，以及模型答案驗證功能。
    - 考慮新增BM25，可能需要增加KV cache，要評估模型優化。
    -最後要封裝工具，寫fastAPI，接入OpenWebUI，以做歷史對話紀錄管理。

升級重點：
    1. Embedding：nomic-embed-text → bge-m3（中英混合更強）
    2. 檢索：純向量 → BM25 + 向量 Hybrid Search
    3. 查詢引擎：SubQuestionQueryEngine → StructuredPlanning
       （LLM 在記憶體內拆解子問題，不寫檔，結束即消失）
    4. 跨 session 記憶：memsearch（記住使用者問過的問題與結論）
    5. Context window：4096 → 32768
"""

import os
import time
import httpx
import json
import asyncio
import fitz #PyMuPDF
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
# ── chromadb（跨 session 記憶）────────────────────────
import chromadb

# ── 連線設定 ───────────────────────────────────────────
http_client = httpx.Client(timeout=httpx.Timeout(14400.0, connect=30.0))

Settings.llm = OpenAILike(
    model="deepseek-r1:32b",
    api_base="http://localhost:11434/v1",
    api_key="ollama",
    is_chat_model=True,
    timeout=14400.0,
    http_client=http_client,
    context_window=32768,
    additional_kwargs={"options": {"num_ctx": 32768}},
    system_prompt=(
        "你是一個學術論文分析助手。"
        "請只根據提供的論文內容回答問題，使用繁體中文。"
        "請務必使用繁體中文，絕對不可使用簡體中文。"
        "如果論文中沒有相關資訊，請直接說明「此論文未涉及此議題」，"
        "不要自行推測或補充論文以外的內容。"
        "回答時請盡量引用論文中的具體數據、步驟與條件。"
    ),
)

# ── bge-m3 embedding（取代 nomic-embed-text）──────────
# bge-m3 同時支援 dense + sparse，中英文混合效果更好
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
    "parser": "pymupdf", #紀錄使用的解析器
    "include_vl": True, #讓config偵測到有加入VL描述
}
INDEX_BASE_DIR= "index_storage"
config_path = "index_storage/config.json"

# 判斷設定有沒有改變
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        saved_config = json.load(f)
    if saved_config != config:
        # 設定改了 → 警告使用者
        print("⚠️  chunk參數已變更！")
        print(f"   舊設定：{saved_config}")
        print(f"   新設定：{config}")
        print("   請手動刪除 index_storage 資料夾後重新執行")
        exit()
    else:
        print("chunk設定與索引一致，直接載入索引")
else:
    # 第一次執行 → 建立資料夾並儲存設定
    os.makedirs(INDEX_BASE_DIR, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print("✓ 首次執行，已儲存chunk設定\n")

# ── PyMuPDF解析PDF + VL圖片描述融合 ─────────────
def load_pdf_with_pymupdf(pdf_path, vl_output_dir="vl_test_output"):
    """
    用PyMuPDF解析PDF，抽取文字內容
    同時載入對應的VL圖片描述（若存在）
    回傳多個 Document：1個文字 + N個圖片描述
    """
    pdf_filename = os.path.basename(pdf_path)
    paper_name = pdf_filename.replace(".pdf", "")

    # ── 1. 抽取PDF文字（過濾參考文獻頁）────────────────────
    doc = fitz.open(pdf_path)
    full_text = ""
    ref_section_started = False  # 一旦偵測到參考文獻區塊，後面全部跳過

    for page_num, page in enumerate(doc, 1):
        text = page.get_text("text")
        if not text.strip():
            continue

        # 偵測參考文獻區塊的起始：
        # 條件1：頁面包含 "References" 或 "參考文獻" 標題行
        # 條件2：同時該頁有大量 doi: 字樣（代表是純引用列表，不是正文引用）
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
        # 安全網：萬一過濾太激進導致全空，退回原始全文
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

    # ── 2. 載入VL圖片描述（新增）──────────────────────────
    vl_result_path = os.path.join(vl_output_dir, paper_name, "vl_test_result.json")

    if not os.path.exists(vl_result_path):
        print(f"  ℹ️  找不到VL描述：{paper_name}，僅使用PDF文字")
        return documents

    with open(vl_result_path, "r", encoding="utf-8") as f:
        vl_data = json.load(f)

    img_count = 0
    skipped_count = 0
    for img in vl_data.get("images", []):
        # 只載入成功且不需要人工確認的圖片
        if not img.get("success", False):
            continue
        if img.get("needs_review", False):
            skipped_count += 1
            continue

        description = img.get("description", "").strip()
        if not description:
            continue

        # 包裝成 Document，加上清楚的前綴讓模型知道這是圖片來源
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

# 建立每篇論文的 Hybrid 查詢引擎（向量 + BM25）
def build_hybrid_query_engine(index):
    """
    建立 Hybrid Retriever：向量搜尋 + BM25 並行，結果融合後回答
    向量：抓語意相關的 chunk
    BM25：抓精確術語相關的 chunk（如 Fe²⁺、NaBH₄ 數值）
    """
    # 向量 retriever
    vector_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=8,
    )

    # BM25 retriever（需要原始 nodes）
    # 若是載入既有 index，從 index 取出 nodes
    nodes = list(index.docstore.docs.values())
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=8,
    )

    # Fusion：把兩個 retriever 的結果合併，用 RRF 算法重新排序
    hybrid_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=8,        # 最終取 8 個 chunk 給 LLM
        num_queries=1,             # 不額外生成 query 變體，保持速度
        mode="reciprocal_rerank",  # RRF 融合算法
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
    # 每篇paper有自己的索引資料夾
    index_dir = os.path.join(INDEX_BASE_DIR, pdf_file.replace(".pdf", ""))
    
    if os.path.exists(index_dir):
        # ✅ 索引已存在 → 直接載入
        print(f"  載入既有索引：{pdf_file}")
        storage_context = StorageContext.from_defaults(persist_dir=index_dir)
        index = load_index_from_storage(storage_context)
    else:
        # 🔨 索引不存在 → 建立並儲存
        print(f"  建立新索引：{pdf_file}")
        # ← 用PyMuPDF取代SimpleDirectoryReader
        docs = load_pdf_with_pymupdf(os.path.join(papers_dir, pdf_file))
        index = VectorStoreIndex.from_documents(docs)
        # 存到硬碟
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

請將這個問題拆解成多個子問題，每個子問題針對一篇或所有論文。
以 JSON 陣列回傳，格式如下，只輸出 JSON，不要其他文字：
[
  {{"paper": "論文檔名（不含.pdf）或 ALL（代表所有論文）", "sub_q": "子問題內容"}},
  ...
]
"""
    # 直接呼叫 LLM，結果只存在變數裡
    response = Settings.llm.complete(prompt)
    
    try:
        # 清理回應，移除可能的 markdown 包裝
        raw = response.text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        sub_questions = json.loads(raw)
        return sub_questions
    except json.JSONDecodeError:
        # 如果 LLM 輸出格式不對，退回對所有論文問同一個問題
        print("  ⚠️  子問題拆解失敗，改為對所有論文問同一問題")
        return [{"paper": "ALL", "sub_q": question}]

# StructuredPlanning：LLM 在記憶體內拆解子問題
# 不寫任何檔案，session 結束即消失
def execute_structured_query(question: str) -> str:
    """
    StructuredPlanning 主流程：
    1. LLM 拆解子問題（記憶體內）
    2. 逐一查詢（線性，避免 VRAM 爆炸）
    3. 合併所有子答案，LLM 做最終綜合
    """
    paper_names = list(paper_engines.keys())
    
    # Step 1：拆解（不寫檔）
    print("\n  📋 拆解子問題中...")
    sub_questions = plan_sub_questions(question, paper_names)
    print(f"  → 拆出 {len(sub_questions)} 個子問題")

    # Step 2：逐一查詢
    sub_answers = []
    for i, sq in enumerate(sub_questions, 1):
        paper = sq.get("paper", "ALL")
        sub_q = sq.get("sub_q", "")
        print(f"\n  [{i}/{len(sub_questions)}] 查詢：{sub_q[:50]}...")

        if paper == "ALL":
            # 對所有論文查詢，收集各自的回答
            for name, engine in paper_engines.items():
                try:
                    result = engine.query(sub_q)
                    sub_answers.append(f"【{name}】\n{result}")
                except Exception as e:
                    sub_answers.append(f"【{name}】查詢失敗：{e}")
        else:
            # 找到對應論文
            engine = paper_engines.get(paper)
            if engine is None:
                # 名字不完全匹配，模糊搜尋
                matched = next(
                    (k for k in paper_engines if paper in k or k in paper),
                    None
                )
                engine = paper_engines.get(matched) if matched else None

            if engine:
                try:
                    result = engine.query(sub_q)
                    sub_answers.append(f"【{paper}】\n{result}")
                except Exception as e:
                    sub_answers.append(f"【{paper}】查詢失敗：{e}")
            else:
                sub_answers.append(f"【{paper}】找不到對應論文")

    # Step 3：LLM 綜合所有子答案
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
    final_response = Settings.llm.complete(synthesis_prompt)
    return final_response.text


# memsearch：跨 session 記憶系統
# 記住使用者問過的問題與重要結論
MEMORY_DIR = "./memory"

chroma_client = chromadb.PersistentClient(path="./memory_db")
memory_collection = chroma_client.get_or_create_collection(name="rag_memory")

def save_memory(question: str, answer: str):
    """把這次問答的摘要寫入今日記憶檔"""
    p = Path(MEMORY_DIR) / f"{date.today()}.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    
    # 只存摘要，不存完整答案（避免記憶檔過大）
    summary = answer[:500] + "..." if len(answer) > 500 else answer
    with open(p, "a", encoding="utf-8") as f:
        f.write(f"\n## 問：{question}\n{summary}\n")

async def recall_memories(question: str) -> str:
    """搜尋過去記憶，找出跟這個問題相關的歷史問答"""
    try:
        memories = await ms.search(question, top_k=3)
        if not memories:
            return ""
        context = "\n".join(
            f"- {m['content'][:200]}" for m in memories
        )
        return f"\n【相關歷史問答記憶】\n{context}\n"
    except Exception:
        return ""  # 記憶搜尋失敗不影響主流程


# ═══════════════════════════════════════════════════════
# 主查詢流程（整合記憶）
# ═══════════════════════════════════════════════════════
async def query_with_memory(question: str) -> str:
    # 1. 先搜尋過去記憶（非同步，不卡主流程）
    memory_context = await recall_memories(question)
    if memory_context:
        print(f"  💭 找到相關記憶，注入 context")
        # 把記憶注入問題 context（不修改原始問題）
        augmented_question = f"{memory_context}\n\n當前問題：{question}"
    else:
        augmented_question = question

    # 2. 執行 StructuredPlanning 查詢
    answer = execute_structured_query(augmented_question)

    # 3. 把這次問答存入記憶（背景執行）
    save_memory(question, answer)
    await ms.index()  # 更新 memsearch index

    return answer

# 主程式
questions = [
    "請摘要每篇論文的核心合成方法與主要發現，請用繁體中文回答",
]

# ── 逐題查詢 + 計時 ────────────────────────────────────
total_start = time.time()

for i, question in enumerate(questions, 1):
    print(f"\n{'='*65}")
    print(f"[問題 {i}/{len(questions)}] {question}")
    print(f"{'='*65}")

    q_start = time.time()
    # 用 asyncio 執行含記憶的查詢
    response = asyncio.run(query_with_memory(question))
    q_elapsed = time.time() - q_start

    minutes = int(q_elapsed // 60)
    seconds = int(q_elapsed % 60)

    print(f"\n回答：\n{response}")
    print(f"\n⏱ 本題耗時：{minutes}分{seconds}秒")

total_elapsed = time.time() - total_start
total_min = int(total_elapsed // 60)
total_sec = int(total_elapsed % 60)

print(f"\n{'='*65}")
print(f"✓ 完成！總耗時：{total_min}分{total_sec}秒")
print(f"{'='*65}")