import os
import time
import httpx
import json
import fitz #PyMuPDF
from llama_index.core import (
    VectorStoreIndex,  
    Settings, StorageContext, 
    load_index_from_storage,
)
from llama_index.core.schema import Document
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata

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

Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

Settings.chunk_size = 1024
Settings.chunk_overlap = 200 

# 儲存當前chunk設定
config = {
    "chunk_size": 1024,
    "chunk_overlap": 200,
    "embed_model": "nomic-embed-text",
    "parser": "pymupdf" #紀錄使用的解析器
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

# ── PyMuPDF解析PDF ──────────────────────────────
def load_pdf_with_pymupdf(pdf_path):
    """
    用PyMuPDF解析PDF，抽取文字內容
    比pypdf更準確，能處理更複雜的PDF版面
    """
    doc = fitz.open(pdf_path)
    full_text = ""
    
    for page_num, page in enumerate(doc, 1):
        # 抽取文字
        text = page.get_text("text")
        if text.strip():
            full_text += f"\n[第{page_num}頁]\n{text}"
    
    doc.close()
    
    # 回傳LlamaIndex的Document物件
    return [Document(
        text=full_text,
        metadata={
            "file_name": os.path.basename(pdf_path),
            "file_path": pdf_path,
            "parser": "pymupdf"
        }
    )]


# ── 每篇paper建立獨立索引 ──────────────────────────────
papers_dir = "papers"
pdf_files = sorted([f for f in os.listdir(papers_dir) if f.endswith(".pdf")])

print(f"找到 {len(pdf_files)} 篇論文，開始建立索引...\n")

query_engine_tools = []

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

    # compact模式：速度比refine快3～5倍
    engine = index.as_query_engine(
        similarity_top_k=8,
        # response_mode="compact",
    )

    tool = QueryEngineTool(
        query_engine=engine,
        metadata=ToolMetadata(
            name=pdf_file.replace(".pdf", "").replace(" ", "_").replace("-", "_"),
            description=f"包含論文 {pdf_file} 的完整內容，涵蓋合成方法、實驗結果與討論",
        ),
    )
    query_engine_tools.append(tool)
    print(f"  ✓ 完成：{pdf_file}")

print("\n建立SubQuestion綜合查詢引擎...")
sub_query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools,
    verbose=True,
    use_async=False,
)
print("✓ 引擎建立完成！\n")

# ── 測試題目（先跑2題驗證效果）─────────────────────────
questions = [
    "請摘要每篇論文的核心合成方法與主要發現，請用繁體中文回答",
    # "如果實驗中ZVI顆粒團聚嚴重，可能的原因是什麼？有哪些策略可以改善？請用繁體中文回答",
]

# ── 逐題查詢 + 計時 ────────────────────────────────────
total_start = time.time()

for i, question in enumerate(questions, 1):
    print(f"\n{'='*65}")
    print(f"[問題 {i}/{len(questions)}] {question}")
    print(f"{'='*65}")

    q_start = time.time()
    response = sub_query_engine.query(question)
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