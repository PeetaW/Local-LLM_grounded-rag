import os
import time
import httpx
# from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core import (
    VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage,
)
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata

# ── 連線設定 ───────────────────────────────────────────
http_client = httpx.Client(timeout=httpx.Timeout(14400.0, connect=30.0))

Settings.llm = OpenAILike(
    model="deepseek-r1:32b",
    api_base="http://localhost:8080/api/v1",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6ImFmYTlkZDc2LTNkYTctNGZiNy1iODFjLWYwNmM2MDgzMDc2YyIsImV4cCI6MTc3NDYyMDEyOSwianRpIjoiZmYzNzU3ZGYtYzQ1ZS00YWExLTg3MGUtODdiNDAxMTgzYjE0In0.NTQPKncbuvtN4--fBLVMuNmxRlOYtfeSHGU7GdGg22I",
    is_chat_model=True,
    timeout=14400.0,
    http_client=http_client,
    system_prompt=(
        "你是一個學術論文分析助手。"
        "請只根據提供的論文內容回答問題，使用繁體中文。"
        "如果論文中沒有相關資訊，請直接說明「此論文未涉及此議題」，"
        "不要自行推測或補充論文以外的內容。"
    ),
)

Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# ── chunk參數優化 ──────────────────────────────────────
Settings.chunk_size = 756
Settings.chunk_overlap = 200

# ── 每篇paper建立獨立索引 ──────────────────────────────
papers_dir = "papers"
pdf_files = sorted([f for f in os.listdir(papers_dir) if f.endswith(".pdf")])

print(f"找到 {len(pdf_files)} 篇論文，開始建立索引...\n")
query_engine_tools = []

for pdf_file in pdf_files:
    print(f"  處理中：{pdf_file}")
    docs = SimpleDirectoryReader(
        input_files=[os.path.join(papers_dir, pdf_file)]
    ).load_data()

    index = VectorStoreIndex.from_documents(docs)

    # compact模式：速度比refine快3～5倍
    engine = index.as_query_engine(
        similarity_top_k=8,
        response_mode="compact",
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
    "如果實驗中ZVI顆粒團聚嚴重，可能的原因是什麼？有哪些策略可以改善？請用繁體中文回答",
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