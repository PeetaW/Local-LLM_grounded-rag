import os
import httpx
import time
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata

http_client = httpx.Client(timeout=httpx.Timeout(14400.0, connect=30.0))

Settings.llm = OpenAILike(
    model="deepseek-r1:32b",
    api_base="http://localhost:8080/api/v1",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6ImFmYTlkZDc2LTNkYTctNGZiNy1iODFjLWYwNmM2MDgzMDc2YyIsImV4cCI6MTc3NDYyMDEyOSwianRpIjoiZmYzNzU3ZGYtYzQ1ZS00YWExLTg3MGUtODdiNDAxMTgzYjE0In0.NTQPKncbuvtN4--fBLVMuNmxRlOYtfeSHGU7GdGg22I",
    is_chat_model=True,
    timeout=14400.0,
    http_client=http_client,
)

Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

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
    engine = index.as_query_engine(similarity_top_k=5)

    tool = QueryEngineTool(
        query_engine=engine,
        metadata=ToolMetadata(
            name=pdf_file.replace(".pdf", "").replace(" ", "_").replace("-", "_"),
            description=f"包含論文 {pdf_file} 的完整內容，涵蓋合成方法、實驗結果與討論"
        )
    )
    query_engine_tools.append(tool)
    print(f"  ✓ 完成：{pdf_file}")

print("\n建立SubQuestion綜合查詢引擎...")
sub_query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools,
    verbose=True,  # 顯示子問題拆解過程，方便觀察
    use_async=False, # 把問題逐一處理
)
print("✓ 引擎建立完成！\n")

# ── 8個測試問題 ────────────────────────────────────────
questions = [
    "請摘要每篇論文的核心合成方法與主要發現，請用繁體中文回答",
    "ZVI奈米粒子的合成步驟中，哪些參數被作者認為最關鍵？請用繁體中文回答",
    "比較各篇論文中ZVI或類似材料的合成方法，它們在反應條件（溫度、pH、還原劑）上有何異同？請用繁體中文回答",
    "各篇論文如何評估材料的去汙/降解效能？使用了哪些不同的評估指標？請用繁體中文回答",
    "哪篇論文的材料在環境應用上最具實際可行性？請從成本、穩定性、效能三個面向比較，請用繁體中文回答",
    "如果要合成小尺寸ZVI@pectin，有哪些可能的參數需要注意或調整？請用繁體中文回答",
    "根據這些論文的結果，如果實驗中ZVI顆粒團聚嚴重，可能的原因是什麼？有哪些策略可以改善？請用繁體中文回答",
    "如果要將這些材料的合成scale up到工業規模，你預測會遇到哪些挑戰？請用繁體中文回答",
]

# ── 逐題查詢 + 計時 + 存檔 ────────────────────────────
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
    print(f"\n⏱ 本題耗時：{minutes}分{seconds}秒")
    print(f"\n回答：\n{response}")
    
total_elapsed = time.time() - total_start
total_min = int(total_elapsed // 60)
total_sec = int(total_elapsed % 60)
print(f"\n{'='*65}")
print(f"✓ 所有問題完成！總耗時：{total_min}分{total_sec}秒")
print(f"{'='*65}")

# ── 顯示結果 ───────────────────────────────────────────
print(f"\n✓ 所有問題已完成！總耗時：{total_min}分{total_sec}秒")