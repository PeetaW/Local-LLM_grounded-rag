import os
import httpx
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.openai_like import OpenAILike
from openai import OpenAI

# 建立自訂timeout的httpx client
http_client = httpx.Client(timeout=httpx.Timeout(1200.0, connect=30.0))

# 透過Open WebUI API呼叫模型
Settings.llm = OpenAILike(
    model="deepseek-r1:32b",
    api_base="http://localhost:8080/api/v1",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6ImFmYTlkZDc2LTNkYTctNGZiNy1iODFjLWYwNmM2MDgzMDc2YyIsImV4cCI6MTc3NDYyMDEyOSwianRpIjoiZmYzNzU3ZGYtYzQ1ZS00YWExLTg3MGUtODdiNDAxMTgzYjE0In0.NTQPKncbuvtN4--fBLVMuNmxRlOYtfeSHGU7GdGg22I",
    is_chat_model=True,
    timeout=1200.0,
    http_client=http_client,
)

# Embedding還是直接用Ollama（輕量模型不影響VRAM）
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# 讀取PDF
print("正在讀取PDF...")
documents = SimpleDirectoryReader("papers").load_data()
print(f"讀取完成！共載入 {len(documents)} 個文件片段")

# 建立向量索引
print("正在建立索引，請稍候...")
index = VectorStoreIndex.from_documents(documents)
print("索引建立完成！")

# 建立查詢引擎
query_engine = index.as_query_engine(similarity_top_k=5)

# 問問題
question = "每篇paper的主要研究目的是什麼？有什麼重要發現？關於ZVI的機制跟原理，這幾篇文獻中有什麼共識跟分歧?請用繁體中文回答"
print(f"\n問題：{question}")
print("正在查詢，請稍候...")
response = query_engine.query(question)
print(f"\n回答：{response}")