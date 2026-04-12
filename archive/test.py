from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings

# 設定使用本地Ollama模型
Settings.llm = Ollama(
    model="deepseek-r1:32b", 
    request_timeout=1200.0
)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# 讀取papers資料夾裡的所有PDF
print("正在讀取PDF...")
documents = SimpleDirectoryReader("papers").load_data()
print(f"讀取完成！共載入 {len(documents)} 個文件片段")

# 建立向量索引
print("正在建立索引，請稍候...")
index = VectorStoreIndex.from_documents(documents)
print("索引建立完成！")

# 建立查詢引擎
query_engine = index.as_query_engine(similarity_top_k=5)

# 開始問問題
question = """請根據提供的文件內容，詳細回答以下問題，並用繁體中文回答：
1. 這篇paper的主要研究目的是什麼？
2. 研究方法有哪些？
3. 有哪些重要的實驗結果和數據？
4. 研究結論是什麼？
"""
print(f"\n問題：{question}")
print("正在查詢，請稍候...")
response = query_engine.query(question)
print(f"\n回答：{response}")