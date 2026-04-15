# rag/llm_client.py
# 負責建立 LLM 與 Embedding 的連線物件
# 其他模組需要用到 LLM 時，從這裡 import，不要重複初始化

import httpx
from llama_index.core import Settings
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.ollama import OllamaEmbedding

import config as cfg

# 全域變數，供其他模組 import 使用
planning_llm = None

def init_llm_and_embedding():
    """
    初始化 LLM 與 Embedding，並寫入 LlamaIndex 全域 Settings。
    只需在程式啟動時呼叫一次。
    """
    global planning_llm

    http_client = httpx.Client(
        timeout=httpx.Timeout(cfg.LLM_TIMEOUT, connect=30.0)
    )

    #主推理LLM: Deepseek-r1:32b
    Settings.llm = OpenAILike(
        model=cfg.LLM_MODEL,
        api_base=f"{cfg.OLLAMA_BASE_URL}/v1",
        api_key=cfg.OLLAMA_API_KEY,
        is_chat_model=True,
        timeout=cfg.LLM_TIMEOUT,
        http_client=http_client,
        context_window=cfg.LLM_CONTEXT_WINDOW,
        system_prompt=cfg.LLM_SYSTEM_PROMPT,
    )

    # 規劃用小模型：qwen2.5:7b（只負責 JSON 格式化輸出）
    planning_llm = OpenAILike(
        model=cfg.PLANNING_LLM_MODEL,
        api_base=f"{cfg.OLLAMA_BASE_URL}/v1",
        api_key=cfg.OLLAMA_API_KEY,
        is_chat_model=True,
        timeout=300.0,           # 小模型用短 timeout 就夠
        http_client=http_client,
        context_window=16384,
    )

    Settings.embed_model = OllamaEmbedding(
        model_name=cfg.EMBED_MODEL,
        base_url=cfg.OLLAMA_BASE_URL,
    )

    Settings.chunk_size = cfg.CHUNK_SIZE
    Settings.chunk_overlap = cfg.CHUNK_OVERLAP

    print(f"✓ LLM 初始化完成：{cfg.LLM_MODEL}")
    print(f"✓ 規劃模型初始化完成：{cfg.PLANNING_LLM_MODEL}")
    print(f"✓ Embedding 初始化完成：{cfg.EMBED_MODEL}")