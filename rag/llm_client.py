# rag/llm_client.py
# 負責建立 LLM 與 Embedding 的連線物件
# 其他模組需要用到 LLM 時，從這裡 import，不要重複初始化
#
# 模型分工：
#   Settings.llm   — gemma4:31b，主推理 LLM
#                    用於 Stage 2 Phase B（子問題回答）、Stage 4（最終綜合）
#                    context_window=32768（LlamaIndex prompt 管理用）
#                    ⚠️  Ollama 實際 num_ctx 取決於模型 Modelfile 設定
#
#   planning_llm   — qwen2.5:14b，規劃用小模型
#                    用於 Stage 1（論文篩選 + 子問題拆解）
#                    只負責 JSON 格式化輸出，不做深度推理
#                    context_window=16384，timeout=300s
#
#   Settings.embed — bge-m3，多語言向量嵌入模型
#                    用於 Stage 2 Phase A（並行向量檢索）
#
# Stage 3 / Stage 5 的 num_ctx 由各自模組直接帶入 Ollama API 參數：
#   Stage 3（knowledge_synthesizer.py）：STAGE3_NUM_CTX=16384
#   Stage 5（answer_verifier.py）       ：STAGE5_NUM_CTX=65536
#   兩者皆直連 Ollama，不走 LlamaIndex Settings.llm

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
    只需在程式啟動時呼叫一次（main.py 的 module-level 呼叫）。
    """
    global planning_llm

    http_client = httpx.Client(
        timeout=httpx.Timeout(cfg.LLM_TIMEOUT, connect=30.0)
    )

    # 主推理 LLM：gemma4:31b
    # Stage 2 Phase B（子問題回答）、Stage 4（最終綜合）使用
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

    # 規劃用小模型：qwen2.5:14b
    # Stage 1（論文篩選 + 子問題拆解）使用，只做 JSON 格式化，不做深度推理
    planning_llm = OpenAILike(
        model=cfg.PLANNING_LLM_MODEL,
        api_base=f"{cfg.OLLAMA_BASE_URL}/v1",
        api_key=cfg.OLLAMA_API_KEY,
        is_chat_model=True,
        timeout=300.0,           # 規劃任務不需要長 timeout
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