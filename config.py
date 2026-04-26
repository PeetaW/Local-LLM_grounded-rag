# config.py
# ══════════════════════════════════════════════════════
# 所有可調整的參數集中在這裡
# 修改任何 INDEX_* 類型的參數後，需刪除 index_storage/ 重新執行
# ══════════════════════════════════════════════════════

# ── Project 設定 ───────────────────────────────────────
ACTIVE_PROJECT = "boron_bnct"   # ← 切換 project 改這裡
PROJECTS_BASE_DIR = "projects"


# ── 路徑設定 ──────────────────────────────────────────
# NOTE: 以下路徑在 module import 時就計算完成（module-level 靜態值）。
# ACTIVE_PROJECT 切換需重啟 server 才會生效，不支援 runtime 動態切換。
# 若未來需要 runtime 切換 project，路徑應改為函數：get_papers_dir(project_name)。
PAPERS_DIR        = f"{PROJECTS_BASE_DIR}/{ACTIVE_PROJECT}/papers"
INDEX_BASE_DIR    = f"{PROJECTS_BASE_DIR}/{ACTIVE_PROJECT}/index_storage"
INDEX_CONFIG_PATH = f"{PROJECTS_BASE_DIR}/{ACTIVE_PROJECT}/index_storage/config.json"
VL_OUTPUT_DIR     = f"{PROJECTS_BASE_DIR}/{ACTIVE_PROJECT}/vl_test_output"
METADATA_PATH     = f"{PROJECTS_BASE_DIR}/{ACTIVE_PROJECT}/papers_metadata.json"
MEMORY_DB_DIR     = "./memory_db"   # 共用，不跟著 project 走

# ── Ollama 連線 ───────────────────────────────────────
OLLAMA_BASE_URL   = "http://localhost:11434"
OLLAMA_API_KEY    = "ollama"   # Ollama 不需要真實 key，固定填這個

# ── LLM 設定 ──────────────────────────────────────────
LLM_MODEL         = "gemma4:31b"
PLANNING_LLM_MODEL = "qwen2.5:14b"
VL_MODEL = "qwen3-vl:32b"
VL_AUTO_RUN = True
LLM_TIMEOUT       = 28800.0    # 8小時，避免長推理被中斷
LLM_CONTEXT_WINDOW = 32768

# ── Contextual Summarization 設定 ─────────────────────
CONTEXT_SUMMARY_ENABLED = True    # 是否啟用
CONTEXT_SUMMARY_MODEL   = LLM_MODEL  # 用 deepseek-r1:32b

# ── Citation Grounding + Answer Relevance Check 設定 ──
CITATION_GROUNDING_ENABLED = True   # 改成 False 可以關閉審查

# ── 推理模式設定 ──────────────────────────────────────
# "strict"    → 只引用論文原文，不做推論（高精確度，防幻覺）
# "reasoning" → 允許跨文獻推論與知識延伸，但必須標注認知層次
# 可在對話中動態切換，或在這裡設定預設值
REASONING_MODE = "reasoning"


LLM_SYSTEM_PROMPT = (
    "你是一個學術論文分析助手。"
    "請只根據提供的論文內容回答問題，使用繁體中文。"
    "請務必使用繁體中文，絕對不可使用簡體中文。"
    "如果論文中沒有相關資訊，請直接說明「此論文未涉及此議題」，"
    "不要自行推測或補充論文以外的內容。"
    "回答時請盡量引用論文中的具體數據、步驟與條件。"
)

# ── Embedding 設定 ────────────────────────────────────
EMBED_MODEL       = "bge-m3"

# ── Chunking 設定（修改後需刪除 index_storage/ 重建）──
CHUNK_SIZE        = 1024
CHUNK_OVERLAP     = 256

# ── 檢索設定 ──────────────────────────────────────────
SIMILARITY_TOP_K  = 8    # vector & BM25 各取幾個候選
GROUNDING_TOP_K   = 20   # Stage 6 NLI 用，比一般檢索多取以提升 grounding 覆蓋率

# ── Review 模式開關（測試用）─────────────────────────
# True  = 查詢所有論文，不篩選，專用於 mini-review 生成
# False = 正常模式，自動篩選最相關 5 篇（預設）
REVIEW_MODE = False

# ── Reranker 設定 ─────────────────────────────────────
RERANKER_MODEL    = "BAAI/bge-reranker-v2-m3"
RERANKER_TOP_N    = 8    # rerank 後保留幾個送進 LLM

# ── Stage 3：知識蒸餾 ─────────────────────────────────────
SYNTHESIS_ENABLED = True
SYNTHESIS_MODEL   = "gemma4:31b"    # 同時用於 Stage 3 和 Stage 4

# ── Stage 5：邏輯自洽驗證 ─────────────────────────────
VERIFY_ENABLED      = True
VERIFY_MODEL        = "qwen3.5:35b-a3b"
MAX_VERIFY_RETRIES  = 2             # fallback 最多重試幾次

# ── Stage 2：並行子查詢 ────────────────────────────────
SUBQUERY_MAX_WORKERS = 4   # 並行子查詢的 thread pool 大小

# ── 各 Stage num_ctx 設定 ──────────────────────────────
# Stage 1（qwen2.5:14b）與 Stage 4（gemma4:31b）透過 LlamaIndex 呼叫，
# 不支援 per-call num_ctx override，沿用 LLM_CONTEXT_WINDOW / planning_llm context_window。
STAGE3_NUM_CTX = 16384    # 知識蒸餾（knowledge_synthesizer.py）
STAGE5_NUM_CTX = 65536    # 邏輯驗證（answer_verifier.py）

# ── Plan-and-Execute 架構開關 ─────────────────────────
PLAN_EXECUTE_ENABLED = False       # 預設關閉，穩定後開啟

# ── NLI 擴展開關 ──────────────────────────────────────
NLI_CONTRADICTION_ENABLED = True   # 矛盾偵測（預設開啟）
NLI_DECOMPOSE_ENABLED = True       # 子命題拆解驗證
NLI_JOINT_VERIFY_ENABLED = True    # 多來源聯合驗證
# English-first pipeline：全流程用英文（Stage 4 輸出英文 → Stage 5 英文驗證 → NLI EN-vs-EN → 最後翻譯成繁體中文）
# 優點：NLI 從跨語言變單語言（大幅提升 entailment 準確度），Verifier 推論邏輯更穩定
# 注意：開啟後 NLI_TRANSLATE_TO_EN 會自動跳過（draft 已是英文，不需要再翻）
EN_DRAFT_PIPELINE = True          # True = 英文 draft 全流程（預設關閉）

# 跨語言補償：中文 hypothesis 批次翻譯成英文後再做 NLI
# EN_DRAFT_PIPELINE=True 時此設定自動無效（draft 本身已是英文）
# 背景：PDF chunks 為英文，中文 answer 做跨語言 NLI 時短句 entailment 分數偏低；
#       翻譯後改為 EN-vs-EN 可顯著提升數值型事實句的 entailment 準確度。
# 注意：開啟後每次 grounding check 增加一次 LLM 批次呼叫（約 5-20s）。
NLI_TRANSLATE_TO_EN = True        # 翻譯 hypothesis 為英文再送 NLI

# ── 記憶系統設定 ──────────────────────────────────────
MEMORY_RECALL_N   = 3    # 每次查詢召回幾筆歷史記憶

# ── ChromaDB Collection 名稱 ──────────────────────────
MEMORY_COLLECTION_EPISODIC   = "episodic_memory"
MEMORY_COLLECTION_PREFERENCE = "preference_memory"

# ══════════════════════════════════════════════════════
#    以下這個 dict 給 index_storage/config.json 比對用
#    不要手動修改，它會自動同步上方的值
# ══════════════════════════════════════════════════════
INDEX_BUILD_CONFIG = {
    "chunk_size":          CHUNK_SIZE,
    "chunk_overlap":       CHUNK_OVERLAP,
    "embed_model":         EMBED_MODEL,
    "parser":              "pymupdf",
    "include_vl":          True,
    "context_summary":     CONTEXT_SUMMARY_ENABLED,
}