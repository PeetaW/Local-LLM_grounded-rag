# rag/memory.py
# 負責 ChromaDB 跨 session 記憶的讀寫
# 三層記憶架構：
#   episodic_memory   → 推論結論、跨文獻比較結果
#   preference_memory → 使用者偏好、思維風格、研究方向

import logging
import uuid
from datetime import date
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
import re

import config as cfg

logger = logging.getLogger(__name__)

# ── 使用者偏好觸發詞 ──────────────────────────────────
# 規則：必須同時含有「偏好意圖詞」+ 至少一個「指令動詞」才觸發
# 避免「我在研究這個主題」這類正常問句被誤判為偏好設定
_PREFERENCE_INTENT = [
    "我比較喜歡", "我偏好", "我希望你", "請你以後",
    "我傾向", "我習慣",
    "i prefer", "please always",
]
_PREFERENCE_ACTION = [
    "回答", "輸出", "顯示", "使用", "以後", "之後", "都要", "記住",
    "respond", "output", "always", "remember", "format",
]
_PREFERENCE_INTENT_RE = re.compile(
    '|'.join(_PREFERENCE_INTENT), re.IGNORECASE
)
_PREFERENCE_ACTION_RE = re.compile(
    '|'.join(_PREFERENCE_ACTION), re.IGNORECASE
)
# 保留舊的 pattern 供 decide_and_save 中的寬鬆偵測使用
_PREFERENCE_PATTERN = _PREFERENCE_INTENT_RE


def _check_is_preference(text: str) -> bool:
    """
    回傳 True 代表這是偏好設定，不應該進 RAG pipeline。
    需同時包含「偏好意圖詞」與「指令動詞」，避免正常問句誤觸發。
    """
    return bool(_PREFERENCE_INTENT_RE.search(text) and _PREFERENCE_ACTION_RE.search(text))


def init_memory():
    """
    初始化 ChromaDB，建立三個 collection。
    回傳 (episodic_collection, preference_collection)。
    第一次執行自動建立，之後自動讀取。
    """
    chroma_client = chromadb.PersistentClient(path=cfg.MEMORY_DB_DIR)

    ollama_ef = OllamaEmbeddingFunction(
        url=f"{cfg.OLLAMA_BASE_URL}/api/embeddings",
        model_name=cfg.EMBED_MODEL,
    )

    episodic_collection = chroma_client.get_or_create_collection(
        name=f"{cfg.MEMORY_COLLECTION_EPISODIC}_{cfg.ACTIVE_PROJECT}",
        metadata={"description": "推論結論與跨文獻比較結果"},
        embedding_function=ollama_ef,
    )

    preference_collection = chroma_client.get_or_create_collection(
        name=cfg.MEMORY_COLLECTION_PREFERENCE,
        metadata={"description": "使用者偏好、思維風格、研究方向"},
        embedding_function=ollama_ef,
    )

    print(f"✓ 記憶系統啟動（ChromaDB）")
    print(f"  episodic：{episodic_collection.count()} 筆")
    print(f"  preference：{preference_collection.count()} 筆")

    return episodic_collection, preference_collection


def save_memory(collection, question: str, answer: str, memory_type: str = "episodic"):
    """
    把問答存入指定的 collection。
    memory_type 用於 metadata 標記，方便之後查詢時過濾。
    """
    collection.add(
        documents=[f"問：{question}\n答：{answer}"],
        metadatas=[{
            "date": str(date.today()),
            "question": question[:200],
            "type": memory_type,
        }],
        ids=[str(uuid.uuid4())]
    )


def _save_preference(preference_collection, observation: str):
    """
    儲存使用者偏好觀察。
    observation 是模型偵測到的偏好描述，例如：
    「使用者偏好條列式回答」「使用者研究方向偏重環境應用」
    """
    preference_collection.add(
        documents=[observation],
        metadatas=[{
            "date": str(date.today()),
            "type": "preference",
        }],
        ids=[str(uuid.uuid4())]
    )


def recall_memories(collection, question: str) -> str:
    """
    用語意搜尋找出過去最相關的 MEMORY_RECALL_N 筆記憶。
    episodic 和 preference 都用同一個函數搜尋，傳不同 collection 進來就好。
    """
    if collection.count() == 0:
        return ""

    try:
        results = collection.query(
            query_texts=[question],
            n_results=min(cfg.MEMORY_RECALL_N, collection.count()),
        )

        documents = results.get("documents", [[]])[0]
        if not documents:
            return ""

        context = "\n".join(f"- {doc}" for doc in documents)
        return context

    except Exception as e:
        print(f"  ⚠️  記憶搜尋失敗（不影響主流程）：{e}")
        return ""


def decide_and_save(
    question: str,
    answer: str,
    grounding_score: float,
    is_speculation: bool,
    is_multi_paper: bool,
    episodic_collection,
    preference_collection,
):
    """
    根據三個維度決定存進哪個 collection。

    判斷邏輯：
    score >= 0.9 + 無推測 + 無多文獻 → 純引用，不存
    score >= 0.7 + 有推測 或 有多文獻 → 跨文獻比較，存 episodic
    0.3 <= score < 0.7 + 有推測     → 發散推論，存 episodic
    score < 0.3 + 有推測            → 純推論，存 episodic
    score < 0.3 + 無推測            → 只留 session，不存
    score < 0（解析失敗）            → 分情況處理
    """
    # ── 解析失敗的情況 ────────────────────────────────
    if grounding_score < 0:
        # 偵測是否為使用者偏好表達
        # 例如：「我比較喜歡...」「請你以後...」「我傾向於...」
        if _PREFERENCE_PATTERN.search(question) or _PREFERENCE_PATTERN.search(answer):
            _save_preference(preference_collection, f"問：{question}\n觀察：{answer[:300]}")
            print(f"  💡 偵測到使用者偏好，存入 preference_memory")
        else:
            # 閒聊或 pipeline 出錯，不存
            print(f"  ℹ️  grounding score 解析失敗，不寫入長期記憶")
        return

    # ── 純引用，不存 ──────────────────────────────────
    if grounding_score >= 0.9 and not is_speculation and not is_multi_paper:
        print(f"  ℹ️  純論文引用（score={grounding_score:.2f}），不寫入長期記憶")
        return

    # ── 跨文獻比較或高分推論，存 episodic ────────────
    if grounding_score >= 0.7 and (is_speculation or is_multi_paper):
        save_memory(episodic_collection, question, answer, "cross_paper")
        print(f"  ✅ 跨文獻比較（score={grounding_score:.2f}），存入 episodic_memory")
        return

    # ── 中分段發散推論，存 episodic ──────────────────
    if 0.3 <= grounding_score < 0.7 and is_speculation:
        save_memory(episodic_collection, question, answer, "inference")
        print(f"  ✅ 發散推論（score={grounding_score:.2f}），存入 episodic_memory")
        return

    # ── 低分純推論，存 episodic ───────────────────────
    if grounding_score < 0.3 and is_speculation:
        save_memory(episodic_collection, question, answer, "pure_inference")
        print(f"  ✅ 純推論（score={grounding_score:.2f}），存入 episodic_memory")
        return

    # ── 其餘情況，只留 session ────────────────────────
    print(f"  ℹ️  score={grounding_score:.2f}，不符合存入條件，只保留 session 記憶")