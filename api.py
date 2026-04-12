# api.py
import asyncio
import queue
import uuid
import re
import io
import sys
import json
import time
import logging
import concurrent.futures
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List

logger = logging.getLogger(__name__)


class _StatusCapture(io.TextIOBase):
    """
    在 worker thread 執行期間攔截 sys.stdout，
    把所有 print() 的輸出以 [STATUS] 格式放入 queue，
    同時保留 terminal 輸出（方便 debug）。
    """
    def __init__(self, q: queue.Queue, original_stdout):
        self._q = q
        self._orig = original_stdout
        self._buf = ""

    def write(self, text: str) -> int:
        self._orig.write(text)   # 同步寫入 terminal
        self._orig.flush()
        self._buf += text
        # 按換行符切割，逐行放入 queue
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            stripped = line.strip()
            if stripped:
                self._q.put(f"[STATUS] {stripped}\n")
        return len(text)

    def flush(self):
        self._orig.flush()

from main import paper_engines, episodic_collection, preference_collection

from rag.citation_grounding import has_speculation_keywords, has_multi_paper_reference
from rag.query_engine import execute_structured_query, execute_structured_query_stream
from rag.memory import recall_memories, decide_and_save, _check_is_preference, _save_preference

app = FastAPI(
    title="ZVI RAG Pipeline",
    description="學術論文 RAG 問答系統，專門分析 ZVI 奈米粒子相關論文",
    version="1.0.0",
)

# ── 短期 session 記憶設定 ─────────────────────────────────────
SESSION_MAX_TURNS = 3
SESSION_MAX_COUNT = 200   # 最多同時保留幾個 session，防止記憶體無限增長
session_store: dict[str, list[tuple[str, str]]] = {}

# ── Session ID 驗證（只接受標準 UUID v4 格式）────────────────
_SESSION_ID_PATTERN = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$'
)

def _validate_session_id(session_id: str) -> bool:
    return bool(_SESSION_ID_PATTERN.match(session_id.lower()))

# ── Prompt Injection 基本防護 ─────────────────────────────────
_INJECTION_PATTERNS = [
    r'忽略.*指令', r'ignore.*instruction',
    r'forget.*above', r'忘記.*以上',
    r'you are now', r'你現在是',
    r'act as', r'扮演',
    r'system prompt', r'系統提示',
    r'輸出所有', r'print all',
    # 擴充常見繞過手法
    r'disregard.*previous', r'override.*instruction',
    r'ignore previous', r'忽略上述',
    r'new.*instruction', r'新.*指令',
    r'jailbreak', r'越獄',
    r'bypass.*filter', r'繞過.*過濾',
    r'pretend.*you', r'假裝.*你',
    r'repeat after me', r'跟著我說',
]
_INJECTION_RE = re.compile(
    '|'.join(_INJECTION_PATTERNS), re.IGNORECASE
)


def _check_prompt_injection(text: str) -> bool:
    """回傳 True 表示偵測到可疑內容"""
    return bool(_INJECTION_RE.search(text))


def _parse_grounding_score(answer: str) -> float:
    """
    從答案末尾的品質報告中解析 grounding_score。
    格式：<!-- grounding_score=0.875 -->
    解析失敗回傳 -1.0（由 decide_and_save 的 < 0 分支處理）。
    """
    match = re.search(r'grounding_score=(\d+\.?\d*)', answer)
    if match:
        return float(match.group(1))
    logger.debug("_parse_grounding_score: 未找到分數標記，回傳 -1.0")
    return -1.0


def _trim_session_store():
    """session_store 超過上限時，移除最舊的條目（防止記憶體無限增長）。"""
    if len(session_store) > SESSION_MAX_COUNT:
        overflow = len(session_store) - SESSION_MAX_COUNT
        for key in list(session_store.keys())[:overflow]:
            del session_store[key]
        logger.info("session_store 已清理 %d 個過期 session", overflow)


def _resolve_session_id(raw_sid: Optional[str]) -> str:
    """
    解析並驗證 session_id。
    過濾掉 Open WebUI 傳來的 placeholder 字串（如 <Your Session ID>）。
    無效或缺少時自動產生新的 UUID。
    """
    if not raw_sid:
        return str(uuid.uuid4())
    # 過濾 placeholder 字串
    stripped = raw_sid.strip("<>").lower()
    if stripped in ("your session id", "session id", ""):
        return str(uuid.uuid4())
    if _validate_session_id(raw_sid):
        return raw_sid.lower()
    print(f"  ⚠️  非法 session_id 被拒絕：{raw_sid[:50]}")
    return str(uuid.uuid4())


# ── Pydantic 資料模型 ─────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    session_id: str


# ── OpenAI 相容格式的資料模型 ─────────────────────────────────
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "rag-pipeline"
    messages: List[ChatMessage]
    stream: Optional[bool] = True
    session_id: Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "papers_loaded": list(paper_engines.keys()),
        "active_sessions": len(session_store),
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    查詢與化學合成或是生技、醫學、藥理、生物、癌症、動物實驗等相關學術論文。
    當使用者詢問以下內容時呼叫此工具：
    - 論文中的合成步驟、實驗方法
    - 試劑名稱與用量
    - 實驗條件（溫度、攪拌速度、pH值等）
    - 論文結果與數據分析
    - 跨論文比較
    重要：收到此工具的回傳結果後，請直接完整輸出，不要改寫或摘要。
    """
    question = request.question

    if _check_prompt_injection(question):
        print(f"  🚨 疑似 prompt injection，拒絕請求：{question[:100]}")
        return QueryResponse(
            answer="⚠️ 偵測到不合法的輸入內容，請重新提問。",
            session_id=str(uuid.uuid4()),
        )

    session_id = _resolve_session_id(request.session_id)

    short_term = session_store.get(session_id, [])
    short_term_context = ""
    if short_term:
        lines = []
        for i, (q, a) in enumerate(short_term, 1):
            lines.append(f"[本session第{i}輪]\n問：{q}\n答：{a[:800]}")
        short_term_context = "\n\n".join(lines)

    if _check_is_preference(question):
        _save_preference(preference_collection, question)
        print(f"  💡 偏好設定，跳過 RAG：{question[:50]}")
        return QueryResponse(
            answer="✅ 已記住你的偏好，之後的回答會參考這個設定。",
            session_id=session_id,
        )

    episodic_context = recall_memories(episodic_collection, question)
    preference_context = recall_memories(preference_collection, question)

    memory_context = ""
    if short_term_context:
        memory_context += f"\n【本次對話的歷史記錄（短期記憶）】\n{short_term_context}\n"
    if episodic_context:
        memory_context += f"\n【過去推論結論】\n{episodic_context}\n"
    if preference_context:
        memory_context += f"\n【使用者偏好】\n{preference_context}\n"

    answer = await asyncio.get_running_loop().run_in_executor(
        None,
        lambda: execute_structured_query(question, paper_engines, memory_context)
    )

    if session_id not in session_store:
        session_store[session_id] = []
    session_store[session_id].append((question, answer))
    if len(session_store[session_id]) > SESSION_MAX_TURNS:
        session_store[session_id] = session_store[session_id][-SESSION_MAX_TURNS:]
    _trim_session_store()

    grounding_score = _parse_grounding_score(answer)
    is_speculation = has_speculation_keywords(answer)
    is_multi_paper = (
        has_multi_paper_reference(answer) and
        has_multi_paper_reference(question)
    )

    decide_and_save(
        question, answer,
        grounding_score, is_speculation, is_multi_paper,
        episodic_collection, preference_collection,
    )

    final_answer = (
        "【以下為 RAG pipeline 直接輸出，內容來自論文原文，請勿改寫】\n\n"
        + answer
        + f"\n\n[session_id: {session_id}]"
        + "\n\n【RAG 輸出結束】"
    )

    return QueryResponse(answer=final_answer, session_id=session_id)


# ── OpenAI 相容：列出可用模型 ─────────────────────────────────
@app.get("/v1/models")
def list_models():
    """Open WebUI 連線時會先呼叫這個確認模型清單"""
    return {
        "object": "list",
        "data": [
            {
                "id": "rag-pipeline",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ],
    }


# ── OpenAI 相容：Chat Completions（支援 streaming）────────────
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI Chat Completions 相容 endpoint。
    Open WebUI 將此 pipeline 當成自訂模型使用時呼叫此 endpoint。
    支援 stream=True（SSE 逐字輸出）和 stream=False（一次回傳）。

    [STATUS] 進度訊息會轉換成 Markdown blockquote（> 🔄 ...）顯示，
    與 LLM 實際輸出視覺上分開，且不寫入記憶或 grounding 審查。
    """
    # ── 從 messages 裡取出最後一則 user 訊息作為問題 ──────────
    question = ""
    for msg in reversed(request.messages):
        if msg.role == "user":
            question = msg.content
            break

    if not question:
        question = request.messages[-1].content if request.messages else ""

    # ── 從 messages[] 提取對話歷史（OpenWebUI 已傳完整歷史）────
    # 倒數第二則 user 以前的 user/assistant 輪次就是歷史
    history_pairs: list[tuple[str, str]] = []
    msgs = request.messages[:-1]   # 去掉最後一則（即當前問題）
    i = 0
    while i < len(msgs):
        if msgs[i].role == "user":
            q_hist = msgs[i].content
            a_hist = msgs[i + 1].content if (i + 1 < len(msgs) and msgs[i + 1].role == "assistant") else ""
            if q_hist and a_hist:
                history_pairs.append((q_hist, a_hist))
            i += 2
        else:
            i += 1
    # 只取最近 SESSION_MAX_TURNS 輪
    history_pairs = history_pairs[-SESSION_MAX_TURNS:]

    # ── 空問題防護 ─────────────────────────────────────────────
    if not question.strip():
        error_text = "⚠️ 問題不可為空，請輸入問題後再送出。"
        if request.stream:
            async def _empty():
                chunk = {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{"index": 0, "delta": {"content": error_text}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(_empty(), media_type="text/event-stream")
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": error_text}, "finish_reason": "stop"}],
        }

    # ── Prompt injection 過濾 ──────────────────────────────────
    if _check_prompt_injection(question):
        print(f"  🚨 疑似 prompt injection，拒絕請求：{question[:100]}")
        error_text = "⚠️ 偵測到不合法的輸入內容，請重新提問。"
        if request.stream:
            async def _blocked():
                chunk = {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{"index": 0, "delta": {"content": error_text}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(_blocked(), media_type="text/event-stream")
        else:
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": error_text}, "finish_reason": "stop"}],
            }

    # ── Session ID ─────────────────────────────────────────────
    session_id = _resolve_session_id(request.session_id)

    # ── 短期記憶：優先用 messages[] 的歷史（OpenWebUI 直接模型模式）
    # 若 history_pairs 有內容（OpenWebUI 傳了歷史），直接用；
    # 否則 fallback 到 session_store（工具呼叫模式或舊客戶端）
    short_term_context = ""
    if history_pairs:
        lines = []
        for idx, (q, a) in enumerate(history_pairs, 1):
            lines.append(f"[本session第{idx}輪]\n問：{q}\n答：{a[:800]}")
        short_term_context = "\n\n".join(lines)
    else:
        short_term = session_store.get(session_id, [])
        if short_term:
            lines = []
            for idx, (q, a) in enumerate(short_term, 1):
                lines.append(f"[本session第{idx}輪]\n問：{q}\n答：{a[:800]}")
            short_term_context = "\n\n".join(lines)

    # ── 偏好設定直接回覆 ───────────────────────────────────────
    if _check_is_preference(question):
        _save_preference(preference_collection, question)
        pref_text = "✅ 已記住你的偏好，之後的回答會參考這個設定。"
        if request.stream:
            async def _pref():
                chunk = {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{"index": 0, "delta": {"content": pref_text}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(_pref(), media_type="text/event-stream")
        else:
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": pref_text}, "finish_reason": "stop"}],
            }

    # ── 長期記憶召回 ───────────────────────────────────────────
    episodic_context = recall_memories(episodic_collection, question)
    preference_context = recall_memories(preference_collection, question)

    memory_context = ""
    if short_term_context:
        memory_context += f"\n【本次對話的歷史記錄（短期記憶）】\n{short_term_context}\n"
    if episodic_context:
        memory_context += f"\n【過去推論結論】\n{episodic_context}\n"
    if preference_context:
        memory_context += f"\n【使用者偏好】\n{preference_context}\n"

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

    # ══════════════════════════════════════════════════════════
    # Streaming 模式
    # ══════════════════════════════════════════════════════════
    if request.stream:
        async def generate():
            # full_answer 只收集 LLM 實際輸出，不含 STATUS 進度訊息
            full_answer = ""
            # 使用 thread-safe 的 queue.Queue，避免 asyncio.Queue 跨執行緒的競態問題
            q_in: queue.Queue = queue.Queue()

            def _run():
                # 攔截 sys.stdout：把 pipeline 內部所有 print() 也轉成 [STATUS] 串流
                capture = _StatusCapture(q_in, sys.stdout)
                old_stdout = sys.stdout
                sys.stdout = capture
                try:
                    for chunk_text in execute_structured_query_stream(
                        question, paper_engines, memory_context
                    ):
                        q_in.put(chunk_text)
                finally:
                    sys.stdout = old_stdout
                    q_in.put(None)  # sentinel

            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            loop = asyncio.get_event_loop()
            loop.run_in_executor(executor, _run)

            while True:
                # 用 asyncio 非阻塞方式輪詢 queue，避免卡住 event loop
                chunk_text = await loop.run_in_executor(None, q_in.get)
                if chunk_text is None:
                    break

                # ── [STATUS] 進度訊息：轉成 blockquote，不寫入記憶 ──
                if chunk_text.startswith("[STATUS]"):
                    display_text = chunk_text.replace("[STATUS]", "> 🔄", 1)
                else:
                    # LLM 實際輸出：正常顯示，寫入記憶
                    display_text = chunk_text
                    full_answer += chunk_text

                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{"index": 0, "delta": {"content": display_text}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

            # ── 串流結束後寫入記憶 ────────────────────────────
            if session_id not in session_store:
                session_store[session_id] = []
            session_store[session_id].append((question, full_answer))
            if len(session_store[session_id]) > SESSION_MAX_TURNS:
                session_store[session_id] = session_store[session_id][-SESSION_MAX_TURNS:]
            _trim_session_store()

            grounding_score = _parse_grounding_score(full_answer)
            is_speculation = has_speculation_keywords(full_answer)
            is_multi_paper = (
                has_multi_paper_reference(full_answer) and
                has_multi_paper_reference(question)
            )
            decide_and_save(
                question, full_answer,
                grounding_score, is_speculation, is_multi_paper,
                episodic_collection, preference_collection,
            )

            # 結束 chunk
            end_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(end_chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            }
        )

    # ══════════════════════════════════════════════════════════
    # Non-streaming 模式（fallback）
    # ══════════════════════════════════════════════════════════
    else:
        answer = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: execute_structured_query(question, paper_engines, memory_context)
        )

        if session_id not in session_store:
            session_store[session_id] = []
        session_store[session_id].append((question, answer))
        if len(session_store[session_id]) > SESSION_MAX_TURNS:
            session_store[session_id] = session_store[session_id][-SESSION_MAX_TURNS:]
        _trim_session_store()

        grounding_score = _parse_grounding_score(answer)
        is_speculation = has_speculation_keywords(answer)
        is_multi_paper = (
            has_multi_paper_reference(answer) and
            has_multi_paper_reference(question)
        )
        decide_and_save(
            question, answer,
            grounding_score, is_speculation, is_multi_paper,
            episodic_collection, preference_collection,
        )

        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    """
    清除指定 session 的短期記憶。
    使用者開新對話時，前端可以呼叫這個 endpoint 主動清除。
    """
    if session_id in session_store:
        del session_store[session_id]
        return {"status": "cleared", "session_id": session_id}
    return {"status": "not_found", "session_id": session_id}


@app.get("/papers")
def list_papers():
    return {"papers": list(paper_engines.keys())}