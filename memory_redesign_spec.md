# Memory Layer Redesign Spec

## Background & Design Intent

The memory system is designed as a **research knowledge management layer**, not just a Q&A log.
The user has multiple identities and projects, switching between them frequently.
Memory should let them pick up where they left off without re-deriving past conclusions.

### Three core memory types

| Collection | Purpose | Scope | Write pattern |
|-----------|---------|-------|--------------|
| `episodic_memory_{project}` | Research conclusions & hypotheses | Per project | Append (with dedup) |
| `preference_memory` | Workflow & output preferences | Global (shared across projects) | Append |
| `work_state` | Current project progress snapshot | Per project | Upsert (overwrite) |

### Memory status lifecycle

```
provisional  →  confirmed      (hypothesis validated by evidence/experiment)
             →  invalidated    (disproven — never recalled again)
             →  superseded     (replaced by a newer understanding)
```

---

## Part 1: Schema Changes

### 1.1 Episodic memory — new metadata schema

```python
{
    "date":         "2026-04-29",
    "type":         "hypothesis" | "conclusion" | "cross_paper_synthesis",
    "status":       "provisional" | "confirmed" | "invalidated" | "superseded",
    "confidence":   0.6,            # float 0.0–1.0
    "papers":       "paper_A,paper_B",  # comma-separated, empty string if none
    "session_id":   "uuid",
    "superseded_by": "",            # ID of the replacing memory, or empty string
}
```

### 1.2 Document format — atomic conclusions (NOT full Q&A)

**Current (to be replaced):**
```
問：{question}
答：{full answer text, potentially 500 words}
```

**New format:**
```
{one-sentence conclusion}（來源：{papers}）
```

Examples:
```
BSH 在 pH 7.4 生理條件下的腫瘤攝取率優於 BPA（來源：paper_A, paper_B）
BNCT 療效與中子束能量分布高度相關，熱中子優於超熱中子（來源：paper_C）
```

At save time, run an LLM extraction step to distill 2–4 such sentences from the full answer.
Each sentence is saved as a **separate document** (one record per conclusion, not one per Q&A).

### 1.3 work_state document format

```
專案：{project}
當前焦點：{current_focus}
最後更新：{date}
進行中的問題：
  - {open_question_1}
  - {open_question_2}
下一步：
  - {next_step_1}
近期結論摘要：
  - {brief_conclusion_1}
```

Fixed document ID: `f"work_state_{project}"` — enables delete-then-add upsert.

### 1.4 New config.py constants

```python
MEMORY_COLLECTION_WORK_STATE        = "work_state"
MEMORY_CONFLICT_SIMILARITY_THRESHOLD = 0.85   # above this → run conflict check LLM call
MEMORY_DEDUP_SIMILARITY_THRESHOLD    = 0.92   # above this → skip save (near-duplicate)

# Trigger phrase lists (Chinese + English)
MEMORY_CONSOLIDATION_TRIGGERS = [
    "總結這次討論", "整理今天的討論", "結束討論", "session 結束",
    "summarize this session", "end of session",
]
MEMORY_CONFIRM_TRIGGERS = [
    "這個假說確認了", "確認這個結論", "這個推論是正確的", "驗證了",
    "confirm this hypothesis", "this is confirmed",
]
MEMORY_INVALIDATE_TRIGGERS = [
    "這個假說是錯的", "這個結論不對", "之前的理解有誤", "推翻了",
    "this hypothesis is wrong", "invalidate",
]
MEMORY_SUPERSEDE_TRIGGERS = [
    "修正一下之前的理解", "更新這個假說", "應該是", "不是之前說的",
    "update my previous understanding",
]
MEMORY_WORK_STATE_TRIGGERS = [
    "更新進度", "記錄進度", "目前進度是", "這個專案現在",
    "update progress", "record progress",
]
```

---

## Part 2: Three Mechanisms

### Mechanism C — Background Conflict Guard (implement first)

Runs silently every time a new conclusion is about to be saved.

```
New conclusion ready to save
        ↓
Query top-3 similar memories (status != "invalidated")
        ↓
Max similarity > MEMORY_CONFLICT_SIMILARITY_THRESHOLD?
   NO  → save normally
   YES → LLM call: "Does the new conclusion contradict, support, or supersede the old one?"
            "contradict" → append ⚠️ warning to response, do NOT auto-change status
            "support"    → save normally (confidence boost optional)
            "supersede"  → append ⚠️ warning to response, do NOT auto-change status
            "unrelated"  → save normally
```

Warning appended to response (after the main answer):
```
---
⚠️ **記憶衝突提示**
這個結論可能與你之前的假說有出入：
> 舊：「{old_conclusion}」（{old_date}，狀態：{old_status}）
若確認要更新，請說「更新這個假說」。
```

Similarity > MEMORY_DEDUP_SIMILARITY_THRESHOLD AND same direction → skip save entirely (silent).

### Mechanism A — Quick Trigger (implement second)

Detected in `api.py` before the RAG pipeline runs, similar to existing `_check_is_preference`.

**Detection function:** `_check_memory_action_trigger(text) -> str | None`
Returns: `"confirm"` | `"invalidate"` | `"supersede"` | `"work_state"` | `None`

**Flow for confirm / invalidate / supersede:**
1. Detect trigger type from user message
2. Query top-3 most similar episodic memories (status != "invalidated")
3. Skip RAG pipeline; return a confirmation prompt:
   ```
   🔍 你是指這筆記憶嗎？
   > 「{conclusion}」（{date}，{type}，狀態：{status}）
   
   回覆「是」確認，「否」顯示下一筆，「取消」放棄操作。
   ```
4. On confirmation → call `update_memory_status(id, new_status)`
5. Return: `"✅ 已將記憶標記為 {new_status}。"`

**Flow for work_state update:**
1. Detect trigger
2. Extract progress content from user message (or prompt user to describe)
3. Call `save_work_state(project, content)` — upsert
4. Return: `"✅ 專案進度已更新。"`

**Multi-turn confirmation state:**
Add `pending_memory_action` field to `session_store[session_id]`:
```python
{
    "action": "invalidate",
    "candidates": [memory_id_1, memory_id_2, memory_id_3],
    "current_index": 0,
}
```
On next user turn, check if `pending_memory_action` exists before routing to RAG.

### Mechanism B — Session Consolidation (implement last)

Triggered when consolidation phrase detected (same detection pattern as A).

**Consolidation prompt sent to LLM (using SYNTHESIS_MODEL):**
```
以下是這次 session 的完整對話記錄：
{session_history}

請從中萃取：
1. NEW_CONCLUSIONS（新的研究結論或假說，每條一句，標明 type: hypothesis/conclusion）
2. POSSIBLE_CONFLICTS（與「現有記憶」可能矛盾的部分，現有記憶列表如下）
3. WORK_STATE_UPDATE（專案進度更新建議）

現有相關記憶：
{top_recalled_memories}

輸出格式（嚴格遵守）：
NEW_CONCLUSIONS:
- [hypothesis] {sentence}（來源：{papers or "本次討論推論"}）
- [conclusion] {sentence}（來源：{papers}）

POSSIBLE_CONFLICTS:
- New: {sentence} vs Existing: {existing_id} 「{existing_content}」

WORK_STATE_UPDATE:
當前焦點：{text}
進行中的問題：{bullet list}
下一步：{bullet list}
```

**Response to user:**
```
📋 **這次討論的整合摘要**

**建議新增的記憶：**
1. [假說] BSH 在酸性環境下穩定性降低（信心度：暫定）
2. [結論] BNCT 療效與中子束…

**可能的記憶衝突：**
⚠️ 新理解「X」與你的舊假說「Y」不同，是否更新？

**專案進度更新建議：**
當前焦點：探索 BSH 的合成路徑優化

---
回覆「確認全部」儲存所有項目，或指定編號選擇性接受（如「1 3 跳過2」）。
```

On user confirmation → call `save_memory_atomic()` for each accepted item.

---

## Part 3: Memory Recall Changes

### 3.1 Episodic recall — filter by status

```python
def recall_memories(collection, question: str) -> str:
    results = collection.query(
        query_texts=[question],
        n_results=min(cfg.MEMORY_RECALL_N, collection.count()),
        where={"status": {"$ne": "invalidated"}}   # ChromaDB metadata filter
    )
```

### 3.2 New: recall_work_state

```python
def recall_work_state(project: str) -> str:
    """
    Retrieve current work state for a project.
    Returns empty string if none exists.
    """
    work_state_collection = ...  # passed in or retrieved from client
    try:
        result = work_state_collection.get(ids=[f"work_state_{project}"])
        docs = result.get("documents", [])
        return docs[0] if docs else ""
    except Exception:
        return ""
```

### 3.3 Work state briefing trigger

When user says `"目前進度"`, `"這個專案進行到哪了"`, `"briefing"` etc. → return `recall_work_state()` directly, skip RAG pipeline.

### 3.4 Memory injection point changes

| Memory type | Inject into | Purpose |
|-------------|------------|---------|
| `work_state` | Session start (on demand) | Quick briefing |
| `preference` | Stage 4 system prompt context | Always active |
| `episodic` (confirmed) | Stage 1 planning prompt | Inform sub-question decomposition |
| `episodic` (all non-invalidated) | Stage 4 synthesis prompt | Reference for answer generation |

Currently memory only goes into Stage 4. Add confirmed episodic to Stage 1 planning.

In `query_planning.py`, add optional `confirmed_context` parameter to the planning prompt:
```
過去已確認的相關結論（可直接引用，無需重新推導）：
{confirmed_conclusions}
```

---

## Part 4: Files to Modify

### `config.py`
- Add all new constants listed in 1.4

### `rag/memory.py`
New / changed functions:

| Function | Change |
|----------|--------|
| `init_memory()` | Add `work_state_collection`; return 3-tuple |
| `save_memory()` | Rename to `save_memory_atomic()`; accept single conclusion sentence + new metadata schema |
| `recall_memories()` | Add `where={"status": {"$ne": "invalidated"}}` filter |
| `decide_and_save()` | Add LLM extraction step to get atomic conclusions; call Mechanism C check |
| `update_memory_status(collection, memory_id, new_status)` | NEW — updates metadata status field |
| `find_similar_memories(collection, text, threshold)` | NEW — returns list of (id, content, similarity, metadata) |
| `save_work_state(project, content)` | NEW — delete-then-add upsert with fixed ID |
| `recall_work_state(work_state_collection, project)` | NEW |
| `_check_memory_action_trigger(text)` | NEW — returns action type string or None |
| `_check_consolidation_trigger(text)` | NEW |
| `_check_work_state_trigger(text)` | NEW |
| `run_consolidation(session_history, recalled_memories, llm)` | NEW — returns structured consolidation output |

### `api.py`
- `init_memory()` now returns 3-tuple → update unpacking
- Before RAG pipeline: check `_check_memory_action_trigger()` and `_check_consolidation_trigger()`
- Check `pending_memory_action` in session_store first (multi-turn confirmation)
- After answer: append conflict warning if Mechanism C flagged one
- Pass `confirmed_episodic` to `execute_structured_query` for Stage 1

### `rag/query_pipeline.py` / `rag/query_planning.py`
- Accept optional `confirmed_conclusions: str` parameter
- Inject into Stage 1 planning prompt when non-empty

### `main.py`
- Update `init_memory()` call to unpack 3 values

---

## Part 5: Implementation Order

1. **Schema + storage format** (`memory.py`)
   - Add status to metadata
   - Change save to atomic conclusions (add LLM extraction step)
   - Update recall to filter invalidated
   - Add `work_state` collection and functions

2. **Mechanism C — conflict guard** (`memory.py`, `api.py`)
   - `find_similar_memories()`
   - Conflict LLM call
   - Warning append to response

3. **Mechanism A — quick triggers** (`memory.py`, `api.py`)
   - Trigger detection functions
   - Multi-turn confirmation state in session_store
   - `update_memory_status()`
   - work_state upsert flow

4. **Stage 1 injection** (`query_pipeline.py`, `query_planning.py`)
   - Pass confirmed conclusions into planning prompt

5. **Mechanism B — session consolidation** (`memory.py`, `api.py`)
   - Consolidation prompt + LLM call
   - Selective confirmation flow

---

## Decisions Made (do not re-derive)

- **No auto status changes**: Mechanisms never change memory status without user confirmation
- **Atomic storage**: One conclusion per document, not one Q&A per document
- **work_state is upsert**: Fixed ID, delete-then-add, one snapshot per project
- **invalidated = never recalled**: Hard filter in `where` clause, not soft weight
- **preference stays global**: Not per-project; workflow preferences apply across all projects
- **per-project episodic**: `episodic_memory_{project}` naming already in place, keep it
- **project switching still requires restart**: No runtime project switching in scope; `work_state` and `episodic` are both per-project and rely on `cfg.ACTIVE_PROJECT` at startup
- **Consolidation uses SYNTHESIS_MODEL**: Same model as Stage 3 (gemma4:31b), no new model needed
- **Conflict detection uses a LLM call**: Required when similarity > threshold; small models (qwen2.5:14b or PLANNING_LLM_MODEL) are sufficient here since it's a simple 3-way classification
