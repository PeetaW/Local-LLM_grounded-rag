# Claude Code 任務清單
> 整合 Pipeline V3 升級 + Code Review 重構
> 請按任務編號順序執行，每完成一個子任務先回報確認，再繼續下一個。

---

## 背景說明

本清單包含兩個來源的任務：

**來源 A：Pipeline V3 升級**
- Bug 修復（synthesis source、grounding score、keep_alive）
- mDeBERTa NLI 驗證擴展（矛盾偵測、子命題拆解、多來源聯合驗證）
- Plan-and-Execute 狀態表架構
- num_ctx 優化

**來源 B：Code Review 重構**
- `api.py` 重複邏輯抽離
- `_StatusCapture` 改為 `status_callback` 模式
- `config.py` 設計隱患標注
- ADR 文件建立

兩個來源的任務**互不衝突**，但執行順序有依賴關係，見最後的執行順序建議。

---

## 任務零：重構前置準備（最優先）

> 先把 code 結構清理乾淨，後續所有修改才有穩固的基礎。

### 0-A：確認 git 狀態

執行前先確認目前 repo 是乾淨的：
```bash
git status
git stash  # 若有未提交的修改先暫存
```

### 0-B：新增 `rag/answer_processor.py`

從 `api.py` 的 streaming 和 non-streaming 兩個分支中，把以下**重複出現兩次**的邏輯抽出來，建立新檔案 `rag/answer_processor.py`：

**要抽出的邏輯一：grounding 後處理**
```python
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
```

**要抽出的邏輯二：session 寫入**
```python
if session_id not in session_store:
    session_store[session_id] = []
session_store[session_id].append((question, answer))
if len(session_store[session_id]) > SESSION_MAX_TURNS:
    session_store[session_id] = session_store[session_id][-SESSION_MAX_TURNS:]
_trim_session_store()
```

整合成：
```python
# rag/answer_processor.py

def post_process_answer(
    question: str,
    answer: str,
    session_id: str,
    session_store: dict,
    session_max_turns: int,
    episodic_collection,
    preference_collection,
):
    """
    串流與非串流兩個分支的統一後處理入口。
    負責：session 寫入 + grounding 審查 + 記憶決策。
    """
    # session 寫入
    ...
    # grounding 審查
    ...
```

`api.py` 的兩個分支改為各呼叫一次 `post_process_answer(...)`，刪除重複 code。

### 0-C：`query_engine.py` 改為 `status_callback` 模式

**問題**：`api.py` 現在用 `_StatusCapture` 攔截全域 `sys.stdout` 來捕捉 pipeline 進度，這在多 worker 並發下會有 race condition。

**修法**：

在 `execute_structured_query_stream()` 函數簽名加入 `on_status` 參數：
```python
def execute_structured_query_stream(
    question, paper_engines, memory_context,
    on_status=None    # 新增，預設 None 保持向下相容
):
```

函數內所有 `print("進度訊息")` 的地方改為：
```python
def _status(msg):
    if on_status:
        on_status(msg)
    else:
        print(msg)  # fallback：terminal 測試不受影響

_status("📄 論文篩選中...")
```

`on_status` 需要一路往下傳給被呼叫的子模組（`knowledge_synthesizer.py`、`answer_verifier.py` 等），這些模組裡的 `print()` 也要同樣替換。

**`api.py` 的 `generate()` 對應修改**：

移除 `_StatusCapture` 的使用，改為：
```python
def _run():
    def status_handler(msg):
        q_in.put(f"[STATUS] {msg}\n")  # 每個請求有自己的 handler

    for chunk_text in execute_structured_query_stream(
        question, paper_engines, memory_context,
        on_status=status_handler
    ):
        q_in.put(chunk_text)
    q_in.put(None)
```

`_StatusCapture` class 整個刪除。

在 `ThreadPoolExecutor` 那行加上說明：
```python
executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
# max_workers=1：本機單卡（RTX 3090）不支援真正並發推理，此為資源限制而非設計限制。
# 改為 status_callback 模式後，若未來多卡部署可直接調整此數值。
```

### 0-D：`config.py` 添加設計說明

在路徑變數區塊上方加上：
```python
# NOTE: 以下路徑在 module import 時就計算完成（module-level 靜態值）。
# ACTIVE_PROJECT 切換需重啟 server 才會生效，不支援 runtime 動態切換。
# 若未來需要 runtime 切換 project，路徑應改為函數：get_papers_dir(project_name)。
```

### 0-E：建立 ADR 文件

新增目錄與檔案：`docs/decisions/001-single-worker-status-callback.md`

內容：
```markdown
# ADR 001：Pipeline 進度訊息的串流設計

## 狀態
已實作（V3）

## 決策
使用 status_callback（on_status 參數）模式傳遞 pipeline 進度訊息，
而非攔截全域 sys.stdout。

## 背景
V2 使用 _StatusCapture 攔截 sys.stdout，搭配 max_workers=1 迴避 race condition。
此設計可運作，但假設是隱性的，且未來難以擴展。

## 取捨
- callback 模式：每個 request 有獨立的 handler，天然並發安全，可測試性高
- stdout 攔截：實作簡單，但依賴全域狀態，並發不安全

## 觸發重新評估的條件
若未來需要把進度訊息寫入 log 系統或 metrics，
在 status_handler 內加入對應邏輯即可，不需修改 pipeline 本身。
```

---

## 任務一：修復已知 Bug

### 1-A：synthesis_chunks source 欄位修正

**檔案**：`rag/query_engine.py`

**問題**：synthesis_chunks 的 source 是匿名的 `retrieved_chunk_N`，導致 Stage 5 無法追蹤來源。

```python
# 修改前
synthesis_chunks = [
    {"text": ans, "source": f"retrieved_chunk_{i}"}
    for i, ans in enumerate(sub_answers)
]

# 修改後
synthesis_chunks = []
for ans in sub_answers:
    lines = ans.split("\n")
    source = lines[0].strip("【】") if lines else "unknown"
    synthesis_chunks.append({"text": ans, "source": source})
```

### 1-B：grounding score 解析失敗修復

**檔案**：`rag/query_engine.py`、`rag/citation_grounding.py`

1. 找出 `compute_grounding_score` 的回傳值格式
2. 找出 `query_engine.py` 裡解析 grounding score 的邏輯
3. 確認兩者格式一致，修復不匹配
4. 修復後確認 episodic memory 有正確寫入

### 1-C：確認 keep_alive 寫入所有 Ollama API payload

搜尋所有 Ollama API 呼叫，確認每個 payload 都包含：
```python
"keep_alive": "30m"
```
如有遺漏，補上。

---

## 任務二：mDeBERTa NLI 驗證邏輯擴展

**檔案**：`rag/citation_grounding.py`

### 2-A：啟用矛盾偵測

在現有 NLI 計算中同時輸出 contradiction score。

當 contradiction score > 0.7 時，標記為知識庫內部矛盾：
```python
{
    "sentence": "...",
    "grounding_score": 0.82,
    "contradiction_detected": True,
    "contradiction_source": "論文B第2段",
    "status": "CONFLICT"
}
```

### 2-B：子命題拆解驗證

新增 `decompose_and_verify()` 函數：

```python
def decompose_and_verify(conclusion: str, facts: list[dict]) -> dict:
    """
    輸入：結論句子 + facts 列表
    輸出：{
        "conclusion": "...",
        "sub_claims": [
            {
                "claim": "子命題文字",
                "grounding_score": 0.91,
                "source": "來源",
                "status": "SUPPORTED" | "INFERENCE_BRIDGE" | "UNSUPPORTED"
            }
        ],
        "chain_complete": True/False
    }
    """
```

閾值：
- grounding_score < 0.4 → UNSUPPORTED
- grounding_score 0.4-0.65 → INFERENCE_BRIDGE
- grounding_score > 0.65 → SUPPORTED

### 2-C：多來源聯合驗證

新增 `joint_verify()` 函數：

```python
def joint_verify(claim: str, facts: list[dict]) -> dict:
    """
    對一個子命題取 top-3 相關 facts，
    單獨各跑一次 NLI + 拼接後再跑一次 NLI。
    單獨分數低但聯合分數高 → inference_bridge。

    輸出：{
        "claim": "...",
        "individual_scores": [0.23, 0.31, 0.28],
        "joint_score": 0.79,
        "is_inference_bridge": True,
        "bridge_sources": ["論文A", "論文B"]
    }
    """
```

---

## 任務三：Plan-and-Execute 狀態表架構

### 3-A：新增 `rag/task_state.py`

建立任務狀態表資料結構，包含：
- `TaskStatus` enum（PENDING / RUNNING / COMPLETED / FAILED）
- `InferenceType` enum（DIRECT / INFERENCE_BRIDGE / UNSUPPORTED）
- `SubTask` dataclass
- `ResearchPlan` dataclass，包含方法：
  - `get_ready_tasks()`：回傳前置已完成、自身 PENDING 的任務
  - `get_task_by_id(task_id)`
  - `all_completed()`
  - `to_summary()`：轉成給最終整合用的結構化摘要

完整資料結構見原始 `pipeline_v3_task_spec.md` 的任務三-A。

### 3-B：修改 Stage 1 輸出格式

**檔案**：`rag/query_engine.py` Stage 1 部分

修改 Stage 1 prompt，要求輸出包含依賴關係的任務列表：

```json
// 修改前
["子問題1", "子問題2", "子問題3"]

// 修改後
[
    {"id": "T1", "question": "子問題1", "depends_on": []},
    {"id": "T2", "question": "子問題2", "depends_on": []},
    {"id": "T3", "question": "子問題3", "depends_on": ["T1", "T2"]}
]
```

### 3-C：新增 `rag/plan_executor.py`

建立 `PlanExecutor` class，包含方法：
- `execute(plan: ResearchPlan) -> ResearchPlan`
- `_build_context_for_task(task, plan) -> str`（只包含前置結論 + 相關 facts，目標 8000-12000 tokens）
- `_synthesize_final_answer(plan) -> str`

完整介面定義見原始 `pipeline_v3_task_spec.md` 的任務三-C。

### 3-D：`config.py` 新增開關

```python
# Plan-and-Execute 架構開關
PLAN_EXECUTE_ENABLED = False       # 預設關閉，穩定後開啟

# NLI 擴展開關
NLI_DECOMPOSE_ENABLED = False      # 子命題拆解驗證
NLI_JOINT_VERIFY_ENABLED = False   # 多來源聯合驗證
NLI_CONTRADICTION_ENABLED = True   # 矛盾偵測（預設開啟）
```

---

## 任務四：num_ctx 優化

### 4-A：`config.py` 新增各 Stage 的 context 限制

```python
STAGE1_NUM_CTX = 4096     # 子問題拆解
STAGE3_NUM_CTX = 16384    # 知識蒸餾
STAGE4_NUM_CTX = 16384    # 學術推理
STAGE5_NUM_CTX = 32768    # 邏輯驗證
```

### 4-B：各 Stage 的 Ollama API payload 套用對應 num_ctx

**檔案**：`rag/query_engine.py`、`rag/knowledge_synthesizer.py`、`rag/answer_verifier.py`

```python
payload = {
    "model": model_name,
    "messages": messages,
    "keep_alive": "30m",
    "options": {
        "num_ctx": STAGE5_NUM_CTX  # 依 stage 使用對應值
    }
}
```

---

## 任務五：測試與驗證

### 5-A：新增 `scripts/test_nli_extensions.py`

針對 mDeBERTa 擴展的單元測試：
- 矛盾偵測測試（固定假資料）
- 子命題拆解測試
- 多來源聯合驗證測試

### 5-B：整合測試（使用現有 `test_query.py`）

```
測試一：PLAN_EXECUTE_ENABLED=False，NLI_CONTRADICTION_ENABLED=True
        驗證矛盾偵測有無正確輸出

測試二：PLAN_EXECUTE_ENABLED=True，其他功能關閉
        驗證狀態表架構能否正確執行並輸出結果

測試三：全部功能開啟
        驗證完整 V3 pipeline 端對端輸出
```

### 5-C：速度基準測試

記錄各 Stage 耗時，與 V2 對比：

```
目標：
- Stage 5 單次驗證：500-2000s → 150-300s
- 全流程：100+ 分鐘 → 30 分鐘以內
```

---

## 不需要修改的檔案

- `rag/memory.py`
- `rag/retriever.py`
- `rag/reranker.py`
- `rag/embeddings.py`

---

## 執行順序

```
任務零（重構前置）
  0-A git 狀態確認
  0-B answer_processor.py 抽離
  0-C status_callback 模式
  0-D config.py 說明
  0-E ADR 文件
    ↓ 確認 api.py 正常運作後繼續
任務一（Bug 修復）
  1-A → 1-B → 1-C
    ↓ 確認 grounding score 有正確寫入記憶後繼續
任務四（num_ctx 優化）
  4-A → 4-B
    ↓ 確認速度改善、CPU offload 消失後繼續
任務二（NLI 擴展）
  2-A（矛盾偵測，最簡單，先驗證）
    ↓
  2-B → 2-C
    ↓
任務三（Plan-and-Execute）
  3-A → 3-B → 3-C → 3-D
    ↓
任務五（完整測試）
  5-A → 5-B → 5-C
```

---

## 給 Claude Code 的執行提示

- 每完成一個子任務（如 0-B、1-A），先回報確認再繼續
- 任務零完成後，務必跑 `python scripts/test_query.py` 確認 pipeline 輸出正常再繼續
- 新增功能優先用 config 開關控制，預設關閉，確保不破壞現有流程
- 修改現有檔案前先確認 git 狀態（`git diff`）
- 有任何不確定的地方先問，不要自行假設
- `rag/` 底下的子模組若有 `print()` 進度訊息，一律改為接受並呼叫 `on_status` callback
