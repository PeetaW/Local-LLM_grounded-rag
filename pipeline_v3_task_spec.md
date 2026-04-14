# Pipeline V3 任務規格
> 給 Claude Code 執行用。請按任務編號順序執行，每完成一個任務先回報確認，再繼續下一個。

---

## 背景說明

現有 pipeline 是 V2 五階段架構：
- Stage 1：子問題拆解（qwen2.5:14b）
- Stage 2：混合檢索（BM25 + bge-m3 + reranker）
- Stage 3：知識蒸餾（gemma4:31b）
- Stage 4：學術推理（gemma4:31b）
- Stage 5：邏輯驗證（qwen3.5:35b-a3b）→ verify→correct→re-verify 閉環

本次 V3 的目標是：
1. 修復已知 bug
2. 優化 context 使用效率
3. 擴展 mDeBERTa NLI 驗證邏輯
4. 引入 Plan-and-Execute 狀態表架構

---

## 任務一：修復已知 Bug（最優先）

### 1-A：synthesis_chunks source 欄位修正

**檔案**：`rag_project/rag/query_engine.py`

**問題**：打包 synthesis_chunks 時 source 是匿名的 `retrieved_chunk_N`，導致 Stage 5 無法追蹤來源，誤判推論跳躍。

**修法**：
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

**檔案**：`rag_project/rag/query_engine.py` 和 `rag_project/rag/citation_grounding.py`

**問題**：每次都出現 `grounding score 解析失敗，不寫入長期記憶`，導致記憶系統完全沒有存入任何東西。

**任務**：
1. 找出 `compute_grounding_score` 的回傳值格式
2. 找出 query_engine.py 裡解析 grounding score 的邏輯
3. 確認兩者格式一致，修復不匹配的地方
4. 修復後確認 episodic memory 有正確寫入

### 1-C：confirm keep_alive 寫入所有 Ollama API payload

**檔案**：所有呼叫 Ollama API 的地方

**任務**：搜尋所有 Ollama API 呼叫，確認每個 payload 都包含：
```python
"keep_alive": "30m"
```
如果有遺漏，補上。

---

## 任務二：mDeBERTa NLI 驗證邏輯擴展

**檔案**：`rag_project/rag/citation_grounding.py`

現有邏輯是一對一驗證：一個結論 vs 一個來源，只用 entailment score。

### 2-A：啟用矛盾偵測

**任務**：在現有 NLI 計算中，同時輸出 contradiction score。

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

**任務**：在 citation_grounding.py 新增 `decompose_and_verify()` 函數。

流程：
1. 接收一個結論句子
2. 呼叫 gemma4:31b 把結論拆解成子命題列表（JSON 格式）
3. 每個子命題分別跑 mDeBERTa NLI，對應最相關的 facts
4. 輸出每個子命題的 grounding score 和 status

```python
def decompose_and_verify(conclusion: str, facts: list[dict]) -> dict:
    """
    輸入：一個結論句子 + facts 列表
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

grounding_score < 0.4 → UNSUPPORTED（幻覺）
grounding_score 0.4-0.65 → INFERENCE_BRIDGE（跨文獻推導，需標記）
grounding_score > 0.65 → SUPPORTED（直接支撐）

### 2-C：多來源聯合驗證

**任務**：在 citation_grounding.py 新增 `joint_verify()` 函數。

流程：
1. 對一個子命題，取 top-3 最相關的 facts
2. 單獨各自跑一次 NLI（已有）
3. 把 top-3 facts 拼接成一個 premise，再跑一次 NLI
4. 如果單獨分數低但聯合分數高 → 自動標記為 inference_bridge

```python
def joint_verify(claim: str, facts: list[dict]) -> dict:
    """
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

這是最大的架構改動，需要新增檔案和修改 Stage 1、Stage 4、Stage 5 的銜接邏輯。

### 3-A：定義任務狀態表資料結構

**新增檔案**：`rag_project/rag/task_state.py`

```python
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class InferenceType(Enum):
    DIRECT = "direct"           # 直接被單一文獻支撐
    INFERENCE_BRIDGE = "bridge" # 需要多文獻聯合推導
    UNSUPPORTED = "unsupported" # 找不到依據

@dataclass
class SubTask:
    id: str                          # 例如 "T1", "T2"
    question: str                    # 子問題
    status: TaskStatus = TaskStatus.PENDING
    depends_on: List[str] = field(default_factory=list)  # 依賴的前置任務 id
    conclusion: Optional[str] = None
    grounding_score: Optional[float] = None
    joint_grounding_score: Optional[float] = None
    sources: List[str] = field(default_factory=list)
    inference_type: Optional[InferenceType] = None
    sub_claims: Optional[list] = None  # 子命題驗證結果
    contradiction_detected: bool = False

@dataclass
class ResearchPlan:
    research_question: str
    tasks: List[SubTask] = field(default_factory=list)
    
    def get_ready_tasks(self) -> List[SubTask]:
        """回傳所有前置任務已完成、自身狀態為 PENDING 的任務"""
        completed_ids = {t.id for t in self.tasks if t.status == TaskStatus.COMPLETED}
        return [
            t for t in self.tasks
            if t.status == TaskStatus.PENDING
            and all(dep in completed_ids for dep in t.depends_on)
        ]
    
    def get_task_by_id(self, task_id: str) -> Optional[SubTask]:
        return next((t for t in self.tasks if t.id == task_id), None)
    
    def all_completed(self) -> bool:
        return all(t.status == TaskStatus.COMPLETED for t in self.tasks)
    
    def to_summary(self) -> str:
        """轉成給最終整合用的結構化摘要"""
        lines = [f"研究問題：{self.research_question}\n"]
        for t in self.tasks:
            if t.status == TaskStatus.COMPLETED:
                lines.append(f"[{t.id}] {t.question}")
                lines.append(f"  結論：{t.conclusion}")
                lines.append(f"  類型：{t.inference_type.value if t.inference_type else 'unknown'}")
                lines.append(f"  Grounding：{t.grounding_score:.2f}" if t.grounding_score else "  Grounding：N/A")
                lines.append(f"  來源：{', '.join(t.sources)}\n")
        return "\n".join(lines)
```

### 3-B：修改 Stage 1 輸出依賴關係

**檔案**：`rag_project/rag/query_engine.py` 的 Stage 1 部分

**任務**：修改 Stage 1 的 prompt，要求 qwen2.5:14b 輸出包含依賴關係的任務列表。

Stage 1 輸出格式從：
```json
["子問題1", "子問題2", "子問題3"]
```

改為：
```json
[
    {"id": "T1", "question": "子問題1", "depends_on": []},
    {"id": "T2", "question": "子問題2", "depends_on": []},
    {"id": "T3", "question": "子問題3", "depends_on": ["T1", "T2"]}
]
```

依賴關係的判斷標準：如果一個子問題需要用到另一個子問題的答案才能回答，就加入 depends_on。

### 3-C：新增 Plan-and-Execute 執行器

**新增檔案**：`rag_project/rag/plan_executor.py`

```python
class PlanExecutor:
    """
    接收 ResearchPlan，按依賴順序執行每個 SubTask。
    每個 SubTask 單獨走 Stage 2 → Stage 3 → Stage 4 → Stage 5 流程。
    """
    
    def execute(self, plan: ResearchPlan) -> ResearchPlan:
        """
        執行流程：
        1. 取得所有 ready tasks（前置完成、自身 pending）
        2. 逐一執行每個 task
           - 以 task.question 為查詢，走 Stage 2-5
           - 執行時 context 只包含：當前子問題 + 相關 facts + 前置任務的結論
           - 不包含原始 chunks
        3. 把結果寫回 task.conclusion、task.grounding_score 等欄位
        4. 標記 task.status = COMPLETED
        5. 重複直到所有 task 完成
        6. 觸發最終整合（synthesis）
        """
    
    def _build_context_for_task(self, task: SubTask, plan: ResearchPlan) -> str:
        """
        為單一 task 建立乾淨的 context：
        - 前置任務的結論（不是原始 chunks）
        - 當前任務相關的 structured facts
        目標：讓每次 LLM call 的 context 維持在 8000-12000 tokens
        """
    
    def _synthesize_final_answer(self, plan: ResearchPlan) -> str:
        """
        所有 task 完成後，以狀態表摘要為輸入，
        呼叫 gemma4:31b 做最終結構化整合輸出。
        """
```

### 3-D：新增 config 開關

**檔案**：`rag_project/config.py`

新增：
```python
# Plan-and-Execute 架構開關
PLAN_EXECUTE_ENABLED = False  # 預設關閉，穩定後開啟

# NLI 擴展開關
NLI_DECOMPOSE_ENABLED = False      # 子命題拆解驗證
NLI_JOINT_VERIFY_ENABLED = False   # 多來源聯合驗證
NLI_CONTRADICTION_ENABLED = True   # 矛盾偵測（最簡單，預設開啟）
```

---

## 任務四：num_ctx 優化

**目標**：讓 qwen3.5:35b-a3b 的 layer 全部留在 GPU，消除 CPU offload。

### 4-A：針對不同 Stage 設定不同 num_ctx

**檔案**：`rag_project/config.py`

新增：
```python
# 各 Stage 的 context 限制
STAGE1_NUM_CTX = 4096    # 子問題拆解，不需要大 context
STAGE3_NUM_CTX = 16384   # 知識蒸餾
STAGE4_NUM_CTX = 16384   # 學術推理
STAGE5_NUM_CTX = 32768   # 邏輯驗證（最大需求，但比 65536 小很多）
```

### 4-B：在各 Stage 的 Ollama API 呼叫中套用對應的 num_ctx

**檔案**：`rag_project/rag/query_engine.py`、`rag_project/rag/knowledge_synthesizer.py`、`rag_project/rag/answer_verifier.py`

每個 Ollama API payload 加入對應的 num_ctx：
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

### 5-A：單元測試 mDeBERTa 擴展

在 `rag_project/scripts/` 新增 `test_nli_extensions.py`：

```python
# 測試矛盾偵測
# 測試子命題拆解
# 測試多來源聯合驗證
# 每個測試都用固定的假資料，不需要跑完整 pipeline
```

### 5-B：整合測試

用現有的 `test_query.py`，測試以下開關組合：

```
測試一：PLAN_EXECUTE_ENABLED=False，NLI_CONTRADICTION_ENABLED=True
        驗證：矛盾偵測有無正確輸出

測試二：PLAN_EXECUTE_ENABLED=True，其他功能關閉
        驗證：狀態表架構能否正確執行並輸出結果

測試三：全部功能開啟
        驗證：完整 V3 pipeline 端對端輸出
```

### 5-C：速度基準測試

每次測試記錄各 Stage 耗時，與 V2 對比：

```
目標：
- Stage 5 單次驗證從 500-2000s 降到 150-300s
- 全流程從 100+ 分鐘降到 30 分鐘以內
```

---

## 不需要修改的檔案

以下檔案保持不動：
- `rag_project/rag/memory.py`（ChromaDB 記憶系統）
- `rag_project/rag/retriever.py`（BM25 + bge-m3 混合檢索）
- `rag_project/rag/reranker.py`（bge-reranker）
- `rag_project/rag/embeddings.py`

---

## 執行順序建議

```
任務一（Bug 修復）→ 測試確認 bug 消失
    ↓
任務四（num_ctx 優化）→ 測試確認速度改善、CPU offload 消失
    ↓
任務二-A（矛盾偵測）→ 最簡單的 NLI 擴展，先驗證
    ↓
任務三-A/B（狀態表資料結構 + Stage 1 輸出格式）
    ↓
任務二-B/C（子命題拆解 + 聯合驗證）
    ↓
任務三-C/D（Plan-and-Execute 執行器 + config 開關）
    ↓
任務五（完整測試）
```

---

## 給 Claude Code 的提示

- 每完成一個子任務（如 1-A、1-B），先回報確認再繼續
- 新增功能優先用 config 開關控制，預設關閉，確保不破壞現有流程
- 有任何不確定的地方先問，不要自行假設
- 修改現有檔案前先備份或確認 git 狀態
