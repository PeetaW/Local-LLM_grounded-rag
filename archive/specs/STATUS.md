# 已完成 Spec 歸檔索引

對照 code 重新驗證於 2026-08-02。以下 spec 的核心目標皆已在現行 code 落地，移此歸檔留存。

| Spec | 完成內容 | 驗證證據 |
|---|---|---|
| `pipeline_v2_task_spec.md` | Stage 3 知識蒸餾 + Stage 4 gemma4 + Stage 5 verify→correct 閉環 | config.py（`SYNTHESIS_ENABLED`/`VERIFY_ENABLED`/`VERIFY_MODEL`）、answer_verifier.py |
| `pipeline_v3_task_spec.md` | NLI 擴展（decompose/joint）、context 優化、Plan-and-Execute 狀態表 | citation_grounding.py、plan_executor.py、task_state.py、`PLAN_EXECUTE_ENABLED`（**預設關閉**） |
| `query-engine-refactor-spec.md` | query_engine.py 拆成 stage 模組（types/planning/embedding_guard/retrieval/grounding_flow/translation/prompts/pipeline）；stream/non-stream 共用 | rag/query_*.py 九模組齊全；api.py/main.py/run_eval 全 import 自 `rag.query_pipeline`；舊 `rag/query_engine.py` 死碼已刪（本檔同層仍留一份歷史備份） |

備註：`query_formatting.py` 依 spec 允許暫緩，未建。

## 現行規劃文件（留在 repository root）

| 文件 | 狀態 |
|---|---|
| `PENDING_TASKS.md` | 唯一現行 checkpoint 與排序後 roadmap |
| `maintainability_refactor_spec.md` | 已完成稽核；待 v11 baseline 凍結後開始 |
| `pipeline_v4_task_spec.md` | 未開始；分階段索引主線 |
| `ingestion_health_spec.md` | MVP 完成；Phase 2 等待 Pipeline v4 |
| `memory_redesign_spec.md` | 未開始；Pipeline v4/健檢後處理 |
| `api-refactor-spec.md` | 未開始；目前優先序最低 |

`claude_code_tasks.md` 是已完成 V3 工作的歷史任務清單，保留在 root
僅為避免本次文件整理混入大範圍搬檔；不得當作現行 pending board。
