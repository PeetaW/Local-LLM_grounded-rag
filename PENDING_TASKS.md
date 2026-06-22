# 待實作任務看板（PENDING TASKS）

> 用途：跟 spec 雙邊對照，確保任務推進。完成的 spec 已移至 `archive/specs/`（見該處 STATUS.md）。
> 最後盤點：2026-06-22（對照 code 逐一驗證）。

## 已完成（歸檔，僅供回顧）
- ✅ pipeline_v2（Stage 3/4/5）
- ✅ pipeline_v3（NLI 擴展、Plan-Execute；後者預設關閉）
- ✅ query-engine-refactor（拆模組、stream/non-stream 共用、死碼 query_engine.py 已刪）
→ 細節見 `archive/specs/STATUS.md`

---

## 待實作（有詳細 spec）

### 1. pipeline_v4 — 分階段索引（`pipeline_v4_task_spec.md`）
**狀態：未開始。** 現況：索引一條龍同步跑 VL（indexer.py:39-42），VL 失敗會卡住建索引。
目標：拆成 fast base-index（先可搜尋）+ 非阻塞 VL/摘要增量豐富化 + per-paper 狀態追蹤
（`text_index_ready`/`vl_pending`/`vl_partial`/`summary_ready`/`last_successful_build`）+ 安全增量重建。
**重要：這是下方「匯入健檢」的地基**——per-paper 狀態 + 分階段索引正好支撐「論文匯入後確認索引健康」。

### 2. memory_redesign — 記憶模組重設計（`memory_redesign_spec.md`）
**狀態：未開始。** 研究知識管理層（非問答 log）。三類 episodic/preference/work_state、
狀態生命週期、原子結論句、三機制（C 衝突守衛→A 快速觸發→B session 整合）。
屬 Tier 2；前置條件「穩定量尺」已於 2026-06 備齊（量尺三軸：檢索/忠實度/正確性）。

### 3. api-refactor — API 分層（`api-refactor-spec.md`）
**狀態：未開始。** api.py（474 行）仍把 routes/schemas/session/injection/memory/orchestration 全混一檔，
import 時還連帶觸發 main.py 全域初始化。目標：拆成薄 transport 層 + 清楚服務邊界 + 安全啟動。
（優先序最低——目前不是瓶頸。）

---

## 待實作（roadmap，尚無正式 spec）

### A. 可答性 gate（answerability gate）
回答前判「檢索 chunk 是否真的**包含**答案，而非只是主題相關」；只相關→走誠實棄答。
根治 Q11 那種「有免責橫幅但仍編造具體數字」的過度延伸。

### B. 匯入健檢（ingestion sanity check）
論文匯入後自動用它自己的摘要/標題生 query，確認索引回得出合理 chunk + grounding 跑得動，
抓出抽取失敗（如掃描檔 OCR 壞）的論文，不需 gold 標籤。**與 pipeline_v4 的 per-paper 狀態追蹤共用地基。**

### C. 延遲：retrieval 327s 拆解
baseline 後 retrieval 是最大階段，待拆（檢索本身 vs Ollama model swap）。

### D. Agentic RAG loop — 自我迭代檢索（Tier 2 核心）
**概念**：系統自評「這次答得好不好」，不夠好就換問法/換論文重檢索，直到夠好或撞迭代上限。
**為何可行**：難的部分（驅動訊號）已有——grounding 分數、`rag_found_anything`、可答性 gate(A) 都是**免標籤、任何論文自動算**的訊號；狀態追蹤有 `plan_executor.py`/`task_state.py` 地基。
**關鍵區分（產品化命題）**：gold/量尺＝**開發者**校準機器的工具（固定 benchmark 跑一次）；運行時 loop 靠**免標籤自評**（grounding/可答性），使用者 import 新論文**永不碰 gold**。
**誠實限制**：grounding 量忠實度非正確性 → loop 可能收斂到「有依據但答錯」。所以靠 benchmark（含正確性 judge）驗證「grounding 驅動的 loop 真能提升正確性」才出貨；運行時則靠出處透明 + 誠實棄答，承諾「誠實標信心」而非「保證正確」。
**另一限制**：loop 只修「問錯」，修不了「內容沒抽到」（表格/圖片/抽取壞 → 要靠 v4/表格抽取，非 loop）。

實驗分階段：
- **Phase 0**：把目前失敗題（grounding 低：Q07 0.48/Q05 0.53/Q10 0.62/Q06 0.72）分類成「問錯(loop 可救)」vs「內容不在(loop 白繞)」。純資料分析，不寫 loop。
- **Phase 1**：最小 loop 藏 flag 後（`AGENTIC_LOOP_ENABLED`、`MAX_RETRIEVAL_RETRIES=2`，預設關，比照 `PLAN_EXECUTE_ENABLED`）。grounding<0.5 且有額度 → 1 次 LLM 重構子問題 → 重跑 retrieval+synthesis+grounding → 取最高分那次。**硬封頂 2 次**（血淚教訓：無界 loop 會爆）。
- **Phase 2**：A/B loop ON vs OFF，看靶題 grounding/正確性升幅 + 延遲代價（每 retry +300–400s）。grounding 已高的題要 early-exit。
- **Phase 3**（Phase 1 有效才做）：門檻觸發升級成可答性判斷（缺什麼→朝缺口重構）；`task_state` 記試過的查詢避免重複。
**框架**：Phase 1 用帶上限的 while 迴圈即可，**先別上 LangGraph**；等分支變多再考慮。

---

## 建議推進序
1. 完整 12 題新基準線（baseline_v3）當對照 ← 進行中
2. 可答性 gate（A）— 安全/誠實，最該優先；也是 agentic loop 的反思訊號
3. agentic loop Phase 0（D，失敗題分類）— 純分析，先確認 loop 對本題組有多少上限空間
4. agentic loop Phase 1–2（D）— flag 後最小 loop + A/B
5. pipeline_v4 分階段索引（1）→ 順帶解鎖匯入健檢（B）
6. memory_redesign（2）
7. retrieval 拆解（C）、api-refactor（3）視情況
