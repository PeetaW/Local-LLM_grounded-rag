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

---

## 建議推進序
1. 完整 12 題新基準線（baseline_v3）當對照 ← 進行中
2. 可答性 gate（A）— 安全/誠實，最該優先
3. pipeline_v4 分階段索引（1）→ 順帶解鎖匯入健檢（B）
4. memory_redesign（2）
5. retrieval 拆解（C）、api-refactor（3）視情況
