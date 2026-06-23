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

### ⚠️ Phase 0 結果（2026-06-23，baseline_v3）→ loop 重定位
7 個低分題**選擇命中全 1.0、檢索覆蓋幾乎全 1.0** → **沒有一題是「問錯/檢索失敗」**。
「問錯」≈0、「內容沒抽到」≈0（figure/table 題覆蓋也 1.0）、grounding 低估(Q05/Q12)、**其餘全是生成錯誤**（從正確素材捏造/誤讀/漏講）。
**結論：原本想的「重檢索 loop」對此題組≈零價值。** 改定位成：
1. **生成自我修正 loop**——針對 grounding 標記的句子、對著該 chunk 精準重寫（＝關掉的 fallback 的精準版）。
2. **前置：先修 grounding 假性低估**。Q05 案例：答案全對(judge 1.0)但 grounding 0.26，因 **atomic-completeness prompt 過度切碎**（列舉句拆成逐項碎片 ×8、表格逐格拆原子句 → NLI 接不回）。修法：refine atomic prompt 讓列舉/清單保持一句，atomic 只切真正不同的事實。
但書：12 題是精選語料+清楚問法；真實使用者的模糊/跨庫/超綱問題會讓重檢索 loop 更有用 → 非放棄，是當前優先序低於生成槓桿。

### ⚠️ Phase 1 結果（2026-06-23，eval_selfcorrect）→ 便宜版生成修正 loop 也是負結果
實作便宜版（一次 batched gemma4 裁定 NLI 標記句，`GENERATION_SELFCORRECT_ENABLED` 預設關，code 留 citation_grounding.selfcorrect_flagged）。
**fix=0 全部題、correctness 零提升。** 真實失敗形狀＝**多講/漏講/無中生有**，**沒有一個是「chunk 矛盾答案」** → 修矛盾的 corrector 用不上。
**S/N 副發現**：NLI 標「不支持」≈**90% 假陰性**（37 keep : 0 矛盾 : 3 真沒依據）→ NLI 逐句訊號很吵，corrector 是好的「噪音過濾器」非「答案修正器」。成本每題 +98–226s gemma4 換零提升 → 不出貨。
**改往「對應失敗形狀」的修法**（見下方新章節「生成品質修正」）。

實驗分階段：
- **Phase 0**：✅ 完成。結論：loop 從「重檢索」改「生成自我修正」。
- **Phase 1**：✅ 完成（便宜版生成修正 loop）。**負結果**（fix=0，失敗形狀非矛盾）→ 改往下方「生成品質修正」。
- 原本規劃的「重檢索 loop Phase 1–3（while + 重構查詢）」**擱置**：Phase 0 已證此語料檢索非瓶頸；真實使用者模糊/超綱問題才用得上，屆時再啟。

---

### E. 生成品質修正（取代 loop，對應真實失敗形狀）
Phase 0/1 證實：失敗都在生成端，分三種形狀，各需不同機制：

| 失敗形狀 | 例 | 可偵測? | 修法 | 信心/成本 |
|---|---|---|---|---|
| **無中生有**（捏造 chunk 沒有的） | Q06 共價鍵 | ✅ corrector UNVERIFIED 高精度 | **把 selfcorrect 的 UNVERIFIED 從「標記」改「刪除」**——移除幻覺句 | 高、近零成本（code 已在），先在 Q06 測 |
| **多講-捏造** | Q04 部分 | ✅ 同上 | 同上（刪除） | 同上 |
| **多講-正確但離題** | Q04 部分 | ❌ grounding 不標（內容真在某 chunk） | 生成 prompt「答題勿轉錄整篇」+ 查 judge 是否過嚴懲罰多餘正確內容 | 中、prompt 易翻車、需獨立 A/B |
| **漏講-數值** | Q01 漏 193 | ⚠️ 可比對 | 從檢索 chunk 抽數值/實體，比對答案缺漏並補 | 中、針對性 |
| **漏講-概念** | Q09 漏策略框架 | ❌ 難 | LLM「答案是否涵蓋問題各面向」檢查 | 低信心、貴、最後做 |

**meta 結論**：此語料天花板＝生成保真度；post-hoc 修正槓桿有限。最高信心起手＝**無中生有→刪除**（小改、可立刻測）。
**注意**：刪除依賴 corrector UNVERIFIED（高精度、偏保守），**不可**用 NLI 原始標記直接刪（90% 假陰性會刪掉正確內容）。

---

## 建議推進序（2026-06-23 更新：Phase 0 已完成，重排）
1. ✅ baseline_v3、✅ agentic loop Phase 0、✅ 生成修正 loop Phase 1（負結果：fix=0）
2. **無中生有→刪除**（E：把 selfcorrect 的 UNVERIFIED 改成刪除幻覺句）— 最高信心起手，Q06 測
3. 可答性 gate（A）— 安全/誠實；與「整篇都不 grounded→棄答」共用訊號
4. 多講/漏講修正（E）— 生成 prompt 答題勿轉錄 + 數值完整性比對；prompt 易翻車，獨立 A/B
5. pipeline_v4 分階段索引（1）→ 順帶解鎖匯入健檢（B）
6. memory_redesign（2）
7. retrieval 拆解（C）、api-refactor（3）視情況
