# 待實作任務看板（PENDING TASKS）

> 用途：跟 spec 雙邊對照，確保任務推進。完成的 spec 已移至 `archive/specs/`（見該處 STATUS.md）。
> 最後盤點：2026-08-02（對照 code、git history、config、模組大小與最新 eval artifacts 驗證）。

## 已完成（歸檔，僅供回顧）
- ✅ pipeline_v2（Stage 3/4/5）
- ✅ pipeline_v3（NLI 擴展、Plan-Execute；後者預設關閉）
- ✅ query-engine-refactor（拆模組、stream/non-stream 共用、死碼 query_engine.py 已刪）
→ 細節見 `archive/specs/STATUS.md`

## 目前 checkpoint（2026-08-02）

- **v11 品質驗收已完成前兩道 gate**：Q07 precursor/water relation focused + stability、Q08 isotope-cost focused，以及最新 `q07_q08_q09_cost_canonical_stability` 都已實跑。三題 stability 的 correctness、grounding、translation 均為 `1.0`，judge coverage `3/3`，`C0/U0`；Q07/Q08/Q09 分別覆蓋 `8/8`、`7/7`、`7/7` reference facts。
- **完整 12 題 baseline v11 執行中**：`baseline_v11_structured_contract_full` 已於 2026-08-02 啟動，正式 Markdown 報告尚未產生，因此目前仍不能宣稱 v11 為完整產品 baseline，也不採用執行中的部分平均值。
- **最後一份已完成的完整 12 題仍是 v9**：`baseline_v9_fact_contract_full` correctness `0.646`、grounding `0.908`、translation coverage `10/12`。八題 v10 smoke 為 correctness `0.906`、grounding `0.964`，但不可與完整題組直接等同。
- **structured contract 主路徑已穩定跨題型**：non-comparison fact contract、method direct render、comparison JSON validator/direct render、structured correctness/translation judge 都在產品路徑啟用；近期 Q07/Q08/Q09 Stage 3 皆 `done_reason=stop`，未觸發 repair、length truncation 或 Stage 4 fallback。
- **仍需在全題組追蹤的診斷項目**：Q07/Q09 的 Stage 2 evidence recall 在最新 stability 為 `50.0%/57.1%`，但 final fact audit 與 grounding 均完整；Q08 answerability 仍可能判為 `PARTIAL` 並顯示保守警告。兩者先以完整 v11 的跨題型分布判斷，不先改 retrieval 或 gate。
- **產品 config 已確認**：`ANSWERABILITY_GATE_ENABLED=True`、`FINAL_TRANSLATION_ENABLED=True`、`STRUCTURED_FACT_CONTRACT_ENABLED=True`、`METHOD_FACT_LIST_DIRECT_RENDER_ENABLED=True`、`COMPARISON_JSON_DIRECT_RENDER_ENABLED=True`、`STAGE2_QUERY_AWARE_EVIDENCE_ENABLED=True`、`NLI_DEVICE="cuda"`；`PLAN_EXECUTE_ENABLED=False`、`RERANK_ENABLED=False`。
- **速度判讀不變**：最新三題 stability 平均 retrieval `6.7s`、總延遲 `547.3s`；主要瓶頸是 Stage 3 本地 LLM 生成，不是 retrieval。
- **維護性稽核已完成**：active tracked Python 共 58 檔、22,904 行；首要熱點為 `query_pipeline.py`（1,854 行）、`eval/judge.py`（1,195）、`citation_grounding.py`（1,062）與三個大型測試檔。拆分計畫、驗收與刪除候選見 `maintainability_refactor_spec.md`。

---

## 待實作（有詳細 spec）

### 1. maintainability-refactor — 模組與測試邊界整理（`maintainability_refactor_spec.md`）
**狀態：已稽核、待 v11 baseline 凍結後開始（2026-08-02）。** 第一階段只做等價重構：刪除 tracked backup/一次性 debug、合併 VL 腳本、把純 unit tests 從 `scripts/` 拆到可 discovery 的 `tests/`，再抽離 `query_pipeline.py` 的 structured rendering 與 stream/non-stream 共用 stage helper。
不得與品質 prompt、threshold、retrieval 或 schema 行為修改混在同一 commit。所有核心搬移先跑 offline tests + 保存 artifact replay，再由最小 focused eval 驗收。

### 2. pipeline_v4 — 分階段索引（`pipeline_v4_task_spec.md`）
**狀態：未開始（2026-08-02 複核）。** 現況：索引一條龍同步跑 VL（indexer.py:39-42），VL 失敗會卡住建索引。
目標：拆成 fast base-index（先可搜尋）+ 非阻塞 VL/摘要增量豐富化 + per-paper 狀態追蹤
（`text_index_ready`/`vl_pending`/`vl_partial`/`summary_ready`/`last_successful_build`）+ 安全增量重建。
**重要：這是下方「匯入健檢 Phase 2」的地基**——MVP 已可做 corpus health/dedup，但 per-paper 狀態仍需分階段索引支撐。

### 3. memory_redesign — 記憶模組重設計（`memory_redesign_spec.md`）
**狀態：未開始（2026-08-02 複核）。** 研究知識管理層（非問答 log）。三類 episodic/preference/work_state、
狀態生命週期、原子結論句、三機制（C 衝突守衛→A 快速觸發→B session 整合）。
屬 Tier 2；前置條件「穩定量尺」已於 2026-06 備齊（量尺三軸：檢索/忠實度/正確性）。

### 4. api-refactor — API 分層（`api-refactor-spec.md`）
**狀態：未開始（2026-08-02 複核）。** `api.py` 仍把 routes/schemas/session/injection/memory/orchestration 混在同一檔，
import 時還連帶觸發 main.py 全域初始化。目標：拆成薄 transport 層 + 清楚服務邊界 + 安全啟動。
2026-04 的 `api-refractor` PR 是較早的 status streaming/query pipeline 整理；目前這份 spec 在該 PR 之後建立，完成條件尚未達成。
（優先序最低——目前不是品質或延遲瓶頸。）

---

## Roadmap 與驗證紀錄（尚無獨立現行 spec）

### A. 可答性 gate — 已完成、產品路徑啟用
回答前判「檢索 chunk 是否真的**包含**答案，而非只是主題相關」；只相關→走誠實棄答。
根治 Q11 那種「有免責橫幅但仍編造具體數字」的過度延伸。

#### Phase 1 完成（2026-06-23）— 分類器建好 + 驗證通過（log-only，未接路由）
做法:`rag/answerability.py` `assess_answerability(question, knowledge_base)`→{verdict, reason}。判 Stage 3 蒸餾後的 KB（生成用的同一份）。用 LLM_MODEL（gemma4，與 Stage 3 同模型→不觸發 VRAM swap）。Stage 3.5 log-only 接進 `execute_structured_query`，flag `ANSWERABILITY_GATE_ENABLED`（預設關）。
**坑 1＝空回應**:gemma4 是 thinking 模型，判定任務開思考會把 num_predict 全燒在思考通道、response 吐空（done_reason=length, eval_count 滿但 response 空）→ **加 `think:False`（同 judge.py 對 qwen3 的修法）**。temperature 0.1（>0）。
**坑 2＝綜合題校準**:初版對比較/聚合題過嚴（要求「比較」現成在 facts 裡）→ 改 prompt 區分「底層輸入缺失（→NOT）」vs「只是最終綜合形式沒現成、但輸入齊全（→ANSWERABLE）」。
**驗證（Q01/05/06/08/12 代表題）**:Q12 三次全中 NOT_ANSWERABLE（假前提/真缺值）;Q01/05/06 ANSWERABLE;**Q08 ANSWERABLE↔NOT 隨 run 變動**——查 kb_chars 釐清:rich KB（5737 字含各路線產率）→ANSWERABLE 對、thin KB→NOT 也對。**gate 沒判錯，是忠實反映 KB**。
**意外發現＝Stage 3 蒸餾對大型比較題不穩定**:同題 Q08 有時蒸出含全部數字、有時壓成高層摘要丟數字;因生成用同一份 KB，薄 KB 時生成本來也答不好 → **gate 在薄 KB 判 NOT 不是誤殺，是正確抓到「這次蒸餾退化」**。gate 意外成為蒸餾品質偵測器。→ 蒸餾穩定性列為未來獨立槓桿。
**結論:Phase 1 通過，gate 訊號可信。**

#### Phase 2 完成（2026-06-23）— 三分分類器 + 路由接好兩條 pipeline（預設關，待 baseline_v5 才開）
**設計（Peter 選「高精度硬棄答 + 其餘軟警告」）**:分類器改**三分** ANSWERABLE/PARTIAL/NOT_ANSWERABLE（`gate_route()` in answerability.py）:
- ANSWERABLE → 正常
- PARTIAL（有素材但不完整/薄 KB）→ 照常生成 + 軟警告橫幅（`WEAK_NOTICE`）
- NOT_ANSWERABLE（底層資料真缺/假前提，高信心）→ **硬棄答**（`ABSTAIN_NOTICE`，跳 Stage 4-7）
**坑（翻譯順序）**:中文軟警告橫幅不能在 EN draft 階段 prepend（會被 Stage 7 翻譯攪亂）→ 改成**翻譯後才 prepend**。棄答橫幅本身中文、Stage 7 已跳過。
**驗證（eval_gate_route：Q12/01/08/06/10）**:
- **Q12 → 硬棄答完美**:答案只剩棄答橫幅（84 字）、零捏造、wall 488s（跳 4-7 較快）、**correctness 1.0**（judge 對 false_premise 誠實不作答給滿，比之前半答的 0.5 更高）;gate 理由還點破假前提「BNCT 靜脈輸注非口服」。
- **Q01/06/08/10 → 全 ANSWERABLE 正常作答、零誤殺**（Q08 此 run KB 4870 字 rich）。
- **PARTIAL 路由未被這 5 題觸發**——分類器已用探針驗過會吐 PARTIAL（三分都用對），路由僅「翻譯後 prepend 一行」低風險，但未見真實 PARTIAL 端到端。
**兩條 pipeline（非串流 execute_structured_query + 串流 _stream）都接了**。flag `ANSWERABILITY_GATE_ENABLED` 預設關。
**待辦＝baseline_v5（gate on 完整 12 題）**:確認全題組無誤殺硬棄答、觀察 PARTIAL 出現率與軟警告呈現，再決定是否預設開。順帶＝新 judge 的首條完整基準線。

#### baseline_v5 結果（2026-06-26）— gate 驗證通過，✅ 預設開
**新 judge + gate on 首條完整基準線**:correctness **0.833**、grounding 0.741、total **918s**（比 v4 1011s 還低，Q12 棄答省 Stage 4-7）、選擇 100%/覆蓋 97.5%。
**gate 全題組**:8 ANSWERABLE（Q01-06,09,10 正常）/ 2 PARTIAL（Q07,Q08 軟警告+作答）/ 1 硬棄答（Q12 corr 1.0）/ 1 None（Q11 rag_found=False 走既有 fallback）。
**三驗收全過**:① **零誤殺硬棄答**（11 in-corpus 無一被錯誤棄答，只 Q12 真假前提棄答）;② **PARTIAL 校準大勝**——Q07(corr 0.25)、Q08(corr 0.5) 正是全題組最弱兩題，gate 精準命中（Q07 理由「KB 沒解釋動態共價鍵如何作用於氟離子結合」屬實，figure_dependent）;③ Q12 硬棄答穩定 corr 1.0。
**意外好處＝gate 誠實訊號與真實正確性正相關**（PARTIAL 抓到最弱答案），順便當「低品質答案偵測器」。
**決定：gate 作為產品候選預設開。** baseline_v5 驗證時為 `ANSWERABILITY_GATE_ENABLED=True`；2026-08-02 產品 config 也已恢復為 `True`。**0.833 = 新 judge 基準線**，未來與此比，≤v4 是舊 judge 不可跨比。
**順帶暴露 robustness 標的**:Q07 corr 0.25（figure_dependent，KB 缺動態共價鍵機制——可能圖/scheme 內容沒抽到）= 真實答不好的題，接下來 robustness 工作的具體入口之一。

### B. 匯入健檢 — MVP 完成，Phase 2 pending
論文匯入後自動用它自己的摘要/標題生 query，確認索引回得出合理 chunk + grounding 跑得動，
抓出抽取失敗（如掃描檔 OCR 壞）的論文，不需 gold 標籤。**與 pipeline_v4 的 per-paper 狀態追蹤共用地基。**

#### MVP 完成（2026-06-26，spec＝ingestion_health_spec.md）
起因＝語料審計發現 31 篇有 4 項污染。`rag/corpus_health.py` + `main.py --health` + indexer 去重 hook。零 gold。
**四類檢查**:① **精確重複**（正規化全文 sha）→ **自動跳過**冗餘份（indexer `duplicate_skip_set`，既有索引由 orphan cleanup 順手清，不刪 PDF）;② **疑似近重複**（同 metadata title、內容 sha 不同）→ 只警告人工審查（不自動跳，避免丟內容;排除含 SI 成員的組＝主文+補充非近重複）;③ **SI 當獨立論文**（檔名 regex:大寫 SI 後綴避開 synthesis + supplement/supporting info）→ 旗標;④ **抽取健康**（text_len<500 或 garbage_ratio>0.05）→ 被動報告。
**審計實測（boron_bnct）**:精確重複 1 組（45464，自動跳過、保留與 gold 同名的 `41467_2024_Article_45464`）;近重複 1 組（Pinacol）;SI 2 篇;抽取 0（全正常）。
**生效**:下次 `main.py` 啟動，indexer 跳過 `s41467-024-45464-z (1)` + orphan 清其索引 → 修掉 Q10 那種「同篇當兩篇引用」的 live 污染（31→30 有效篇）。
**驗證（2026-06-26，eval_dedup_check，Q07/Q10，dedup 生效後）**:Q10 citation 完全收斂（`s41467-024-45464-z` 提及 0、全 32 次指向單一正規名）= 同篇當兩篇徹底修好;corr 0.75→1.0。**意外加碼**:dedup 騰出的選擇槽被真正不同的相關論文補上——Q07 從 `[45464, s41467...(1)]`（同篇佔兩槽）變 `[45464, Chemistry…Ono boroxine]`，corr 0.25→0.5。零回歸（兩題皆升）。確認 dedup 兩益:① citation 不分裂（確定性）② 騰槽讓相關論文進得來。
**Phase 2**:SI→主文出處綁定、自我查詢可答性（LLM/篇）、`--fix` 自動刪檔、health 寫入 metadata（pipeline_v4 地基）。皆未做（YAGNI/風險）。

### C. 延遲拆解 — retrieval 已改善，Stage 3 優化 pending
**狀態：核心拆解與主要修正已完成。** Retrieval timing 已拆成 Phase A（embed/vector/BM25）與 Phase B（子答案生成）；`STAGE2_LLM_SUBANSWERS_ENABLED=False` 後直接打包 evidence blocks，移除昂貴的逐題子答案 LLM。

`q08_atomic_evidence_render_r3_dedup` 該輪：retrieval `7.4s`、Phase A `7.4s`、Phase B `0.0s`；Stage 3 `375.2s`，其中一次 JSON repair 約 `180.8s`。最新三題 stability 的整體平均則是 retrieval `6.7s`、total `547.3s`。下一個速度槓桿仍是避免不必要 repair/生成，不是再壓 retrieval。

### D. Agentic RAG loop — 暫緩
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

### E. 生成品質修正 — 早期實驗已由 structured contract 主線取代
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

#### 第一輪進度（2026-06-23，eval_citfix）
- ✅ **citation 精確**（query_prompts 直引段：只標該事實自己的來源）→ 明確大勝，Q09 從「掛全 5 篇」→ 單篇來源。**修好出處透明支柱。**
- ✅ **跨論文比較**（比較題按機制分類別平鋪）→ Q09 質性改善。
- ✅ **reference 補強**（Q04 儲存條件、Q06 boronate ester）→ 量測變誠實。
- ✅ **selfcorrect UNVERIFIED→刪除 + entailment<0.2 gating**（`SELFCORRECT_ENTAIL_MAX`），flag 仍預設關。
- ⚠️ **未解張力**：atomic-completeness（列全部，救 Q01）vs 比較/精簡（Q08 1.0→0.5）。同 prompt 兩股力對撞，待調。下次完整 baseline 觀察 Q08。

#### baseline_v4 驗收（2026-06-23）
avg correctness 0.688→**0.667**（持平微降）。Q04 +0.25（reference#1 落地）、Q10 +0.25、**Q08 −0.5（真退步，兩步）**、Q12 −0.25。
citation 修正全題組站得住（無 bullet 掛 4–5 篇）＝結構性勝利，但 judge 不獎勵 → correctness 不漲。
**新發現＝重複入庫**：`41467_2024_Article_45464`≡`s41467-024-45464-z (1)`、`LAT1 ChemComm 2026`vs`2026SI`（正文/補充）→ B 匯入健檢應加 dedup（疑為 Q09 選擇命中 75% 肇因）。

#### Q08 張力 A/B（2026-06-23，eval_q08fix）— 生成端已解，殘留是 judge 假象
改 query_prompts 比較段：比較題的「完整性」＝每對象在每**指名面向**上交代（不是抄每步/每數值），數值只在關乎對比面向時才列。
結果：Q08 0.5→**0.5**（沒動）、Q01 0.75（護欄守住，完整性沒波及）、Q09 0.5（沒退步）。**零量測損失。**
**但讀答案＝生成端真的修好了**：q08fix 答案已按成本/可擴展性/同位素富集分段 + 跨文獻推論真的在對比路線（Turbo Grignard 87% vs Pd <48%、混合 100%ee vs 氫化 76–88%）。v4 缺的對比結構補上了。
**殘留 0.5＝judge/reference 假象**（第三次確認此鐵則）：judge system prompt 明寫「忽略額外有依據細節」(judge.py:21) 卻仍扣，理由「大量 reference 沒有的細節＋一段推測」。722 字 reference 太精簡、答案太豐富 → judge 緊張預設扣。**不補 Q08 reference**（它沒錯、只是高層次，補＝拿答案改考卷＝作弊）。
**決定**：保留 prompt 修正（結構真變好、零退步）。誠實修法＝**judge-prompt 強化**（別扣額外正確細節/明確標注的推測層），但那改全 12 題量尺 → **獨立 A/B**（下一步）。
#### judge-prompt 強化 A/B（2026-06-23）— ✅ 已採用
做法：re-judge 既有 baseline_v4 答案（同候選、只換 judge prompt → 完美隔離，~24 次呼叫、不跑 pipeline）。
強化版 judge：correctness 只管「對 reference 涵蓋＋不矛盾」；**額外細節 + 明確標注的推論/推測層 OUT OF SCOPE（不獎不罰）**——捏造偵測交給 grounding，非 judge 職責。拒答/假前提規則原封保留。
結果 avg **0.667→0.812**，全是定點上升、無全面灌水:
- 假象修正:Q04/Q06/Q08 →1.0（正確答案不再因額外細節/推測層被罰，Q08 已人工讀過確實好）、Q09 →0.75（部分涵蓋）、Q11 →1.0（正確拒答舊 judge 低估）。
- **鑑別力保留（關鍵證據）**:Q07 0.5（真漏「自發脫水成 boroxine」）、Q10 0.75、Q12 0.5（沒點破「BPA 口服」假前提）→ 真漏講/假前提仍被扣。
- 判讀:這是**更準的尺非更鬆的尺**（假象移除後的真值）。
**已套用 judge.py**（_JUDGE_SYSTEM/_RUBRIC 換強化版）。
**已知邊界**:捏造偵測現完全靠 grounding——若某句捏造既不在 reference、grounding 又沒抓到，correctness judge 不再兜底（可接受:reference 無法裁決超綱內容）。
→ **未來重跑 baseline 用新 judge 當基準**；舊分數（≤baseline_v4）是舊 judge，不可直接跨版本比 correctness。

#### Q08 atomic comparison milestone（2026-07-12 → 2026-08-01）— focused 與 stability 已通過

多輪 A/B 已把根因從 prompt 約束收斂成資料結構問題。現行比較題路徑使用 source-bound atomic evidence、獨立 comparison JSON validator 與 deterministic renderer，避免 Stage 4 把多篇來源壓成同一個句子。Planner 也會在所有入選論文已有 specific task 時移除冗餘 `ALL`，避免 evidence/prompt 翻倍造成 JSON 截斷。

`q08_atomic_evidence_render_r3_dedup` 結果：correctness `1.0`、grounding `1.0`、selection/coverage `100%`、Stage 4 LLM `0.0s`。Ollama metadata 證實兩次 Stage 3 generation 都是 `done_reason=stop`，不是 context 截斷。剩餘問題是首次 JSON 少了 validator 要求的 high-purity/isotopic-enrichment trade-off framing，因此多跑一次 repair；品質已通過，但速度仍可改善。

`q08_atomic_evidence_render_r4_stability` correctness 仍為 `1.0`，但 grounding 降至 `0.875`，不能視為穩定重現。Planner 產生 4 specific + 1 `ALL`，舊 guard 因尚有一篇未被 specific 覆蓋而保留 `ALL`，task builder 再將它展開到全部 5 篇，形成 9 tasks。Stage 3 原始 atomic JSON 內容完整，卻因 validator 只認字面 `high-purity` 而拒絕等價的 `high optical purity + 10B`；repair 在 14734 prompt tokens 後以 `done_reason=length` 截斷，遂回退 Stage 4/Corrector。Grounding 唯一失敗則源於 evidence clip 把原文 `normal boric acid` 截斷，後續補成 `normal boron`。現行修正讓 `ALL` 只補未覆蓋論文、validator 接受純度詞與同位素詞組合，並在 200 字內補齊關鍵句句尾。

後續 relation/renderer/citation-scope 修正已將 Q08 收斂為 deterministic source-close output。`q08_cost_literal_canonical_rootfix` 與 `q07_q08_q09_cost_canonical_stability` 的 Q08 correctness、grounding、translation 均為 `1.0`，`7/7` facts、`C0/U0`，Stage 3 一次完成且 Stage 4 direct render 未觸發 repair/fallback。單題問題已通過；是否能升為正式 baseline 仍由完整 12 題 v11 決定。

#### v10 → v11 structured-contract milestone（2026-07-27 → 2026-08-02）— stability 通過、完整回歸執行中

v10 已將 requirement-aware evidence selection、method fact direct render、comparison JSON validator、structured correctness/translation judge 串成同一條產品路徑。八題 smoke 的 correctness/grounding 明顯優於 v9 smoke，但後續 focused runs 證明「validator 接受」還必須等價於「renderer 實際輸出」，不能只檢查藏在 JSON metadata 的內容。

`baseline_v10_contract_rootfix_r4` 的 Q07/Q08/Q09 結果為 correctness `0.75/1.0/0.5`、grounding `1.0/0.833/0.714`。Q07、Q09 缺口都可由 Stage 2 與 Stage 3 artifact 重現，已修在 shared contract/renderer，而非再加 prompt：stability mapping 接受所有語義有效 witness；named mechanism 必須存在於可渲染的 `supporting_mechanisms`；含省略號的 mechanism evidence 回退完整 role claim；translation audit reason 受 JSON schema 長度約束。離線測試與實際 artifact replay 均通過，正式分數仍以 r5 新流程為準。

2026-08-01 的新流程已完成 Q07 precursor/water relation 兩輪穩定性驗證，並在最終 Q07/Q08/Q09 regression 同時取得 correctness、grounding、translation `1.0`。這代表 focused 與 stability 兩道 gate 已完成。`baseline_v11_structured_contract_full` 已於 2026-08-02 啟動；完成前不使用部分題目平均值宣稱整體品質。

---

## 建議推進序（2026-08-02）

1. **完成完整 12 題 baseline v11（執行中）**：驗收 correctness/translation judge coverage `12/12`、paper selection 無回歸，逐題人工檢查 correctness、Stage 2 recall、grounding、unsupported/conflict、repair/truncation/fallback；不只比較平均分。
2. **凍結品質基準**：若通過，提交完整 report、更新 README/PENDING checkpoint 並把 v11 定為下一輪 A/B 的固定 baseline；若失敗，只針對失敗題的 evidence contract/renderer/evaluator 根因修正。
3. **Maintainability M0**：依 `maintainability_refactor_spec.md` 清除 backup/一次性 debug、合併 VL CLI、拆分大型 unit tests。這一階段不改產品行為，也不需 AI pipeline。
4. **Stage 3 速度 + Maintainability M1**：量測 repair rate、prompt tokens、generation duration；先移除 deterministic normalizer 可處理的 repair，再抽離 structured rendering 與共用 stream/non-stream stage helper。不提高 `num_ctx`，不先調 retrieval。
5. **產品匯入主線**：pipeline_v4 分階段索引 → ingestion health Phase 2。
6. **後續維護與產品層**：在修改到對應模組前完成 judge/grounding/contract 的 M2/M3 邊界；其後 memory redesign。API refactor 保留但目前順位最低。

### 方向決定

維持 robustness/品質優先。Retrieval 已不是瓶頸；focused 與 stability 已完成，當前唯一品質 gate 是完整 12 題 v11。等價 refactor 不得與品質行為修改混在同一 commit，Plan-and-Execute/agentic loop 繼續延後，避免在核心流程縮小前增加更多分支。
