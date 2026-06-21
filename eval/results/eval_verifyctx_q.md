# Eval Report — `verifyctx_q`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-06-22 00:46
- 題數：3

## 彙總

| 指標 | 值 |
|------|-----|
| 平均 grounding 分數 | 0.698 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 91.7% |
| 平均總延遲 | 808.2s |
| 平均 planning 延遲 | 7.6s |
| 平均 retrieval 延遲 | 280.6s |
| 平均 grounding 延遲 | 34.1s |
| └ 其中 NLI | 16.9s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ⚠️ | Q05 | single_paper | 100.0% | 100.0% | 0.611 | 729.6s | C0/U11 |
| ⚠️ | Q06 | multi_chunk | 100.0% | 75.0% | 0.714 | 943.4s | C0/U5 |
| ⚠️ | Q08 | cross_paper | 100.0% | 100.0% | 0.769 | 751.5s | C0/U5 |

## 逐題細節

### ⚠️ Q05 · single_paper

**問題**：What is the solvent-free green protocol for N-Boc protection of amines using picric acid as a catalyst, and what reaction conditions are used?

- detected_paper：`None`
- 選出論文：['1-s2.0-S2773223124000268-main']
- gold_papers：['1-s2.0-S2773223124000268-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.611
- 延遲：729.6s　問題標記：{'conflicts': 0, 'unsupported': 11}

**答案預覽**：

> ## 【論文直接依據】
> * 苦味酸被用作 Brønsted 酸催化劑或有機催化劑 【1-s2.0-S2773223124000268-main】。
> * 所使用的苦味酸量為 2 mol% 或 0.2 當量 【1-s2.0-S2773223124000268-main】。
> * 二叔丁基二碳酸酯 ((Boc)₂O / Boc 酐) 被用作保護劑 【1-s2.0-S2773223124000268-main】。
> * 反應在無溶劑（純物質）條件下進行 【1-s2.0-S2773223124000268-main】。
> * 反應在室溫 (rt) 下進行 【1-s2.0-S2773223124000268-main】。
> * 優化後的反應時間為 60 分鐘 【1-s2.0-S2773223124000268-main】。
> * 此方法的最高產率可達 98% 【1-s2.0-S2773223124000268-main】。
> * 在室溫下使用 0.2 當量催化劑反應 60 分鐘時，產率可達 95% 【1-s2.0-S2773223124000268-main】。
> * 此方法適用於芳香胺 【1-s2.0-S2773223124000268-main】。
> * 此方法適用於雜環芳香胺 【1-s2.0-S2773223124000268-main】。
> * 此方法適用於開鏈脂肪胺 【1-s2.0-S2773223124000268-main】。
> * 此方法適用於環狀脂肪胺 【1-s2.0-S2773223124000268-main】。
> * 在反應機制中，苦味酸活化 Boc 酐的羰基氧 【1-s2.0-S2773223124000268-main】。
> * 胺的氮原子提供一對孤對電子，攻擊 Boc 酐的羰基碳 【1-s2.0-S2773223124000268-main】。
> * 反應機制涉及質子取代 【1-s …（完整內容見 JSON）

---

### ⚠️ Q06 · multi_chunk

**問題**：How does poly(vinyl alcohol) enhance the therapeutic effect of 4-L-boronophenylalanine in neutron capture therapy of thoracic tumors? Explain the proposed mechanism with supporting data.

- detected_paper：`None`
- 選出論文：['1-s2.0-S0378517325007926-main', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'CMDC-20-e202500059']
- gold_papers：['1-s2.0-S0378517325007926-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：75.0%　grounding：0.714
- 延遲：943.4s　問題標記：{'conflicts': 0, 'unsupported': 5}

**答案預覽**：

> ## 【論文直接依據】
> 
> * 聚乙烯醇 (Poly(vinyl alcohol), PVA) 通過硼酸酯 (boronate esters) 與多個 BPA 分子相互作用，形成稱為 PVA-BPA 的複合物 [1-s2.0-S0378517325007926-main]。
> * PVA-BPA 保留了靶向 LAT1（大中性氨基酸轉運蛋白 1）所需的關鍵類苯丙氨酸 (phenylalanine-like) 結構 [1-s2.0-S0378517325007926-main]。
> * PVA-BPA 在與 LAT1 相互作用後，通過「LAT1 介導的內吞作用」(LAT1-mediated endocytosis) 被癌細胞攝取 [1-s2.0-S0378517325007926-main]。
> * 一旦進入癌細胞，PVA-BPA 會定位於內體/溶酶體 (endo-/lysosomes) 中 [1-s2.0-S0378517325007926-main]。
> * PVA-BPA 的特定攝取路徑和定位增加了細胞攝取的 BPA 數量 [1-s2.0-S0378517325007926-main]。
> * 此機制延緩了 BPA 從細胞質中的外排 (efflux) [1-s2.0-S0378517325007926-main]。
> * 外排的延緩延長了 BPA 在腫瘤內的滯留時間 [1-s2.0-S0378517325007926-main]。
> * PVA-BPA 在 1 hour 內能高效地在腫瘤中累積 [1-s2.0-S0378517325007926-main]。
> * PVA-BPA 在腫瘤中表現出長期滯留 [1-s2.0-S0378517325007926-main]。
> * PVA-BPA 在正常器官中的累積有限 [1-s2.0-S0378517325007926-main]。
> * 使用山梨 …（完整內容見 JSON）

---

### ⚠️ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.769
- 延遲：751.5s　問題標記：{'conflicts': 0, 'unsupported': 5}

**答案預覽**：

> ## 【論文直接依據】
> 
> * 【CMDC-20-e202500059】是一篇整合了專利文獻與同行評審合成方法，用於 4-borono-L-phenylalanine (L-BPA) 的綜述 [CMDC-20-e202500059]。
> * 【CMDC-20-e202500059】的目的在於系統性地對 L-BPA 的合成路徑進行分類、批判性分析與比較 [CMDC-20-e202500059]。
> * 【CMDC-20-e202500059】旨在協助設計具可擴展性 (scalable) 的製程 [CMDC-20-e202500059]。
> * 【CMDC-20-e202500059】旨在協助設計低成本的製程 [CMDC-20-e202500059]。
> * 【CMDC-20-e202500059】旨在協助設計符合 GMP 標準的製程 [CMDC-20-e202500059]。
> * 論文【bbb0683】開發了一種使用「混合製程」(hybrid process) 合成光學純 L-BPA 的方法 [bbb0683]。
> * 【bbb0683】中的混合製程結合了對映選擇性烷基化 (enantioselective alkylation) 與酶水解 (enzymatic hydrolysis) [bbb0683]。
> * 【bbb0683】提出的合成路徑與先前報導的方法相比，反應步驟較少 [bbb0683]。
> * 【bbb0683】提出的合成路徑比先前報導的方法具有更簡單的操作特性 [bbb0683]。
> * 【bbb0683】提出的合成路徑比先前報導的方法具有更簡單的後處理過程 [bbb0683]。
> * 論文【bbb0683】未討論同位素富集 (isotopic enrichment) 的問題 [bbb0683]。
> * 論文【bbb0683】未討論可擴展性 (scalability) 的問題 [b …（完整內容見 JSON）

---
