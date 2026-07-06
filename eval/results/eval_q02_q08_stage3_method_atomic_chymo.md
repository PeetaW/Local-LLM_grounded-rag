# Eval Report — `q02_q08_stage3_method_atomic_chymo`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-06 22:56
- 題數：2

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.75 |
| 平均 grounding 分數 | 0.834 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 566.3s |
| 平均 planning 延遲 | 8.0s |
| 平均 retrieval 延遲 | 7.6s |
| └ Phase A embed/vector/BM25 | 7.6s |
| └ Phase B 子答案生成 | 0.0s |
| 平均 grounding 延遲 | 19.5s |
| └ 其中 NLI | 5.1s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ⚠️ | Q02 | single_paper | 100.0% | 100.0% | 0.667 | 483.1s | C0/U1 |
| ✅ | Q08 | cross_paper | 100.0% | 100.0% | 1.0 | 649.5s | C0/U0 |

## 逐題細節

### ⚠️ Q02 · single_paper

**問題**：What is the hybrid process used for the asymmetric synthesis of optically pure L-p-boronophenylalanine, and what are its key steps?

- detected_paper：`None`
- 選出論文：['bbb0683', 'CMDC-20-e202500059']
- gold_papers：['bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.667
- 延遲：483.1s　問題標記：{'conflicts': 0, 'unsupported': 1}

**答案預覽**：

> 光學純 (optically pure) L-p-boronophenylalanine (L-BPA) 的不對稱合成 (asymmetric synthesis) 採用了一種包含對映選擇性烷基化 (enantioselective alkylation) 及隨後酶水解 (enzymatic hydrolysis) 的混合製程 (hybrid process) 【bbb0683】。其核心關鍵步驟定義如下：
> 
> *   對映選擇性烷基化 (enantioselective alkylation) 使用 lithiated bislactimether (3) 與 bromide (2) 在 -78°C 的 THF 中反應以設定立體中心 (stereocentre) 【bbb0683】。此反應產生 e.e. 為 74% 的加成物 (adduct) 4 【bbb0683】。
> *   加成物 (adduct) 4 經過 hydrochloric acid 的分步處理 (stepwise treatment)，透過酸水解 (acidic hydrolysis) 和去保護 (deprotection) 產生 L-BPA methyl ester (6) 【bbb0683】。此過程將加成物轉化為酶拆分 (enzymatic resolution) 所需的特定底物 (substrate) 【bbb0683】。
> *   由 chymotrypsin 催化的酶水解 (enzymatic hydrolysis) 將 L-BPA methyl ester (6) 轉化為光學純 L-p-boronophenylalanine 【bbb0683】。此最終步驟確保了該混合路徑 (hybrid route) 中目標化合物 (target compound) 的高光學純度 (optical purity) 【 …（完整內容見 JSON）

---

### ✅ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：1.0
- 延遲：649.5s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> Comparison scaffold:
> - Route: 【bbb0683】報告了一種對映選擇性烷基化 (enantioselective alkylation)，隨後進行胰凝乳蛋白酶 (chymotrypsin) 催化的酶水解 (enzymatic hydrolysis)，以生產光學純的 L-p-boronophenylalanine 【bbb0683】。
> - Review/comparison source: 【CMDC-20-e202500059】回顧並比較了多種含有 $^{10}\text{B}$ 的 BNCT 藥劑 (BNCT agents) 合成方法，包括不對稱氫化方法 (asymmetric hydrogenation approaches) 【CMDC-20-e202500059】。
> 
> Evidence summary:
> - 直接合成路徑採用了一種混合化學-酶法製程 (hybrid chemo-enzymatic process)，其中對映選擇性烷基化 (enantioselective alkylation) 確定了立體中心 (stereocentre)，而酶水解 (enzymatic hydrolysis) 則確保了光學純度，且無需依賴外消旋體的拆分 (resolution of racemates) 【bbb0683】。
> - 比較分析強調，利用金屬催化劑如 `[(COD)Rh(R-Prophos)][BF4]` 或特定保護基策略 (protecting group strategies) 的路徑雖然解決了同位素富集 (isotopic enrichment) 問題，但涉及影響製程複雜度的多步序列 【CMDC-20-e202500059】。
> 
> Central trade-off:
> 透過複雜的化學-酶法或金屬催化方案 (chemo-enzymatic o …（完整內容見 JSON）

---
