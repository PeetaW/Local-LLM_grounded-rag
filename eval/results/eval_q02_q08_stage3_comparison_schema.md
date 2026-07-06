# Eval Report — `q02_q08_stage3_comparison_schema`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-06 01:00
- 題數：2

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.875 |
| 平均 grounding 分數 | 0.75 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 450.6s |
| 平均 planning 延遲 | 8.1s |
| 平均 retrieval 延遲 | 7.6s |
| └ Phase A embed/vector/BM25 | 7.6s |
| └ Phase B 子答案生成 | 0.0s |
| 平均 grounding 延遲 | 20.1s |
| └ 其中 NLI | 5.5s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ⚠️ | Q02 | single_paper | 100.0% | 100.0% | 0.5 | 415.4s | C0/U2 |
| ✅ | Q08 | cross_paper | 100.0% | 100.0% | 1.0 | 485.8s | C0/U0 |

## 逐題細節

### ⚠️ Q02 · single_paper

**問題**：What is the hybrid process used for the asymmetric synthesis of optically pure L-p-boronophenylalanine, and what are its key steps?

- detected_paper：`None`
- 選出論文：['bbb0683', 'CMDC-20-e202500059']
- gold_papers：['bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.5
- 延遲：415.4s　問題標記：{'conflicts': 0, 'unsupported': 2}

**答案預覽**：

> 光學純的 L-p-boronophenylalanine (L-BPA) 的不對稱合成 (asymmetric synthesis) 採用了一種將對映選擇性烷基化 (enantioselective alkylation) 與隨後的酶水解 (enzymatic hydrolysis) 相結合的混合製程 (hybrid process) [bbb0683]。此化學-酶路徑 (chemo-enzymatic route) 由三個核心關鍵步驟組成，旨在建立立體中心 (stereocentre) 並實現光學純度 (optical purity)。
> 
> 第一個關鍵步驟涉及在 -78°C 的 THF 中進行的對映選擇性烷基化 (enantioselective alkylation)，其中鋰化的 (2R)-(-)-2,5-dihydro-2-isopropyl-3,6-dimethoxypyrazine (bislactimether, 3) 與 bromide 2 反應生成 adduct 4 [bbb0683]。此反應設定了分子的立體中心 (stereocentre)，由此產生的 Adduct 4 在此步驟中具有報導的 e.e. 值 (e.e. value) [bbb0683]。
> 
> 第二個關鍵步驟是酸性水解與脫保護過程 (acidic hydrolysis and deprotection process)，其中 adduct 4 經過鹽酸 (hydrochloric acid) 的逐步處理，以產生 L-BPA methyl ester 6 [bbb0683]。此轉化移除了輔助基團 (auxiliary groups)，並產生了最終純化所需的氨基酸甲酯中間體 (amino-acid methyl ester intermediate)。
> 
> 第三個關鍵步驟利用由 chymotrypsin …（完整內容見 JSON）

---

### ✅ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：1.0
- 延遲：485.8s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> 比較框架：
> - 路徑：【bbb0683】描述了一種涉及對映選擇性烷基化 (enantioselective alkylation) 與酶水解 (enzymatic hydrolysis) 的混合化學-酶法製程 (hybrid chemo-enzymatic process)，用以生產光學純 L-BPA。
> - 評論/比較來源：【CMDC-20-e202500059】回顧了合成方法，包括使用 Rh 催化劑 (Rh catalysts) 的不對稱氫化路徑 (asymmetric hydrogenation route)，並評估其在 BNCT 中的應用。
> 
> - 該混合製程 (hybrid process) 利用 chymotrypsin 進行酶水解 (enzymatic hydrolysis)，以確保在氨基酸甲酯 (amino-acid methyl ester) 的酸性脫保護 (acidic deprotection) 後的光學純度 (optical purity) 【bbb0683】。
> - 一項被回顧的路徑採用 Rh 催化劑不對稱氫化 (Rh catalyst asymmetric hydrogenation)，隨後進行酰胺水解 (amide hydrolysis)，在生產 L-BPA 時實現了高對映體過量 (>96% ee)，同時指出 BNCT 需要 10B 富集劑 (10B-enriched agents) 【CMDC-20-e202500059】。
> 
> 核心權衡 (Central trade-off)：為了實現有效 BNCT 遞送 (BNCT delivery) 所需的高光學與同位素純度 (optical and isotopic purity)，必須採取涉及特殊催化劑或酶的複雜多步合成策略 (multi-step synthetic strategies)，這將治療功效  …（完整內容見 JSON）

---
