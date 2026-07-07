# Eval Report — `q02_q08_stage3_safety_no_alt_values`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-07 01:12
- 題數：2

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.875 |
| 平均 grounding 分數 | 1.0 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 613.4s |
| 平均 planning 延遲 | 8.4s |
| 平均 retrieval 延遲 | 6.3s |
| └ Phase A embed/vector/BM25 | 6.2s |
| └ Phase B 子答案生成 | 0.0s |
| 平均 grounding 延遲 | 22.4s |
| └ 其中 NLI | 5.0s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ✅ | Q02 | single_paper | 100.0% | 100.0% | 1.0 | 567.0s | C0/U0 |
| ✅ | Q08 | cross_paper | 100.0% | 100.0% | 1.0 | 659.8s | C0/U0 |

## 逐題細節

### ✅ Q02 · single_paper

**問題**：What is the hybrid process used for the asymmetric synthesis of optically pure L-p-boronophenylalanine, and what are its key steps?

- detected_paper：`None`
- 選出論文：['bbb0683', 'CMDC-20-e202500059']
- gold_papers：['bbb0683']
- correctness candidate：`answer_for_judge`
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：1.0
- 延遲：567.0s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> 根據提供的數據，光學純的 L-p-boronophenylalanine (L-BPA) 是透過 【bbb0683】 所描述的化學-酶雜合製程 (hybrid chemo-enzymatic process) 合成的。此方法結合了用以建立立體化學的對映選擇性烷基化 (enantioselective alkylation) 與用於最終純化的酶水解 (enzymatic hydrolysis) 【bbb0683】。
> 
> **雜合製程之關鍵步驟：**
> 
> 1. **對映選擇性烷基化 (Enantioselective Alkylation)：** 此製程始於 lithiated (2R)-(-)-2,5-dihydro-2-isopropyl-3,6-dimethoxypyrazine 與 4-bromomethylbenzeneboronate 的保護衍生物 (protected derivative) 之間的對映選擇性烷基化反應 【bbb0683】。此步驟在 -78°C 的 THF 中進行，以產生對映體過量值 (enantiomeric excess, e.e.) 約 74% 的加成物 (adduct) 【bbb0683】。
> 2. **酸水解與去保護 (Acidic Hydrolysis and Deprotection)：** 所得加成物經由 hydrochloric acid 的逐步處理以去除輔助基 (auxiliary) 與保護基 (protecting groups)，從而產生 L-BPA methyl ester 【bbb0683】。
> 3. **酶水解 (Enzymatic Hydrolysis)：** 最後，由 chymotrypsin 催化氨基酸甲酯的 enzymatic hydrolysis 以產生光學純的 L-p-boronophenylalanine (L- …（完整內容見 JSON）

---

### ✅ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- correctness candidate：`answer_for_judge`
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：1.0
- 延遲：659.8s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> Comparison scaffold:
> *   Route: 【bbb0683】報告了一種利用對映選擇性烷基化 (enantioselective alkylation) 與酶水解 (enzymatic hydrolysis) 來合成光學純 L-BPA 的雜合化學-酶法製程 (hybrid chemo-enzymatic process) [bbb0683]。
> *   Review/comparison source: 【CMDC-20-e202500059】評估了多種化學方法，包括不對稱氫化 (asymmetric hydrogenation) 與專利路徑，旨在分析其步驟效率與成本效益 [CMDC-20-e202500059]。
> 
> Evidence bullets:
> *   Direct route evidence: 該雜合製程採用鋰化的 `(2R)-(-)-2,5-dihydro-2-isopropyl-3,6-dimethoxypyrazine` 進行對映選擇性烷基化 (enantioselective alkylation)，隨後經過 `chymotrypsin` 水解以產生光學純 L-BPA [bbb0683]。
> *   Review/comparison source evidence: 該評論將 Rh 催化不對稱氫化路徑與使用 `4-iodobenzyl bromide` 的專利路徑進行對比，指出後者透過減少步驟來避免硼保護基 (boron protecting groups)，從而提高成本效益與可擴展性 [CMDC-20-e202500059]。
> 
> Central trade-off: 透過雜合化學-酶法製程 (hybrid chemo-enzymatic processes) 或經由 Palladium Catalyzed Reaction 進行同位素 …（完整內容見 JSON）

---
