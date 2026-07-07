# Eval Report — `q02_q08_no_precursor_note_safety_tradeoff`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-07 00:32
- 題數：2

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.75 |
| 平均 grounding 分數 | 1.0 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 497.4s |
| 平均 planning 延遲 | 8.5s |
| 平均 retrieval 延遲 | 6.2s |
| └ Phase A embed/vector/BM25 | 6.2s |
| └ Phase B 子答案生成 | 0.0s |
| 平均 grounding 延遲 | 20.1s |
| └ 其中 NLI | 5.3s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ✅ | Q02 | single_paper | 100.0% | 100.0% | 1.0 | 504.4s | C0/U0 |
| ✅ | Q08 | cross_paper | 100.0% | 100.0% | 1.0 | 490.4s | C0/U0 |

## 逐題細節

### ✅ Q02 · single_paper

**問題**：What is the hybrid process used for the asymmetric synthesis of optically pure L-p-boronophenylalanine, and what are its key steps?

- detected_paper：`None`
- 選出論文：['bbb0683', 'CMDC-20-e202500059']
- gold_papers：['bbb0683']
- correctness candidate：`answer_for_judge`
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：1.0
- 延遲：504.4s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> 根據來源【bbb0683】提供的數據，光學純 L-p-boronophenylalanine (L-BPA) 的不對稱合成採用了一種混合化學-酶法製程 (hybrid chemo-enzymatic process)，其定義為包含對映選擇性烷基化 (enantioselective alkylation) 與酶水解 (enzymatic hydrolysis) 的特定關鍵步驟。
> 
> **混合製程概述**
> 該合成被描述為一種結合了**對映選擇性烷基化 (enantioselective alkylation)** 與隨後的**酶水解 (enzymatic hydrolysis)** 以生產 L-BPA 的混合製程 【bbb0683】。此路徑使用 **(2R)-(-)-2,5-dihydro-2-isopropyl-3,6-dimethoxypyrazine** (bislactimether) 作為手性輔助劑 (chiral auxiliary)，並使用受保護的 **4-bromomethylbenzeneboronic acid derivative** 作為親電試劑 (electrophile) 【bbb0683】。
> 
> **合成之關鍵步驟**
> 1.  **對映選擇性烷基化 (Enantioselective Alkylation)：** 立體中心 (stereocentre) 是透過鋰化的 **(2R)-(-)-2,5-dihydro-2-isopropyl-3,6-dimethoxypyrazine** 與受保護的 boronate derivative 之間的對映選擇性烷基化反應來建立的 【bbb0683】。此步驟產生一個加成物 (adduct)，其報導的對映體過量值 (enantiomeric excess, ee) 為 **74% ee**，而文中描述的優化條件顯示 …（完整內容見 JSON）

---

### ✅ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- correctness candidate：`answer_for_judge`
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：1.0
- 延遲：490.4s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> Comparison Scaffold:
> - **route**: 【bbb0683】報導了一種混合化學-酶合成法（hybrid chemo-enzymatic synthesis），涉及對映選擇性烷基化（enantioselective alkylation），隨後進行 chymotrypsin-catalysed enzymatic hydrolysis 以生產光學純的 L-BPA 【bbb0683】。
> - **review/comparison source**: 【CMDC-20-e202500059】回顧了多種合成方法，包括不對稱氫化（asymmetric hydrogenation）和保護策略（protection strategies），同時評估了與 BNCT 應用相關的 $^{10}\text{B}$ 同位素富集方法 【CMDC-20-e202500059】。
> 
> **Evidence Summary:**
> - 直接路徑證據詳細描述了一種混合過程，使用鋰化的 (2R)-(-)-2,5-dihydro-2-isopropyl-3,6-dimethoxypyrazine 進行對映選擇性烷基化（enantioselective alkylation），以及 chymotrypsin 水解以確保光學純度 【bbb0683】。
> - 回顧證據概述了涉及 Rh 催化劑（如 [(COD)Rh(R-Prophos)][BF4]）的替代路徑，並討論了 $^{10}\text{B}$ 富集選項以及不同保護/去保護序列（protection/deprotection sequences）的產率比較 【CMDC-20-e202500059】。
> 
> **Central trade-off:** 核心權衡在於平衡治療功效所需的高對映純度（enantiopurity）和同位素富集 ($^{10 …（完整內容見 JSON）

---
