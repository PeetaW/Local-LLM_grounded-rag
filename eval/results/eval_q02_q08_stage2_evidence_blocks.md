# Eval Report — `q02_q08_stage2_evidence_blocks`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-06 00:05
- 題數：2

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.75 |
| 平均 grounding 分數 | 0.55 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 511.4s |
| 平均 planning 延遲 | 8.6s |
| 平均 retrieval 延遲 | 7.7s |
| └ Phase A embed/vector/BM25 | 7.7s |
| └ Phase B 子答案生成 | 0.0s |
| 平均 grounding 延遲 | 20.7s |
| └ 其中 NLI | 5.5s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ⚠️ | Q02 | single_paper | 100.0% | 100.0% | 0.5 | 451.9s | C0/U1 |
| ⚠️ | Q08 | cross_paper | 100.0% | 100.0% | 0.6 | 571.0s | C0/U2 |

## 逐題細節

### ⚠️ Q02 · single_paper

**問題**：What is the hybrid process used for the asymmetric synthesis of optically pure L-p-boronophenylalanine, and what are its key steps?

- detected_paper：`None`
- 選出論文：['bbb0683', 'CMDC-20-e202500059']
- gold_papers：['bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.5
- 延遲：451.9s　問題標記：{'conflicts': 0, 'unsupported': 1}

**答案預覽**：

> 光學純的 L-p-boronophenylalanine (BPA) 的不對稱合成 (asymmetric synthesis) 採用了一種整合了化學與酶促轉化 (chemical and enzymatic transformations) 的混合製程 (hybrid process) [bbb0683]。此化學-酶路徑 (chemo-enzymatic route) 由三個核心關鍵步驟定義：用於設定立體中心 (stereocentre) 的對映選擇性烷基化 (enantioselective alkylation)、輔助基 (auxiliary) 的酸性水解/去保護 (acidic hydrolysis/deprotection) 以產生氨基酸甲酯 (amino-acid methyl ester)，以及由 胰凝乳蛋白酶 (chymotrypsin) 催化的酶促水解 (enzymatic hydrolysis) 以製得光學純的 L-BPA [bbb0683]。
> 
> **關鍵步驟 (Key Steps)：**
> 1.  **對映選擇性烷基化 (Enantioselective Alkylation)：** 此製程始於鋰化的 (2R)-(-)-2,5-dihydro-2-isopropyl-3,6-dimethoxypyrazine (bislactimether) 與受保護的 4-bromomethylbenzeneboronic acid 衍生物 (derivative) 的對映選擇性烷基化 (enantioselective alkylation) [bbb0683]。此反應產生一個加成物 (adduct)，為後續步驟建立了必要的立體化學 (stereochemistry) [bbb0683]。
> 2.  **酸性水解/去保護 (Acidic Hydrolysis/Depr …（完整內容見 JSON）

---

### ⚠️ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.6
- 延遲：571.0s　問題標記：{'conflicts': 0, 'unsupported': 2}

**答案預覽**：

> 比較框架：
> *   **route**: 【CMDC-20-e202500059】描述了一種涉及 Rh 催化不對稱氫化 (asymmetric hydrogenation) 和結晶 (crystallization) 以生產 L-BPA 的合成路徑 【CMDC-20-e202500059】。
> *   **route**: 【bbb0683】詳細介紹了一種化學-酶雜合製程 (hybrid chemo-enzymatic process)，利用對映選擇性烷基化 (enantioselective alkylation) 隨後進行酶水解 (enzymatic hydrolysis) 以產生光學純的 L-p-boronophenylalanine 【bbb0683】。
> 
> 證據：
> *   CMDC 路徑採用 $[(\text{COD})\text{Rh}(\text{R-Prophos})][\text{BF}_4]$ 進行不對稱氫化 (asymmetric hydrogenation)，在結晶後達到 >96% ee，而 【bbb0683】 的雜合製程則使用胰凝乳蛋白酶 (chymotrypsin) 催化的酶水解 (enzymatic hydrolysis)，將初始 74% ee 加成物 (adduct) 的光學純度進行精製 【CMDC-20-e202500059】, 【bbb0683】。
> *   所提供的摘錄均未明確量化金屬催化與酶法合成 L-BPA 之間在同位素富集步驟 (isotopic enrichment steps)、可擴展性指標 (scalability metrics) 或成本效益分析 (cost-effectiveness analysis) 方面的比較數據 【CMDC-20-e202500059】, 【bbb0683】。
> 
> 核心權衡：
> 雖然兩種路徑都透過不同的機制 …（完整內容見 JSON）

---
