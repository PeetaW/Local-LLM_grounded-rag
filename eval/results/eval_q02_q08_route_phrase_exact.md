# Eval Report — `q02_q08_route_phrase_exact`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-07 02:36
- 題數：2

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.875 |
| 平均 grounding 分數 | 0.75 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 555.0s |
| 平均 planning 延遲 | 8.7s |
| 平均 retrieval 延遲 | 7.7s |
| └ Phase A embed/vector/BM25 | 7.7s |
| └ Phase B 子答案生成 | 0.0s |
| 平均 grounding 延遲 | 21.3s |
| └ 其中 NLI | 5.8s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ⚠️ | Q02 | single_paper | 100.0% | 100.0% | 0.5 | 516.4s | C0/U2 |
| ✅ | Q08 | cross_paper | 100.0% | 100.0% | 1.0 | 593.6s | C0/U0 |

## 逐題細節

### ⚠️ Q02 · single_paper

**問題**：What is the hybrid process used for the asymmetric synthesis of optically pure L-p-boronophenylalanine, and what are its key steps?

- detected_paper：`None`
- 選出論文：['bbb0683', 'CMDC-20-e202500059']
- gold_papers：['bbb0683']
- correctness candidate：`answer_for_judge`
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.5
- 延遲：516.4s　問題標記：{'conflicts': 0, 'unsupported': 2}

**答案預覽**：

> 光學純的 L-p-boronophenylalanine (L-BPA) 的不對稱合成 (Asymmetric Synthesis) 採用了一種結合了對映選擇性烷基化 (Enantioselective Alkylation) 與隨後酶促水解 (Enzymatic Hydrolysis) 的混合製程 (Hybrid Process) [bbb0683]。此化學-酶路徑 (Chemo-enzymatic Route) 由三個核心關鍵步驟定義，用以建立立體中心 (Stereocentre) 並實現光學純度 (Optical Purity) [bbb0683]。
> 
> 第一個關鍵步驟涉及鋰化的 (2R)-(-)-2,5-dihydro-2-isopropyl-3,6-dimethoxypyrazine 與 4-bromomethylbenzeneboronate 之間的對映選擇性烷基化反應 (Enantioselective Alkylation Reaction) [bbb0683]。此反應在 -78°C 的 THF 中進行，以產生對映體過量值 (Enantiomeric Excess, ee) 為 74% 的加成物 4 (Adduct 4) [bbb0683]。
> 
> 第二個關鍵步驟包括依次使用鹽酸 (Hydrochloric Acid) 處理加成物 4 (Adduct 4)，以產生 L-BPA methyl ester 6 [bbb0683]。此酸處理作為去保護與水解階段 (Deprotection and Hydrolysis Stage)，為中間體進行酶促拆分 (Enzymatic Resolution) 做準備 [bbb0683]。
> 
> 最後一個關鍵步驟採用 chymotrypsin-catalysed enzymatic hydrolysis (由胰凝乳蛋白酶催化的酶促水解) …（完整內容見 JSON）

---

### ✅ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- correctness candidate：`answer_for_judge`
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：1.0
- 延遲：593.6s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> 比較框架：
> *   路徑：【bbb0683】報導了一種混合化學-酶合成 (hybrid chemo-enzymatic synthesis)，利用對映選擇性烷基化 (enantioselective alkylation) 隨後進行 chymotrypsin-catalysed enzymatic hydrolysis（凝乳蛋白酶催化的酶水解）以生產光學純 L-BPA 【bbb0683】。
> *   評論/比較來源：【CMDC-20-e202500059】評估了多種合成方法，包括一種 Rh 催化劑不對稱氫化路徑 (Rh catalyst asymmetric hydrogenation route)，並探討了 $^{10}\text{B}$ 富集劑 (enriched agents) 的開發 【CMDC-20-e202500059】。
> 
> *   直接的混合過程涉及將鋰化吡嗪加合物 (lithiated pyrazine adduct) 經由酸水解 (acidic hydrolysis) 轉化為氨基酸甲酯 (amino-acid methyl ester)，隨後使用 chymotrypsin 進行酶法拆分 (enzymatic resolution) 【bbb0683】。
> *   評論來源詳細介紹了一條涉及二乙醇胺硼酸酯保護 (diethanolamine boronate protection) 和具有高對映體過量值 (enantiomeric excess) 的 Rh 催化劑不對稱氫化 (Rh catalyst asymmetric hydrogenation) 路徑，同時指出在治療應用中對 $^{10}\text{B}$ 同位素富集 (isotopic enrichment) 的特定要求 【CMDC-20-e202500059】。
> 
> 核心權衡 (Central trade …（完整內容見 JSON）

---
