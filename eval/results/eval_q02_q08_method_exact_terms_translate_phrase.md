# Eval Report — `q02_q08_method_exact_terms_translate_phrase`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-06 23:26
- 題數：2

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.75 |
| 平均 grounding 分數 | 0.625 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 516.7s |
| 平均 planning 延遲 | 8.6s |
| 平均 retrieval 延遲 | 7.8s |
| └ Phase A embed/vector/BM25 | 7.8s |
| └ Phase B 子答案生成 | 0.0s |
| 平均 grounding 延遲 | 18.8s |
| └ 其中 NLI | 5.6s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ❌ | Q02 | single_paper | 100.0% | 100.0% | 0.25 | 455.4s | C0/U3 |
| ✅ | Q08 | cross_paper | 100.0% | 100.0% | 1.0 | 578.0s | C0/U0 |

## 逐題細節

### ❌ Q02 · single_paper

**問題**：What is the hybrid process used for the asymmetric synthesis of optically pure L-p-boronophenylalanine, and what are its key steps?

- detected_paper：`None`
- 選出論文：['bbb0683', 'CMDC-20-e202500059']
- gold_papers：['bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.25
- 延遲：455.4s　問題標記：{'conflicts': 0, 'unsupported': 3}

**答案預覽**：

> 光學純 L-p-boronophenylalanine (L-BPA) 的不對稱合成採用了一種將 enantioselective alkylation（對映選擇性烷基化）與隨後的 enzymatic hydrolysis（酶水解）相結合的混合製程 【bbb0683】。定義此 chemo-enzymatic route（化學-酶路徑）的核心關鍵步驟概述如下：
> 
> *   **Enantioselective Alkylation (對映選擇性烷基化)：** 此初始步驟透過使 lithiated (2R)-(-)-2,5-dihydro-2-isopropyl-3,6-dimethoxypyrazine 與 4-bromomethylbenzeneboronate 反應來建立 stereocentre（立體中心） 【bbb0683】。該反應在 THF 中於 -78°C 下使用 n-BuLi 進行，以產生一個 adduct（加成物），根據所述的特定合成條件，其報告的 enantiomeric excess (ee) 值從中間體的 74% 到最高 86% ee 不等 【bbb0683】。
> *   **Acidic Hydrolysis and Deprotection (酸水解與脫保護)：** 所得 adduct（加成物）經過 hydrochloric acid（鹽酸）的分步處理，去除 auxiliary protecting group（輔助保護基），從而產生 L-BPA methyl ester 【bbb0683】。這一非酶的酸性步驟將烷基化中間體轉化為適用於 enzymatic resolution（酶拆分）的 amino-acid methyl ester precursor（氨基酸甲酯前驅物） 【bbb0683】。
> *   **Enzymatic Hydrolysis ( …（完整內容見 JSON）

---

### ✅ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：1.0
- 延遲：578.0s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> 比較框架：
> - 路徑：【bbb0683】描述了一種混合製程 (hybrid process)，利用對映選擇性烷基化 (enantioselective alkylation) 隨後進行 chymotrypsin-catalysed enzymatic hydrolysis (胰凝乳蛋白酶催化的酶水解) 以生產光學純 L-BPA 【bbb0683】。
> - 回顧/比較來源：【CMDC-20-e202500059】回顧了 $^{10}\text{B}$ 富集 L-BPA 的合成方法，包括使用 Rh 催化劑的非對稱氫化 (asymmetric hydrogenation) 以及通過結晶來提高對映體過量值 (enantiomeric excess) 【CMDC-20-e202500059】。
> 
> 證據：
> - 直接路徑證據表明，該混合化學-酶法 (hybrid chemo-enzymatic approach) 通過 pyrazine 輔助基的烷基化來設定立體中心 (stereocentre)，並通過酶水解 (enzymatic hydrolysis) 解析光學純度，從而避免了大量的結晶過程 【bbb0683】。
> - 回顧來源證據強調了替代的化學路徑，其中涉及硼保護基（如 diethanolamine boronate (DABO)）以及針對 BNCT 應用中同位素富集要求的催化氫化步驟 (catalytic hydrogenation steps) 【CMDC-20-e202500059】。
> 
> 核心權衡：對高對映純度 (enantiopurity) 與 $^{10}\text{B}$ 同位素富集的追求，通常需要複雜的酶法或催化解析步驟 (enzymatic or catalytic resolution steps)，與較簡單的化學合成相比，需要在材料質量與製程可擴展性及成本效益之間取 …（完整內容見 JSON）

---
