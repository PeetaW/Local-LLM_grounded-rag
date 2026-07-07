# Eval Report — `q02_q08_method_exact_terms_en_draft_judge`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-06 23:52
- 題數：2

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.875 |
| 平均 grounding 分數 | 0.834 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 508.9s |
| 平均 planning 延遲 | 8.5s |
| 平均 retrieval 延遲 | 7.7s |
| └ Phase A embed/vector/BM25 | 7.7s |
| └ Phase B 子答案生成 | 0.0s |
| 平均 grounding 延遲 | 17.8s |
| └ 其中 NLI | 4.5s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ⚠️ | Q02 | single_paper | 100.0% | 100.0% | 0.667 | 456.4s | C0/U1 |
| ✅ | Q08 | cross_paper | 100.0% | 100.0% | 1.0 | 561.5s | C0/U0 |

## 逐題細節

### ⚠️ Q02 · single_paper

**問題**：What is the hybrid process used for the asymmetric synthesis of optically pure L-p-boronophenylalanine, and what are its key steps?

- detected_paper：`None`
- 選出論文：['bbb0683', 'CMDC-20-e202500059']
- gold_papers：['bbb0683']
- correctness candidate：`answer_for_judge`
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.667
- 延遲：456.4s　問題標記：{'conflicts': 0, 'unsupported': 1}

**答案預覽**：

> 根據來自 **bbb0683** 的提供數據，光學純 L-p-boronophenylalanine 的不對稱合成採用了一種由特定化學-酶促步驟 (chemo-enzymatic steps) 定義的混合製程 [bbb0683]。其核心方法論省略了完整的實驗方案細節，並著重於如下的路徑定義轉化 (route-defining transformations)：
> 
> **混合製程概述 (Hybrid Process Overview)**
> 光學純 L-p-boronophenylalanine (BPA) 是透過一種包含對映選擇性烷基化 (enantioselective alkylation) 隨後進行酶促水解 (enzymatic hydrolysis) 的混合製程合成而得 [bbb0683]。此方法與其他替代方法有所區別，例如文中提到的烯丙基取代反應 (allylic substitution reaction)，其報告的產率為 72%，且 ee 值達到 86% [bbb0683]。
> 
> **混合製程之關鍵步驟 (Key Steps of the Hybrid Process)**
> 1. **對映選擇性烷基化 (Enantioselective Alkylation)：** 透過使 lithiated (2R)-(-)-2,5-dihydro-2-isopropyl-3,6-dimethoxypyrazine 與受保護的 4-bromomethylbenzeneboronic acid 衍生物反應來建立立體中心 [bbb0683]。此反應在 -78°C 下於 THF 中使用 n-BuLi，以產生對映體過量值 (enantiomeric excess, e.e.) 約為 74% 的 adduct 4 [bbb0683]。
> 2. **酸性水解與去保護 (Acidic Hydro …（完整內容見 JSON）

---

### ✅ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- correctness candidate：`answer_for_judge`
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：1.0
- 延遲：561.5s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> 比較框架：
> - 路線：【bbb0683】報導了一種合成 L-p-boronophenylalanine (L-BPA) 的混合製程，透過 enantioselective alkylation (對映選擇性烷基化)，隨後進行 chymotrypsin-catalysed enzymatic hydrolysis (由凝乳蛋白酶催化的酶水解) 以達到光學純度。
> - 評論/比較來源：【CMDC-20-e202500059】回顧並比較了 L-BPA 的合成方法，包括 $^{10}\text{B}$ isotopic enrichment (同位素富集)，詳細介紹了涉及 diethanolamine boronate 保護和 Rh catalyst asymmetric hydrogenation (不對稱氫化) 的方法。
> 
> 直接路線依據：該混合製程涉及使用受保護的 4-bromomethylbenzeneboronic acid derivative (衍生物) 對 lithiated bislactimether 進行 enantioselective alkylation 以設定 stereocentre (立體中心)，隨後經過 HCl 處理以產生 L-BPA methyl ester，最後進行 chymotrypsin hydrolysis (凝乳蛋白酶水解) 【bbb0683】。
> 評論/比較來源依據：該回顧比較了 $^{10}\text{B}$-enriched L-BPA 的合成方法，強調了使用 diethanolamine boronate 保護和使用 Rh catalysts 進行 asymmetric hydrogenation 以實現高 enantiomeric excess (對映體過量) 的路線 【CMDC-20-e202500059】。
> 
> 核心權衡：透過  …（完整內容見 JSON）

---
