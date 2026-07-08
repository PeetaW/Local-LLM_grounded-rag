# Eval Report — `q02_q08_atomic_steps_safety_literal`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-08 01:52
- 題數：2

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.875 |
| 平均 grounding 分數 | 1.0 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 698.8s |
| 平均 planning 延遲 | 8.4s |
| 平均 retrieval 延遲 | 6.2s |
| └ Phase A embed/vector/BM25 | 6.2s |
| └ Phase B 子答案生成 | 0.0s |
| 平均 grounding 延遲 | 24.1s |
| └ 其中 NLI | 5.7s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ✅ | Q02 | single_paper | 100.0% | 100.0% | 1.0 | 605.7s | C0/U0 |
| ✅ | Q08 | cross_paper | 100.0% | 100.0% | 1.0 | 792.0s | C0/U0 |

## 逐題細節

### ✅ Q02 · single_paper

**問題**：What is the hybrid process used for the asymmetric synthesis of optically pure L-p-boronophenylalanine, and what are its key steps?

- detected_paper：`None`
- 選出論文：['bbb0683', 'CMDC-20-e202500059']
- gold_papers：['bbb0683']
- correctness candidate：`answer_for_judge`
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：1.0
- 延遲：605.7s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> 光學純的 L-p-boronophenylalanine (L-BPA) 是透過包含對映選擇性烷基化 (enantioselective alkylation) 與隨後的酶水解 (enzymatic hydrolysis) 的混合製程合成的【bbb0683】。此化學-酶路徑 (chemo-enzymatic route) 的核心關鍵步驟定義如下：
> 
> *   對映選擇性烷基化 (enantioselective alkylation) 涉及 lithiated (2R)-(-)-2,5-dihydro-2-isopropyl-3,6-dimethoxypyrazine 與 bromide 4-bromomethylbenzeneboronate 的反應，以產生加成物 4 (adduct 4)【bbb0683】。
> *   此烷基化步驟 (alkylation step) 建立了立體中心 (stereocentre)，其 ee 值為 74%【bbb0683】。
> *   使用鹽酸 (hydrochloric acid) 的分步處理將加成物 4 (adduct 4) 轉化為 L-BPA methyl ester 6【bbb0683】。
> *   Chymotrypsin 催化 L-BPA methyl ester 6 的酶水解 (enzymatic hydrolysis)，以產生光學純的 L-BPA【bbb0683】。
> 
> ---
> 📋 **答案品質報告**
> 
> ✅ **整體論文依據率**：100.0%　高（答案高度忠實於論文內容）
> 
> ✅ **所有陳述均有論文依據**
> 
> 
> <!-- grounding_score=1.000 -->
> ---

---

### ✅ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- correctness candidate：`answer_for_judge`
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：1.0
- 延遲：792.0s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> 比較框架 (Comparison scaffold)：
> *   路徑 (Route)：【bbb0683】報導了一種混合化學-酶法過程 (hybrid chemo-enzymatic process)，利用對映選擇性烷基化 (enantioselective alkylation) 隨後進行 chymotrypsin-catalysed enzymatic hydrolysis (凝血酶催化酶水解) 【bbb0683】。
> *   回顧/比較來源 (Review/comparison source)：【CMDC-20-e202500059】回顧並比較了含有 $^{10}\text{B}$ 的 L-BPA 合成方法，以評估 BNCT (硼中子捕獲治療) 應用 【CMDC-20-e202500059】。
> 
> 直接路徑依據 (Direct route evidence)：
> 該過程涉及對映選擇性烷基化 (enantioselective alkylation) 以設定立體中心 (stereocentre)，輔助基的酸性水解/去保護 (acidic hydrolysis/deprotection of the auxiliary) 以產生氨基酸甲酯 (amino-acid methyl ester)，以及 chymotrypsin-catalysed enzymatic hydrolysis (凝血酶催化酶水解) 以獲得光學純 L-BPA 【bbb0683】。
> 
> 回顧/比較來源依據 (Review/comparison source evidence)：
> BNCT 治療利用硼-10 ($^{10}\text{B}$) 核反應來破壞癌細胞，並對合成方法進行回顧以評估其應用 【CMDC-20-e202500059, cas0106-0279】。
> 
> 核心權衡 (Central trade-off …（完整內容見 JSON）

---
