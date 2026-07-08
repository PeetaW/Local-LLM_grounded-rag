# Eval Report — `q02_q08_comparison_json`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-08 02:49
- 題數：2

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.75 |
| 平均 grounding 分數 | 1.0 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 533.8s |
| 平均 planning 延遲 | 8.7s |
| 平均 retrieval 延遲 | 7.7s |
| └ Phase A embed/vector/BM25 | 7.7s |
| └ Phase B 子答案生成 | 0.0s |
| 平均 grounding 延遲 | 19.9s |
| └ 其中 NLI | 5.0s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ✅ | Q02 | single_paper | 100.0% | 100.0% | 1.0 | 424.8s | C0/U0 |
| ✅ | Q08 | cross_paper | 100.0% | 100.0% | 1.0 | 642.8s | C0/U0 |

## 逐題細節

### ✅ Q02 · single_paper

**問題**：What is the hybrid process used for the asymmetric synthesis of optically pure L-p-boronophenylalanine, and what are its key steps?

- detected_paper：`None`
- 選出論文：['bbb0683', 'CMDC-20-e202500059']
- gold_papers：['bbb0683']
- correctness candidate：`answer_for_judge`
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：1.0
- 延遲：424.8s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> 光學純的 L-p-boronophenylalanine (L-BPA) 是透過一個包含對映選擇性烷基化 (enantioselective alkylation) 與隨後的酶水解 (enzymatic hydrolysis) 的混合製程 (hybrid process) 合成的 [bbb0683]。此不對稱合成 (asymmetric synthesis) 的核心關鍵步驟定義如下：
> 
> *   以鋰化的 (lithiated) (2R)-(-)-2,5-dihydro-2-isopropyl-3,6-dimethoxypyrazine 與 4-bromomethylbenzeneboronate 進行對映選擇性烷基化 (enantioselective alkylation)，以 74% e.e. 產出加成物 (adduct) 4 [bbb0683]。
> *   將加成物 (adduct) 4 以鹽酸 (hydrochloric acid) 逐步處理，產生 L-BPA methyl ester 6 [bbb0683]。
> *   透過使用 chymotrypsin 對 L-BPA methyl ester 6 進行水解 (hydrolysis)，獲得光學純的 L-BPA [bbb0683]。
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
- 延遲：642.8s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> 比較框架：
> - 【bbb0683】 (route)：透過包含 enantioselective alkylation (對映選擇性烷基化) 隨後進行 chymotrypsin-catalysed enzymatic hydrolysis (胰凝乳蛋白酶催化的酶水解) 的混合製程，實現光學純 L-p-Boronophenylalanine 的不對稱合成。
> - 【CMDC-20-e202500059】 (review/comparison source)：關於多樣化合成演進的全面回顧 (Comprehensive Review of the Evolution of the Diverse Syntheses...)，比較了包含 10B 及同位素富集維度的 BNCT 藥劑 L-BPA 的合成方法。
> 
> 證據：
> - 直接路徑證據表明，【bbb0683】利用 Lithiated (2R)-(-)-2,5-dihydro-2-isopropyl-3,6-dimethoxypyrazine 與 4-bromomethylbenzeneboronate 反應產生加成物 (adduct)，隨後用 hydrochloric acid (鹽酸) 處理，接著使用 chymotrypsin (胰凝乳蛋白酶) 【bbb0683】。在【CMDC-20-e202500059】中描述的路徑涉及使用 diethanolamine 進行保護 (protection)，透過 2-phenyloxazol-5(4H)-one 引入氨基酸骨架 (amino acid scaffold)，以及使用 Rh 催化劑 [(COD)Rh(R-Prophos)][BF4] 進行 asymmetric hydrogenation (不對稱氫化)，隨後進行 amide hydrolysis (醯胺水解) 【CMDC-20-e2025 …（完整內容見 JSON）

---
