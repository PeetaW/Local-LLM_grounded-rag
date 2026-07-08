# Eval Report — `q02_q08_stage3_english_facts`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-08 02:14
- 題數：2

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.875 |
| 平均 grounding 分數 | 1.0 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 556.4s |
| 平均 planning 延遲 | 8.7s |
| 平均 retrieval 延遲 | 6.3s |
| └ Phase A embed/vector/BM25 | 6.3s |
| └ Phase B 子答案生成 | 0.0s |
| 平均 grounding 延遲 | 21.4s |
| └ 其中 NLI | 5.7s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ✅ | Q02 | single_paper | 100.0% | 100.0% | 1.0 | 474.4s | C0/U0 |
| ✅ | Q08 | cross_paper | 100.0% | 100.0% | 1.0 | 638.3s | C0/U0 |

## 逐題細節

### ✅ Q02 · single_paper

**問題**：What is the hybrid process used for the asymmetric synthesis of optically pure L-p-boronophenylalanine, and what are its key steps?

- detected_paper：`None`
- 選出論文：['bbb0683', 'CMDC-20-e202500059']
- gold_papers：['bbb0683']
- correctness candidate：`answer_for_judge`
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：1.0
- 延遲：474.4s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> 光學純 L-p-boronophenylalanine 的不對稱合成 (asymmetric synthesis) 利用了一種涉及對映選擇性烷基化 (enantioselective alkylation) 與隨後酶水解 (enzymatic hydrolysis) 的混合製程 (hybrid process) [bbb0683]。其核心關鍵步驟定義如下：
> 
> *   對映選擇性烷基化 (enantioselective alkylation) 是藉由使鋰化的 (2R)-(-)-2,5-dihydro-2-isopropyl-3,6-dimethoxypyrazine 與 bromide 2 在 -78°C 的 THF 中反應而進行 [bbb0683]。
> *   加成物 (adduct) 4 的對映體過量值為 74% e.e. [bbb0683]。
> *   將加成物 (adduct) 4 以鹽酸 (hydrochloric acid) 逐步處理，產生 L-BPA methyl ester 6 [bbb0683]。
> *   透過使用 chymotrypsin 水解 L-BPA methyl ester 6，製得光學純的 L-p-boronophenylalanine [bbb0683]。
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
- 延遲：638.3s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> Comparison scaffold (比較框架):
> - Route (路徑): Hybrid chemo-enzymatic process (混合化學-酶法製程) utilizing enantioselective alkylation (對映選擇性烷基化) followed by chymotrypsin-catalysed enzymatic hydrolysis (chymotrypsin 催化的酶水解) 【bbb0683】.
> - Review/Comparison Source (回顧/比較來源): Evaluates diverse syntheses including asymmetric hydrogenation (不對稱氫化) and patent routes (專利路徑), noting synthesis shortening impacts scalability (可擴展性) 【CMDC-20-e202500059】.
> 
> Evidence (證據):
> - The hybrid process employs enantioselective alkylation of lithiated (2R)-(-)-2,5-dihydro-2-isopropyl-3,6-dimethoxypyrazine with protected boronic acid derivative (受保護的硼酸衍生物) to yield an adduct (加成物) 【bbb0683】. Acidic hydrolysis (酸水解) produces the amino-acid methyl ester (氨基酸甲酯) 【bbb0683】. Chymotrypsin-catalysed enzymatic hydrolysis furnishes opt …（完整內容見 JSON）

---
