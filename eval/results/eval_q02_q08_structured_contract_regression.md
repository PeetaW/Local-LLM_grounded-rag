# Eval Report — `q02_q08_structured_contract_regression`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-16 01:22
- 題數：2

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 1.0 |
| Correctness judge 覆蓋 | 2/2（N/A 0） |
| 平均 grounding 分數 | 1.0 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 356.2s |
| 平均 planning 延遲 | 8.4s |
| 平均 retrieval 延遲 | 6.2s |
| └ Phase A embed/vector/BM25 | 6.2s |
| └ Phase B 子答案生成 | 0.0s |
| 平均 grounding 延遲 | 16.4s |
| └ 其中 NLI | 1.4s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ✅ | Q02 | single_paper | 100.0% | 100.0% | 1.0 | 166.6s | C0/U0 |
| ✅ | Q08 | cross_paper | 100.0% | 100.0% | 1.0 | 545.7s | C0/U0 |

## 逐題細節

### ✅ Q02 · single_paper

**問題**：What is the hybrid process used for the asymmetric synthesis of optically pure L-p-boronophenylalanine, and what are its key steps?

- detected_paper：`None`
- 選出論文：['bbb0683', 'CMDC-20-e202500059']
- gold_papers：['bbb0683']
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 6/6
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：1.0
- 延遲：166.6s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> Method evidence:
> - Optically pure L-p-boronophenylalanine (L-BPA) was synthesized by a hybrid process involving enantioselective alkylation and subsequent enzymatic hydrolysis. [bbb0683]
> - Enantioselective alkylation was performed by reacting lithiated (2R)-(-)-2,5-dihydro-2-isopropyl-3,6-dimethoxypyrazine (bislactimether, 3) with bromide 2 in THF at -78°C to yield adduct 4. [bbb0683]
> - Adduct 4 was yielded in a 74% e.e. [bbb0683]
> - Stepwise treatment of adduct 4 with hydrochloric acid produced L-BPA methyl ester 6. [bbb0683]
> - Optically pure L-BPA was furnished by hydrolyzing L-BPA methyl ester 6 with chymotrypsin. [bbb0683]
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
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 7/7
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：1.0
- 延遲：545.7s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> Comparison scaffold:
> - Route: `bbb0683` reports enantioselective alkylation followed by chymotrypsin-catalysed enzymatic hydrolysis, yielding optically pure L-BPA at high e.e. [bbb0683].
> - Review/comparison source: `CMDC-20-e202500059` reports that the synthesis of 4-borono-L-phenylalanine (L-BPA) has been approached through multiple routes [CMDC-20-e202500059].
> - Review dimensions: The review highlights limitations of each method regarding scalability, cost-effectiveness, and safety [CMDC-20-e202500059].
>
> Central trade-off (high-purity/isotopic enrichment versus scalability and cost-effectiveness):
> - High-purity/isotopic enrichment: Producing high-purity, isotopically enriched material (specifically 10B) for L-BPA is a significant challenge [CMDC-20-e202500059].
> - Scalability: The use of  …（完整內容見 JSON）

---
