# Eval Report — `q08_r14_stability_r3`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-15 23:05
- 題數：1

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 1.0 |
| 平均 grounding 分數 | 1.0 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 560.6s |
| 平均 planning 延遲 | 10.0s |
| 平均 retrieval 延遲 | 7.6s |
| └ Phase A embed/vector/BM25 | 7.5s |
| └ Phase B 子答案生成 | 0.0s |
| 平均 grounding 延遲 | 21.2s |
| └ 其中 NLI | 2.1s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ✅ | Q08 | cross_paper | 100.0% | 100.0% | 1.0 | 560.6s | C0/U0 |

## 逐題細節

### ✅ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 7/7
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：1.0
- 延遲：560.6s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> Comparison scaffold:
> - Route: `bbb0683` reports enantioselective alkylation followed by chymotrypsin-catalysed enzymatic hydrolysis, yielding optically pure L-BPA at high e.e. [bbb0683].
> - Review/comparison source: `CMDC-20-e202500059` reports that the synthesis of 4-borono-L-phenylalanine has been approached through multiple routes [CMDC-20-e202500059].
> - Review dimensions: The review highlights limitations of each method regarding scalability, cost-effectiveness, and safety [CMDC-20-e202500059].
>
> Central trade-off (high-purity/isotopic enrichment versus scalability and cost-effectiveness):
> - High-purity/isotopic enrichment: There are challenges in producing high-purity, isotopically enriched material for L-BPA (specifically involving boron-10/10B) [CMDC-20-e202500059].
> - Scalability: The …（完整內容見 JSON）

---
