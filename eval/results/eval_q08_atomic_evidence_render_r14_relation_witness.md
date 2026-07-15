# Eval Report — `q08_atomic_evidence_render_r14_relation_witness`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-15 03:44
- 題數：1

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.75 |
| 平均 grounding 分數 | 1.0 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 537.8s |
| 平均 planning 延遲 | 11.0s |
| 平均 retrieval 延遲 | 7.5s |
| └ Phase A embed/vector/BM25 | 7.4s |
| └ Phase B 子答案生成 | 0.0s |
| 平均 grounding 延遲 | 21.0s |
| └ 其中 NLI | 1.8s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ✅ | Q08 | cross_paper | 100.0% | 100.0% | 1.0 | 537.8s | C0/U0 |

## 逐題細節

### ✅ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- correctness candidate：`answer_for_judge`
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：1.0
- 延遲：537.8s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> Comparison scaffold:
> - Route: `bbb0683` reports enantioselective alkylation followed by chymotrypsin-catalysed enzymatic hydrolysis, yielding optically pure L-BPA [bbb0683].
> - Review/comparison source: `CMDC-20-e202500059` reports that the synthesis of 4-borono-L-phenylalanine (L-BPA) has been approached through multiple routes [CMDC-20-e202500059].
> - Review dimensions: The review highlights limitations of each method regarding cost-effectiveness, safety, and scalability [CMDC-20-e202500059].
>
> Central trade-off (high-purity/isotopic enrichment versus scalability and cost-effectiveness):
> - High-purity/isotopic enrichment: Producing high-purity, isotopically enriched 10B material is a challenge in L-BPA synthesis [CMDC-20-e202500059].
> - Scalability: The hybrid synthesis method has an advanta …（完整內容見 JSON）

---
