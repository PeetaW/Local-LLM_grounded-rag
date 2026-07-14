# Eval Report — `q08_atomic_evidence_render_r5_partial_dedup`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-13 00:06
- 題數：1

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 1.0 |
| 平均 grounding 分數 | 0.857 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 286.7s |
| 平均 planning 延遲 | 10.1s |
| 平均 retrieval 延遲 | 7.4s |
| └ Phase A embed/vector/BM25 | 7.4s |
| └ Phase B 子答案生成 | 0.0s |
| 平均 grounding 延遲 | 29.0s |
| └ 其中 NLI | 11.4s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ✅ | Q08 | cross_paper | 100.0% | 100.0% | 0.857 | 286.7s | C0/U1 |

## 逐題細節

### ✅ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- correctness candidate：`answer_for_judge`
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.857
- 延遲：286.7s　問題標記：{'conflicts': 0, 'unsupported': 1}

**答案預覽**：

> Comparison scaffold:
> - Route: `bbb0683` reports enantioselective alkylation followed by chymotrypsin-catalysed enzymatic hydrolysis, yielding optically pure L-BPA (optical purity 100%, final step yield 79%; intermediate adduct 4 yielded 74% e.e.) [bbb0683].
> - Review/comparison source: `CMDC-20-e202500059` Comparison of diverse syntheses including cross-coupling for direct borylation and the use of boronate esters (pinacol, 1,3-diphenyl-1,3-propanediol, 2,2-dimethyl-1,3-propanediol) [CMDC-20-e202500059].
> 
> Central trade-off for the production of high-purity, isotopically enriched L-BPA:
> - Isotopic enrichment: 10B is separated from the more common 11B generally by chemical exchange distillation [CMDC-20-e202500059].
> - Scalability: In greater than gram-scale reactions a small amount of boronic …（完整內容見 JSON）

---
