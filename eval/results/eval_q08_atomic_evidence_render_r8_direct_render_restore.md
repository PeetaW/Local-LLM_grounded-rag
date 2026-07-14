# Eval Report — `q08_atomic_evidence_render_r8_direct_render_restore`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-14 00:32
- 題數：1

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.75 |
| 平均 grounding 分數 | 0.5 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 272.3s |
| 平均 planning 延遲 | 10.7s |
| 平均 retrieval 延遲 | 7.6s |
| └ Phase A embed/vector/BM25 | 7.6s |
| └ Phase B 子答案生成 | 0.0s |
| 平均 grounding 延遲 | 22.8s |
| └ 其中 NLI | 3.7s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ⚠️ | Q08 | cross_paper | 100.0% | 100.0% | 0.5 | 272.3s | C0/U4 |

## 逐題細節

### ⚠️ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- correctness candidate：`answer_for_judge`
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.5
- 延遲：272.3s　問題標記：{'conflicts': 0, 'unsupported': 4}

**答案預覽**：

> Comparison scaffold:
> - Route: `bbb0683` reports enantioselective alkylation followed by chymotrypsin-catalysed enzymatic hydrolysis, yielding 74% e.e. for adduct 4; optically pure L-BPA (100% optical purity) with 79% yield [bbb0683].
> - Review/comparison source: `CMDC-20-e202500059` reports that the synthesis of 4-borono-L-phenylalanine (L-BPA) has been approached through multiple routes [CMDC-20-e202500059].
> 
> Central trade-off:
> - High-purity/isotopic enrichment: Producing high-purity, isotopically enriched material is a primary challenge in L-BPA synthesis [CMDC-20-e202500059].
> - Scalability: The use of any oxidant on scale is inherently a process safety risk [CMDC-20-e202500059].
> - Scalability: The hybrid synthesis method has an advantage in its ease of performance and workup, and few rea …（完整內容見 JSON）

---
