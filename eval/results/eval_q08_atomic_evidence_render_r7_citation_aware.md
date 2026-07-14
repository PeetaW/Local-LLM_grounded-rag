# Eval Report — `q08_atomic_evidence_render_r7_citation_aware`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-14 00:16
- 題數：1

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.75 |
| 平均 grounding 分數 | 0.667 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 441.7s |
| 平均 planning 延遲 | 9.8s |
| 平均 retrieval 延遲 | 7.5s |
| └ Phase A embed/vector/BM25 | 7.5s |
| └ Phase B 子答案生成 | 0.0s |
| 平均 grounding 延遲 | 22.4s |
| └ 其中 NLI | 2.9s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ⚠️ | Q08 | cross_paper | 100.0% | 100.0% | 0.667 | 441.7s | C0/U1 |

## 逐題細節

### ⚠️ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- correctness candidate：`answer_for_judge`
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.667
- 延遲：441.7s　問題標記：{'conflicts': 0, 'unsupported': 1}

**答案預覽**：

> Comparison scaffold:
> *   Route source identifying bbb0683 utilizes enantioselective alkylation followed by chymotrypsin-catalysed enzymatic hydrolysis [bbb0683].
> *   Review/comparison source CMDC-20-e202500059 evaluates multiple routes regarding challenges in producing high-purity, isotopically enriched material [CMDC-20-e202500059].
> 
> Evidence bullets:
> 1.  The direct route produces optically pure L-BPA with intermediate adducts yielding in 74% e.e. before final enzymatic hydrolysis yields of 79% [bbb0683].
> 2.  Review evidence highlights that major costs come from isotope starting materials and oxidant use on scale presents process safety risks affecting scalability [CMDC-20-e202500059].
> 
> Central trade-off: The synthesis of L-BPA involves a tradeoff between achieving high purity and isotopi …（完整內容見 JSON）

---
