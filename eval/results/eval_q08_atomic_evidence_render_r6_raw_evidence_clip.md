# Eval Report — `q08_atomic_evidence_render_r6_raw_evidence_clip`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-13 23:40
- 題數：1

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 1.0 |
| 平均 grounding 分數 | 0.714 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 253.3s |
| 平均 planning 延遲 | 10.7s |
| 平均 retrieval 延遲 | 7.4s |
| └ Phase A embed/vector/BM25 | 7.4s |
| └ Phase B 子答案生成 | 0.0s |
| 平均 grounding 延遲 | 35.3s |
| └ 其中 NLI | 11.2s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ⚠️ | Q08 | cross_paper | 100.0% | 100.0% | 0.714 | 253.3s | C0/U2 |

## 逐題細節

### ⚠️ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- correctness candidate：`answer_for_judge`
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.714
- 延遲：253.3s　問題標記：{'conflicts': 0, 'unsupported': 2}

**答案預覽**：

> Comparison scaffold:
> - Route: `bbb0683` reports enantioselective alkylation followed by chymotrypsin-catalysed enzymatic hydrolysis, yielding optically pure L-BPA (optical purity determined to be 100% by an HPLC analysis; enzymatic hydrolysis yield of 79%) [bbb0683].
> - Review/comparison source: `CMDC-20-e202500059` Synthesis of L-BPA is complicated by the need for high purity and isotopic enrichment, with specific safety risks in late-stage deprotection [CMDC-20-e202500059].
> 
> Central trade-off:
> - High-purity/isotopic enrichment: Producing high-purity, isotopically enriched material is a challenge that has led to multiple synthetic routes. [CMDC-20-e202500059].
> - Scalability: The use of any oxidant on scale is inherently a process safety risk. [CMDC-20-e202500059].
> - Scalability: The hybrid …（完整內容見 JSON）

---
