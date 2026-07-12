# Eval Report — `q08_atomic_evidence_render_r2`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-12 22:28
- 題數：1

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.75 |
| 平均 grounding 分數 | 0.75 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 834.1s |
| 平均 planning 延遲 | 10.5s |
| 平均 retrieval 延遲 | 10.5s |
| └ Phase A embed/vector/BM25 | 10.5s |
| └ Phase B 子答案生成 | 0.0s |
| 平均 grounding 延遲 | 29.7s |
| └ 其中 NLI | 8.0s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ⚠️ | Q08 | cross_paper | 100.0% | 100.0% | 0.75 | 834.1s | C0/U1 |

## 逐題細節

### ⚠️ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- correctness candidate：`answer_for_judge`
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.75
- 延遲：834.1s　問題標記：{'conflicts': 0, 'unsupported': 1}

**答案預覽**：

> Comparison scaffold:
> - route: bbb0683 reports an asymmetric synthesis of optically pure L-p-Boronophenylalanine via enantioselective alkylation followed by chymotrypsin-catalysed enzymatic hydrolysis [bbb0683].
> - review/comparison source: CMDC-20-e202500059 provides a comprehensive review of diverse syntheses and evaluates application, noting cross-coupling feasibility and varying yields across methods [CMDC-20-e202500059].
>
> The hybrid process utilizes lithiated $(2\text{R})\text{-}(-)\text{-}2,5\text{-dihydro-2-isopropyl-3,6-dimethoxypyrazine}$ and protected $4\text{-bromomethylbenzeneboronic acid}$ derivative to yield optically pure L-BPA with high enantiomeric excess [bbb0683].
> The review highlights that substituting para-boronic acid gives yields around $68\%$ while other routes provid …（完整內容見 JSON）

---
