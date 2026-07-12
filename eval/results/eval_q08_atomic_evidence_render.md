# Eval Report — `q08_atomic_evidence_render`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-12 01:34
- 題數：1

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.75 |
| 平均 grounding 分數 | 0.875 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 333.9s |
| 平均 planning 延遲 | 10.3s |
| 平均 retrieval 延遲 | 7.5s |
| └ Phase A embed/vector/BM25 | 7.4s |
| └ Phase B 子答案生成 | 0.0s |
| 平均 grounding 延遲 | 26.9s |
| └ 其中 NLI | 11.6s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ✅ | Q08 | cross_paper | 100.0% | 100.0% | 0.875 | 333.9s | C0/U1 |

## 逐題細節

### ✅ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- correctness candidate：`answer_for_judge`
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.875
- 延遲：333.9s　問題標記：{'conflicts': 0, 'unsupported': 1}

**答案預覽**：

> Comparison scaffold:
> - Route: `bbb0683` reports enantioselective alkylation followed by chymotrypsin-catalysed enzymatic hydrolysis, yielding 74% e.e. for adduct 4; optically pure L-BPA (100% optical purity) with a 79% yield from the hydrolysis of L-BPA methyl ester 6 [bbb0683].
> - Review/comparison source: `CMDC-20-e202500059` Various synthetic routes exist, including direct borylation via cross-coupling and methods using haloarene precursors such as 4-Iodo-L-phenylalanine, 1,4-Diiodobenzene, and 1-Bromo-4-methylbenzene [CMDC-20-e202500059].
> - Isotopic enrichment: For BNCT, it is essential for L-BPA to be 10B enriched [CMDC-20-e202500059].
> - Scalability: The present synthetic method for optically pure L-BPA has an advantage over the reported method in its ease of performance and workup, an …（完整內容見 JSON）

---
