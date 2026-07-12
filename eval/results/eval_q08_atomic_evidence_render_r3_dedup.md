# Eval Report — `q08_atomic_evidence_render_r3_dedup`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-12 22:57
- 題數：1

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 1.0 |
| 平均 grounding 分數 | 1.0 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 418.2s |
| 平均 planning 延遲 | 10.9s |
| 平均 retrieval 延遲 | 7.4s |
| └ Phase A embed/vector/BM25 | 7.4s |
| └ Phase B 子答案生成 | 0.0s |
| 平均 grounding 延遲 | 24.8s |
| └ 其中 NLI | 9.9s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ✅ | Q08 | cross_paper | 100.0% | 100.0% | 1.0 | 418.2s | C0/U0 |

## 逐題細節

### ✅ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- correctness candidate：`answer_for_judge`
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：1.0
- 延遲：418.2s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> Comparison scaffold:
> - Route: `bbb0683` reports enantioselective alkylation followed by chymotrypsin-catalysed enzymatic hydrolysis, yielding initial adduct in 74% e.e.; final L-BPA is optically pure (100% optical purity) with a 79% yield from methyl ester [bbb0683].
> - Review/comparison source: `CMDC-20-e202500059` Comparison of synthetic routes for L-BPA, noting the use of boronate esters and challenges in deprotection and safety [CMDC-20-e202500059].
>
> Central trade-off:
> - High-purity/isotopic enrichment: 10B is separated from 11B generally by chemical exchange distillation and then converted to boronic esters or hydrolyzed to 10B boric acid. [CMDC-20-e202500059].
> - Scalability: The present synthetic method for optically pure L-BPA has an advantage over the reported method in its ease of  …（完整內容見 JSON）

---
