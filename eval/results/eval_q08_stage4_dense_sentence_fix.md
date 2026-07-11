# Eval Report — `q08_stage4_dense_sentence_fix`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-08 21:21
- 題數：1

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.25 |
| 平均 grounding 分數 | 0.333 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 616.3s |
| 平均 planning 延遲 | 10.9s |
| 平均 retrieval 延遲 | 10.3s |
| └ Phase A embed/vector/BM25 | 10.3s |
| └ Phase B 子答案生成 | 0.0s |
| 平均 grounding 延遲 | 29.4s |
| └ 其中 NLI | 6.4s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ⚠️ | Q08 | cross_paper | 100.0% | 100.0% | 0.333 | 616.3s | C0/U2 |

## 逐題細節

### ⚠️ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- correctness candidate：`answer_for_judge`
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.333
- 延遲：616.3s　問題標記：{'conflicts': 0, 'unsupported': 2}

**答案預覽**：

> The provided input contains formatting instructions and constraints (e.g., TERM FIDELITY, METHOD KEY STEPS) but does not include any actual query results or paper data following the line "The following are query results for each sub-question:".
>
> Per the constraint **"Only use the content from the above data; do not add your own information"** and **"Every factual statement must be labeled with its source [Paper Name]"**, it is impossible to generate a synthesized answer comparing synthetic routes or citing specific papers without hallucinating sources that are not present in the input.
>
> Therefore, no comparison scaffold, evidence bullets, or central trade-off sentence can be accurately constructed based on the provided text alone. If you provide the query results containing the paper abst …（完整內容見 JSON）

---
