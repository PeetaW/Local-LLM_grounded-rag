# Eval Report — `q08_comparison_json_validator_dim_repair`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-08 07:40
- 題數：1

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.25 |
| 平均 grounding 分數 | 0.333 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 519.6s |
| 平均 planning 延遲 | 10.5s |
| 平均 retrieval 延遲 | 10.4s |
| └ Phase A embed/vector/BM25 | 10.4s |
| └ Phase B 子答案生成 | 0.0s |
| 平均 grounding 延遲 | 23.1s |
| └ 其中 NLI | 6.6s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ⚠️ | Q08 | cross_paper | 100.0% | 100.0% | 0.333 | 519.6s | C0/U2 |

## 逐題細節

### ⚠️ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- correctness candidate：`answer_for_judge`
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.333
- 延遲：519.6s　問題標記：{'conflicts': 0, 'unsupported': 2}

**答案預覽**：

> No relevant query results or paper data were provided in the input text to answer this question factually. The input contains formatting guidelines and constraints (TERM FIDELITY, METHOD KEY STEPS, COMPARISON SCAFFOLD) but lacks the actual search snippets, abstracts, or experimental details from specific papers required to generate a comparison with citations [Paper Name].
> 
> To comply with the constraint "Only use the content from the above data; do not add your own information," I cannot synthesize an answer regarding synthetic routes for 4-borono-L-phenylalanine without the source material. The provided text mentions specific chemical details (e.g., chymotrypsin, Schoellkopf-type auxiliary) within the context of formatting rules rather than as retrievable evidence from a paper list.
> 
> Plea …（完整內容見 JSON）

---
