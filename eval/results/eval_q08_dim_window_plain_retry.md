# Eval Report — `q08_dim_window_plain_retry`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-09 03:51
- 題數：1

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.75 |
| 平均 grounding 分數 | 1.0 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 853.9s |
| 平均 planning 延遲 | 11.0s |
| 平均 retrieval 延遲 | 10.6s |
| └ Phase A embed/vector/BM25 | 10.6s |
| └ Phase B 子答案生成 | 0.0s |
| 平均 grounding 延遲 | 24.5s |
| └ 其中 NLI | 8.2s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ✅ | Q08 | cross_paper | 100.0% | 100.0% | 1.0 | 853.9s | C0/U0 |

## 逐題細節

### ✅ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- correctness candidate：`answer_for_judge`
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：1.0
- 延遲：853.9s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> Comparison scaffold:
> - Route: `bbb0683` describes a hybrid process for L-BPA synthesis utilizing enantioselective alkylation followed by chymotrypsin-catalysed enzymatic hydrolysis [Source: bbb0683].
> - Review/Comparison Source: `CMDC-20-e202500059` compares synthetic methods focusing on scalability challenges with deprotection and cost-effectiveness regarding isotopic enrichment [Source: CMDC-20-e202500059].
>
> - Direct Route Evidence: Optically pure L-p-boronophenylalanine is synthesized via a hybrid process where lithiated (2R)-(-)-2,5-dihydro-2-isopropyl-3,6-dimethoxypyrazine reacts with 4-bromomethylbenzeneboronate to yield adduct in 74% e.e. [Source: bbb0683]. The intermediate is treated with hydrochloric acid to produce L-BPA methyl ester, which is then hydrolyzed with chymotrypsin at  …（完整內容見 JSON）

---
