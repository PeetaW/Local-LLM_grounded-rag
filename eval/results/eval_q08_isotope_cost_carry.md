# Eval Report — `q08_isotope_cost_carry`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-09 04:26
- 題數：1

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 1.0 |
| 平均 grounding 分數 | 1.0 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 720.4s |
| 平均 planning 延遲 | 9.9s |
| 平均 retrieval 延遲 | 10.2s |
| └ Phase A embed/vector/BM25 | 10.2s |
| └ Phase B 子答案生成 | 0.0s |
| 平均 grounding 延遲 | 23.6s |
| └ 其中 NLI | 8.0s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ✅ | Q08 | cross_paper | 100.0% | 100.0% | 1.0 | 720.4s | C0/U0 |

## 逐題細節

### ✅ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- correctness candidate：`answer_for_judge`
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：1.0
- 延遲：720.4s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> Comparison scaffold:
> *   Route: bbb0683 reports asymmetric synthesis of optically pure L-p-Boronophenylalanine by a hybrid process involving enantioselective alkylation followed by chymotrypsin-catalysed enzymatic hydrolysis [bbb0683].
> *   Review/Comparison Source: CMDC-20-e202500059 provides a comprehensive review and comparison of the evolution of diverse syntheses of 4-Borono-L-Phenylalanine for BNCT, evaluating borylation strategies and deprotection challenges [CMDC-20-e202500059].
>
> *   The hybrid process utilizes enantioselective alkylation of a bislactimether auxiliary with protected boronic acid derivative followed by chymotrypsin-catalysed enzymatic hydrolysis to furnish optically pure L-BPA [bbb0683].
> *   Isotopic enrichment requires expensive 10B starting materials which constitu …（完整內容見 JSON）

---
