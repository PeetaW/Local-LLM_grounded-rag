# Eval Report — `q08_isotope_cost_carry_r2`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-12 01:05
- 題數：1

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.75 |
| 平均 grounding 分數 | 0.667 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 509.2s |
| 平均 planning 延遲 | 10.3s |
| 平均 retrieval 延遲 | 7.4s |
| └ Phase A embed/vector/BM25 | 7.4s |
| └ Phase B 子答案生成 | 0.0s |
| 平均 grounding 延遲 | 30.3s |
| └ 其中 NLI | 10.4s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ⚠️ | Q08 | cross_paper | 100.0% | 100.0% | 0.667 | 509.2s | C0/U2 |

## 逐題細節

### ⚠️ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- correctness candidate：`answer_for_judge`
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.667
- 延遲：509.2s　問題標記：{'conflicts': 0, 'unsupported': 2}

**答案預覽**：

> Comparison scaffold:
> - **Review/Comparison Source:** CMDC-20-e202500059 reviews diverse syntheses of L-BPA, highlighting boronate esters and deprotection risks [CMDC-20-e202500059].
> - **Route:** bbb0683 reports an asymmetric synthesis using enantioselective alkylation followed by chymotrypsin-catalysed enzymatic hydrolysis [bbb0683].
>
> The hybrid process employs Lithiated (2R)(-)-2,5-dihydro-2-isopropyl-3,6-dimethoxypyrazine and 4-bromomethylbenzeneboronate to yield adducts hydrolyzed with chymotrypsin for optical purity [bbb0683]. This route is noted for having few reaction steps which aids cost-effectiveness when preparing isotopically enriched compounds where the major cost comes from the starting material [CMDC-20-e202500059, bbb0683].
>
> Diverse syntheses often utilize boronate esters wh …（完整內容見 JSON）

---
