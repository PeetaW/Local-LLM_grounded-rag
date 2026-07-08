# Eval Report — `q08_stage4_rewrite_validator`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-08 07:58
- 題數：1

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.75 |
| 平均 grounding 分數 | 0.75 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 467.4s |
| 平均 planning 延遲 | 10.0s |
| 平均 retrieval 延遲 | 7.5s |
| └ Phase A embed/vector/BM25 | 7.5s |
| └ Phase B 子答案生成 | 0.0s |
| 平均 grounding 延遲 | 23.3s |
| └ 其中 NLI | 7.4s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ⚠️ | Q08 | cross_paper | 100.0% | 100.0% | 0.75 | 467.4s | C0/U1 |

## 逐題細節

### ⚠️ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- correctness candidate：`answer_for_judge`
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.75
- 延遲：467.4s　問題標記：{'conflicts': 0, 'unsupported': 1}

**答案預覽**：

> **Comparison scaffold:**
> * **Route:** bbb0683 reports an asymmetric synthesis of optically pure L-BPA using enantioselective alkylation followed by chymotrypsin-catalysed enzymatic hydrolysis [bbb0683].
> * **Review/comparison source:** CMDC-20-e202500059 reviews diverse syntheses of L-BPA, comparing various starting materials and protecting group strategies regarding scalability and cost-effectiveness [CMDC-20-e202500059].
> 
> The hybrid process utilizes enantioselective alkylation of (2R)-(-)-2,5-dihydro-2-isopropyl-3,6-dimethoxypyrazine and subsequent enzymatic hydrolysis with chymotrypsin at pH 5.0 to furnish optically pure L-BPA in a 79% yield [bbb0683]. Comparative analysis indicates that introducing the boronic acid after malonation shortens synthesis by eliminating the need for a boron  …（完整內容見 JSON）

---
