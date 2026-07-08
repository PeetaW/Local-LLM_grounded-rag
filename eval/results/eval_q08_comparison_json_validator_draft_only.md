# Eval Report — `q08_comparison_json_validator_draft_only`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-08 03:43
- 題數：1

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.5 |
| 平均 grounding 分數 | 0.8 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 449.3s |
| 平均 planning 延遲 | 10.7s |
| 平均 retrieval 延遲 | 7.5s |
| └ Phase A embed/vector/BM25 | 7.5s |
| └ Phase B 子答案生成 | 0.0s |
| 平均 grounding 延遲 | 28.0s |
| └ 其中 NLI | 8.8s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ✅ | Q08 | cross_paper | 100.0% | 100.0% | 0.8 | 449.3s | C0/U1 |

## 逐題細節

### ✅ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- correctness candidate：`answer_for_judge`
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.8
- 延遲：449.3s　問題標記：{'conflicts': 0, 'unsupported': 1}

**答案預覽**：

> Comparison scaffold:
> - **Route**: bbb0683 reports an asymmetric synthesis of optically pure L-p-boronophenylalanine using enantioselective alkylation followed by chymotrypsin-catalysed enzymatic hydrolysis [bbb0683].
> - **Review/Comparison Source**: CMDC-20-e202500059 reviews the evolution of diverse syntheses, comparing approaches using boron protecting groups against routes that eliminate them to improve efficiency [CMDC-20-e202500059].
> 
> - The direct hybrid process employs enantioselective alkylation followed by chymotrypsin-catalysed enzymatic hydrolysis to furnish optically pure L-BPA without relying on boron protecting groups in the final steps [bbb0683].
> - Isotopic enrichment with boron-10 is required for Boron Neutron Capture Therapy applications, but retrieved evidence did not provi …（完整內容見 JSON）

---
