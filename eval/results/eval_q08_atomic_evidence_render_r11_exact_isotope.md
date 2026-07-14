# Eval Report — `q08_atomic_evidence_render_r11_exact_isotope`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-15 02:23
- 題數：1

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.75 |
| 平均 grounding 分數 | 0.333 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 847.1s |
| 平均 planning 延遲 | 10.0s |
| 平均 retrieval 延遲 | 7.5s |
| └ Phase A embed/vector/BM25 | 7.5s |
| └ Phase B 子答案生成 | 0.0s |
| 平均 grounding 延遲 | 26.8s |
| └ 其中 NLI | 3.6s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ⚠️ | Q08 | cross_paper | 100.0% | 100.0% | 0.333 | 847.1s | C0/U4 |

## 逐題細節

### ⚠️ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- correctness candidate：`answer_for_judge`
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.333
- 延遲：847.1s　問題標記：{'conflicts': 0, 'unsupported': 4}

**答案預覽**：

> Comparison scaffold:
> *   Route [bbb0683]: Optically pure L-BPA is synthesized via enantioselective alkylation followed by chymotrypsin-catalysed enzymatic hydrolysis, offering advantages in ease of performance and workup compared to reported methods.
> *   Review/comparison source [CMDC-20-e202500059]: The synthesis lacks a consensus approach due to challenges in producing high-purity, isotopically enriched material alongside safety risks from oxidants or Pd contamination.
> 
> Evidence Summary:
> *   Direct route evidence indicates the hybrid process reduces reaction steps and simplifies workup compared to other methods [bbb0683].
> *   Review source evidence highlights that 10B separation via chemical exchange distillation incurs costs >1000 fold normal boric acid, while oxidant use on scale prese …（完整內容見 JSON）

---
