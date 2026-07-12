# Eval Report — `q08_atomic_evidence_render_r4_stability`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-12 23:23
- 題數：1

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 1.0 |
| 平均 grounding 分數 | 0.875 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 804.9s |
| 平均 planning 延遲 | 10.2s |
| 平均 retrieval 延遲 | 10.3s |
| └ Phase A embed/vector/BM25 | 10.3s |
| └ Phase B 子答案生成 | 0.0s |
| 平均 grounding 延遲 | 32.9s |
| └ 其中 NLI | 12.0s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ✅ | Q08 | cross_paper | 100.0% | 100.0% | 0.875 | 804.9s | C0/U1 |

## 逐題細節

### ✅ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- correctness candidate：`answer_for_judge`
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.875
- 延遲：804.9s　問題標記：{'conflicts': 0, 'unsupported': 1}

**答案預覽**：

> Comparison scaffold:
> *   **Route:** bbb0683 reports a hybrid process involving enantioselective alkylation followed by chymotrypsin-catalysed enzymatic hydrolysis [bbb0683].
> *   **Review/Comparison Source:** CMDC-20-e202500059 evaluates diverse syntheses including cross-coupling and boronate esters across isotopic enrichment, scalability, cost-effectiveness, and safety dimensions [CMDC-20-e202500059].
> 
> *   The hybrid route achieves optically pure L-BPA through enantioselective alkylation to set the stereocentre followed by chymotrypsin-catalysed enzymatic hydrolysis, offering advantages in ease of performance and workup [bbb0683].
> *   Isotopic enrichment requires separating 10B from 11B via chemical exchange distillation before conversion to boronic esters or hydrolysis to 10B boric acid [ …（完整內容見 JSON）

---
