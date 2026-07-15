# Eval Report — `q02_q08_lexical_window_fix`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-16 00:26
- 題數：2

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.75 |
| 平均 grounding 分數 | 1.0 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 425.9s |
| 平均 planning 延遲 | 7.7s |
| 平均 retrieval 延遲 | 6.1s |
| └ Phase A embed/vector/BM25 | 6.1s |
| └ Phase B 子答案生成 | 0.0s |
| 平均 grounding 延遲 | 17.3s |
| └ 其中 NLI | 1.8s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ✅ | Q02 | single_paper | 100.0% | 100.0% | 1.0 | 337.5s | C0/U0 |
| ✅ | Q08 | cross_paper | 100.0% | 100.0% | 1.0 | 514.3s | C0/U0 |

## 逐題細節

### ✅ Q02 · single_paper

**問題**：What is the hybrid process used for the asymmetric synthesis of optically pure L-p-boronophenylalanine, and what are its key steps?

- detected_paper：`None`
- 選出論文：['bbb0683', 'CMDC-20-e202500059']
- gold_papers：['bbb0683']
- correctness candidate：`answer_for_judge`
- correctness：0.75　raw：4/5　judge：`structured_fact_audit_v1`
- judge reason：covered 5/6; missing F3
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：1.0
- 延遲：337.5s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> The synthesis of optically pure L-p-boronophenylalanine (BPA) employs a hybrid process involving enantioselective alkylation and subsequent enzymatic hydrolysis [bbb0683]. This route is based on an enantioselective alkylation of lithiated bislactimether with a protected boronic acid derivative followed by enzymatic steps to achieve optical purity [bbb0683].
>
> **Key Steps:**
> *   Enantioselective alkylation of lithiated (2R)-( - )-2,5-dihydro- 2-isopropyl-3,6-dimethoxypyrazine with 4-bromomethylbenzeneboronate yields adduct in a 74% e.e. [bbb0683].
> *   Stepwise treatment of the resulting adduct with hydrochloric acid gives L-BPA methyl ester [bbb0683].
> *   Hydrolysis of L-BPA methyl ester with chymotrypsin furnishes optically pure L-p-boronophenylalanine [bbb0683].
>
> ---
> 📋 **答案品質報告**
>
> ✅ **整體論文 …（完整內容見 JSON）

---

### ✅ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- correctness candidate：`answer_for_judge`
- correctness：N/A　raw：N/A　judge：`structured_fact_audit_v1`
- judge reason：invalid structured judge output: F3 evidence is not a verbatim candidate excerpt; F4 evidence is not a verbatim candidate excerpt; missing fact results: F3, F4
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：1.0
- 延遲：514.3s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> Comparison scaffold:
> - Route: `bbb0683` reports enantioselective alkylation followed by chymotrypsin-catalysed enzymatic hydrolysis, yielding optically pure L-BPA at high e.e. [bbb0683].
> - Review/comparison source: `CMDC-20-e202500059` reports that the synthesis of 4-borono-L-phenylalanine (L-BPA) has been approached through multiple routes [CMDC-20-e202500059].
> - Review dimensions: The review highlights limitations of each method regarding scalability, cost-effectiveness, and safety [CMDC-20-e202500059].
>
> Central trade-off (high-purity/isotopic enrichment versus scalability and cost-effectiveness):
> - High-purity/isotopic enrichment: Challenges exist in producing high-purity, isotopically enriched material (10B) for L-BPA [CMDC-20-e202500059].
> - Scalability: The hybrid synthesis method has …（完整內容見 JSON）

---
