# Eval Report — `q02_q08_post_q08_stability`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-15 23:50
- 題數：2

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.875 |
| 平均 grounding 分數 | 0.829 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 344.3s |
| 平均 planning 延遲 | 8.0s |
| 平均 retrieval 延遲 | 6.4s |
| └ Phase A embed/vector/BM25 | 6.3s |
| └ Phase B 子答案生成 | 0.1s |
| 平均 grounding 延遲 | 19.4s |
| └ 其中 NLI | 1.9s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ✅ | Q02 | single_paper | 100.0% | 100.0% | 0.8 | 394.1s | C0/U1 |
| ✅ | Q08 | cross_paper | 100.0% | 100.0% | 0.857 | 294.5s | C0/U1 |

## 逐題細節

### ✅ Q02 · single_paper

**問題**：What is the hybrid process used for the asymmetric synthesis of optically pure L-p-boronophenylalanine, and what are its key steps?

- detected_paper：`None`
- 選出論文：['bbb0683', 'CMDC-20-e202500059']
- gold_papers：['bbb0683']
- correctness candidate：`answer_for_judge`
- correctness：0.75　raw：4/5　judge：`legacy_holistic`
- judge reason：The candidate correctly identifies all key steps, reagents, conditions, and the specific auxiliary used, but omits the initial enantiomeric excess value (~74% e.e.) for the intermediate adduct which is explicitly stated in the reference as a detail of that step.
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.8
- 延遲：394.1s　問題標記：{'conflicts': 0, 'unsupported': 1}

**答案預覽**：

> The synthesis of optically pure L-p-boronophenylalanine (L-BPA) utilizes a hybrid process involving enantioselective alkylation and subsequent enzymatic hydrolysis [bbb0683]. This route is based on the reaction of an auxiliary with a protected boronic acid derivative followed by resolution steps to achieve optical purity [bbb0683].
>
> **Key Steps:**
> *   Enantioselective alkylation of lithiated (2R)-( - )-2,5-dihydro-2-isopropyl-3,6-dimethoxypyrazine with 4-bromomethylbenzeneboronate yielded adduct 4 in a 74% e.e. [bbb0683]
> *   Stepwise treatment of adduct 4 with hydrochloric acid produced L-BPA methyl ester 6 [bbb0683].
> *   Hydrolysis of L-BPA methyl ester 6 with chymotrypsin furnished optically pure L-BPA [bbb0683].
>
> The enantioselective alkylation was conducted in THF at -78°C to set the s …（完整內容見 JSON）

---

### ✅ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 7/7
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.857
- 延遲：294.5s　問題標記：{'conflicts': 0, 'unsupported': 1}

**答案預覽**：

> Comparison scaffold:
> - Route: `bbb0683` reports enantioselective alkylation followed by chymotrypsin-catalysed enzymatic hydrolysis, yielding optically pure L-BPA [bbb0683].
> - Review/comparison source: `CMDC-20-e202500059` reports that the synthesis of 4-borono-L-phenylalanine has been approached through multiple routes [CMDC-20-e202500059].
> - Review dimensions: The review highlights limitations of each method regarding cost-effectiveness, safety, and scalability [CMDC-20-e202500059].
>
> Central trade-off (high-purity/isotopic enrichment versus scalability and cost-effectiveness):
> - High-purity/isotopic enrichment: 10B is separated from 11B generally by chemical exchange distillation and then converted to boronic esters or hydrolyzed to 10B boric acid [CMDC-20-e202500059].
> - Scalability: The …（完整內容見 JSON）

---
