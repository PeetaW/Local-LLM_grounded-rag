# Eval Report — `q02_structured_judge`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-16 00:07
- 題數：1

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 1.0 |
| 平均 grounding 分數 | 0.8 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 394.1s |
| 平均 planning 延遲 | 6.1s |
| 平均 retrieval 延遲 | 4.9s |
| └ Phase A embed/vector/BM25 | 4.9s |
| └ Phase B 子答案生成 | 0.0s |
| 平均 grounding 延遲 | 22.6s |
| └ 其中 NLI | 2.1s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ✅ | Q02 | single_paper | 100.0% | 100.0% | 0.8 | 394.1s | C0/U1 |

## 逐題細節

### ✅ Q02 · single_paper

**問題**：What is the hybrid process used for the asymmetric synthesis of optically pure L-p-boronophenylalanine, and what are its key steps?

- detected_paper：`None`
- 選出論文：['bbb0683', 'CMDC-20-e202500059']
- gold_papers：['bbb0683']
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 6/6
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
