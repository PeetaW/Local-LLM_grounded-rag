# Eval Report — `q02_method_fact_direct_render`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-16 01:05
- 題數：1

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 1.0 |
| Correctness judge 覆蓋 | 1/1（N/A 0） |
| 平均 grounding 分數 | 1.0 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 100.0% |
| 平均總延遲 | 172.9s |
| 平均 planning 延遲 | 6.1s |
| 平均 retrieval 延遲 | 4.8s |
| └ Phase A embed/vector/BM25 | 4.8s |
| └ Phase B 子答案生成 | 0.0s |
| 平均 grounding 延遲 | 16.9s |
| └ 其中 NLI | 1.1s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ✅ | Q02 | single_paper | 100.0% | 100.0% | 1.0 | 172.9s | C0/U0 |

## 逐題細節

### ✅ Q02 · single_paper

**問題**：What is the hybrid process used for the asymmetric synthesis of optically pure L-p-boronophenylalanine, and what are its key steps?

- detected_paper：`None`
- 選出論文：['bbb0683', 'CMDC-20-e202500059']
- gold_papers：['bbb0683']
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 6/6
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：1.0
- 延遲：172.9s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> Method evidence:
> - Optically pure L-p-boronophenylalanine (L-BPA) was synthesized by a hybrid process involving enantioselective alkylation and subsequent enzymatic hydrolysis. [bbb0683]
> - Enantioselective alkylation was performed by reacting lithiated (2R)-(-)-2,5-dihydro-2-isopropyl-3,6-dimethoxypyrazine (bislactimether, 3) with protected 4-bromomethylbenzeneboronic acid derivative 2 (also referred to as 4-bromo-methylbenzeneboronate (2)) in THF at -78°C. [bbb0683]
> - The reaction between lithiated (2R)-(-)-2,5-dihydro-2-isopropyl-3,6-dimethoxypyrazine and 4-bromo-methylbenzeneboronate (2) yielded adduct 4 in a 74% e.e. [bbb0683]
> - Stepwise treatment of adduct 4 with hydrochloric acid produced L-BPA methyl ester 6. [bbb0683]
> - Optically pure L-BPA was furnished by the hydrolysis of L-BPA  …（完整內容見 JSON）

---
