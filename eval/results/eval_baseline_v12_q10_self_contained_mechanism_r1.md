# Eval Report — `baseline_v12_q10_self_contained_mechanism_r1`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-08-06 21:18
- 題數：1

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 1.0 |
| Correctness judge 覆蓋 | 1/1（N/A 0） |
| 平均翻譯忠實度（LLM-judge） | 1.0 |
| Translation judge 覆蓋 | 1/1（N/A 0） |
| 平均 grounding 分數 | 1.0 |
| 平均論文選擇命中率 | 100.0% |
| 平均 retriever candidate 覆蓋率 | 100.0% |
| 平均 Stage 2 evidence 覆蓋率 | 75.0% |
| 平均總延遲 | 474.4s |
| 平均 planning 延遲 | 30.5s |
| 平均 retrieval 延遲 | 4.9s |
| └ Phase A embed/vector/BM25 | 4.9s |
| └ Phase B 子答案生成 | 0.1s |
| 平均 grounding 延遲 | 12.8s |
| └ 其中 NLI | 0.0s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | correctness | 翻譯忠實度 | 選擇命中 | candidate recall | Stage 2 recall | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|-------------|------------|---------|------------------|----------------|-----------|------|------|
| ✅ | Q10 | contradiction_check | 1.0 | 1.0 | 100.0% | 100.0% | 75.0% | 1.0 | 474.4s | C0/U0 |

## 逐題細節

### ✅ Q10 · contradiction_check

**問題**：Do the papers agree on the hydrolytic stability of boroxine structures? Describe any agreements or differences between them.

- detected_paper：`None`
- 選出論文：['Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation', '41467_2024_Article_45464', 'a-method-for-the-deprotection-of-alkylpinacolyl-boronate-esters']
- gold_papers：['41467_2024_Article_45464', 'Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation']
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 4/4
- translation fidelity：1.0　raw：5/5　judge：`translation_fidelity_v2`
- translation reason：0 material and 0 minor semantic errors
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：75.0%　grounding：1.0
- 延遲：474.4s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> 比較框架：
> - 策略：【Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation】報導將三個硼酸 (boronic acid) 單元併入一個柔性大環 (flexible macrocycle) 中，從而提高了 boroxines 對於水解的穩定性【Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation】。
> - 策略：【41467_2024_Article_45464】報導 2-hydroxyphenylboronic acid 的自發脫水，隨後接觸水，從而形成具有優異 pH 穩定性和水兼容動態共價鍵 (water-compatible dynamic covalent bonds) 的 boroxine 結構【41467_2024_Article_45464】。
> - 機制：【Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation】報導由分子內 boroxine 形成 (intramolecular boroxine formation) 而得的三環 boroxine 2 (Tricyclic boroxine 2)，比 boroxine 4 具有更高的熵穩定性 (entropically stable) 且路易斯酸性 (Lewis acidic) 較低【Chemistry A European J - 2 …（完整內容見 JSON）

---
