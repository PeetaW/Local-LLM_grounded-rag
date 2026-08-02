# Eval Report — `baseline_v11_evaluator_rejudge`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-08-02 20:29
- 題數：2

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 1.0 |
| Correctness judge 覆蓋 | 2/2（N/A 0） |
| 平均翻譯忠實度（LLM-judge） | 1.0 |
| Translation judge 覆蓋 | 2/2（N/A 0） |
| 平均 grounding 分數 | 1.0 |
| 平均論文選擇命中率 | 100.0% |
| 平均 retriever candidate 覆蓋率 | 100.0% |
| 平均 Stage 2 evidence 覆蓋率 | 87.5% |
| 平均總延遲 | 427.4s |
| 平均 planning 延遲 | 29.4s |
| 平均 retrieval 延遲 | 4.9s |
| └ Phase A embed/vector/BM25 | 4.8s |
| └ Phase B 子答案生成 | 0.0s |
| 平均 grounding 延遲 | 15.5s |
| └ 其中 NLI | 0.8s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | correctness | 翻譯忠實度 | 選擇命中 | candidate recall | Stage 2 recall | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|-------------|------------|---------|------------------|----------------|-----------|------|------|
| ✅ | Q04 | table_lookup | 1.0 | 1.0 | 100.0% | 100.0% | 100.0% | 1.0 | 364.1s | C0/U0 |
| ✅ | Q10 | contradiction_check | 1.0 | 1.0 | 100.0% | 100.0% | 75.0% | 1.0 | 490.6s | C0/U0 |

## 逐題細節

### ✅ Q04 · table_lookup

**問題**：What are the main degradation products and impurities of BPA identified by HPLC, and under which storage conditions do they form?

- detected_paper：`None`
- 選出論文：['1-s2.0-S0731708511003451-main', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water']
- gold_papers：['1-s2.0-S0731708511003451-main']
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 7/7
- translation fidelity：1.0　raw：5/5　judge：`translation_fidelity_v2`
- translation reason：0 material and 0 minor semantic errors
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：100.0%　grounding：1.0
- 延遲：364.1s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
>
> - BPA 與 tyrosine 的分離在 230 nm 處定量，其保留時間 (retention times) 分別為 5.3 和 4.5 min [Source: 1-s2.0-S0731708511003451-main]。
> - 在強制降解 (forced degradation) 與對照條件 (control conditions) 下，BPA 峰的峰純度值 (peak purity values) >0.997 [Source: 1-s2.0-S0731708511003451-main]。
> - 降解：作為原料粉末 (raw powder)，BPA 是穩定的，在 55 °C 下儲存 6 個月或在 40 °C 下儲存 12 個月時，未產生可檢測到的降解 (detectable degradation) [Source: 1-s2.0-S0731708511003451-main]。
> - 樣品製備與降解分析：BPA 藥品 (BPA drug product) 是透過將 BPA (100 mg/ml) 與 mannitol (110 mg/ml) 冷凍乾燥 (freeze drying) 製得，在 WFI 中將 pH 調節至 8.0 ± 0.1，樣品瓶在黑暗中於 4, 25 和 40 °C 下孵育 (incubated) 數月 [Source: 1-s2.0-S0731708511003451-main]。
> - BPA 強制降解 (forced degradation) 測試使用溶解在 100 mM NaOH, 100 mM HCl 或 5% FeCl3 中的 BPA 進行，且這些樣品在 55 °C 下孵育 24 h。6 mM H2O2 的 BPA 溶液樣品在 HPLC 分析 (HPLC analysis) 前立即製備 [Source: 1-s …（完整內容見 JSON）

---

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
- 延遲：490.6s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> Comparison scaffold:
> - Strategy: 【Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation】報導將三個 boronic acid (硼酸) 單元併入一個靈活的 macrocycle (大環分子) 中，藉此增強 boroxines (硼氧環) 對於水解 (hydrolysis) 的穩定性【Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation】。
> - Strategy: 【41467_2024_Article_45464】報導 2-hydroxyphenylboronic acid (2-羥基苯硼酸) 的自發性脫水 (spontaneous dehydration)，隨後接觸水，形成具有優異 pH 穩定性和水兼容動態共價鍵 (water-compatible dynamic covalent bonds) 的 boroxine (硼氧環) 結構【41467_2024_Article_45464】。
> - Mechanism: 【Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation】報導水解抗性 (Hydrolytic resistance) 通過熵因子 (entropic factors) 和 Lewis acidity (路易斯酸度) 的降低而得到增強【Chemistry …（完整內容見 JSON）

---
