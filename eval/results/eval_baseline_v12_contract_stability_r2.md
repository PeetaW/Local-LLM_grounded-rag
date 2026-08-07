# Eval Report — `baseline_v12_contract_stability_r2`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-08-07 02:54
- 題數：3

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 1.0 |
| Correctness judge 覆蓋 | 3/3（N/A 0） |
| 平均翻譯忠實度（LLM-judge） | 1.0 |
| Translation judge 覆蓋 | 3/3（N/A 0） |
| 平均 grounding 分數 | 1.0 |
| 平均論文選擇命中率 | 100.0% |
| 平均 retriever candidate 覆蓋率 | 100.0% |
| 平均 Stage 2 evidence 覆蓋率 | 77.4% |
| 平均總延遲 | 529.5s |
| 平均 planning 延遲 | 29.1s |
| 平均 retrieval 延遲 | 5.7s |
| └ Phase A embed/vector/BM25 | 5.7s |
| └ Phase B 子答案生成 | 0.1s |
| 平均 grounding 延遲 | 19.8s |
| └ 其中 NLI | 1.2s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | correctness | 翻譯忠實度 | 選擇命中 | candidate recall | Stage 2 recall | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|-------------|------------|---------|------------------|----------------|-----------|------|------|
| ✅ | Q04 | table_lookup | 1.0 | 1.0 | 100.0% | 100.0% | 100.0% | 1.0 | 374.9s | C0/U0 |
| ⚠️ | Q09 | cross_paper | 1.0 | 1.0 | 100.0% | 100.0% | 57.1% | 1.0 | 707.2s | C0/U0 |
| ✅ | Q10 | contradiction_check | 1.0 | 1.0 | 100.0% | 100.0% | 75.0% | 1.0 | 506.5s | C0/U0 |

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
- 延遲：374.9s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
>
> - BPA 與 tyrosine 的分離在 230 nm 處進行定量，其保留時間（retention times）分別為 5.3 和 4.5 min [Source: 1-s2.0-S0731708511003451-main]。
> - 樣本製備與降解分析：BPA 藥品（drug product）是透過將 BPA (100 mg/ml) 與 mannitol (110 mg/ml) 共同冷凍乾燥（freeze drying）產生，pH 值在 WFI 中調整至 8.0 ± 0.1，樣本瓶在黑暗中於 4, 25 和 40 °C 下孵育數月 [Source: 1-s2.0-S0731708511003451-main]。
> - 降解：作為原粉（raw powder），BPA 是穩定的，在 55 °C 下儲存 6 個月或在 40 °C 下儲存 12 個月時，未產生可檢測到的降解 [Source: 1-s2.0-S0731708511003451-main]。
> - BPA 強制降解試驗（forced degradation tests）使用溶解於 100 mM NaOH, 100 mM HCl 或 5% FeCl3 的 BPA 進行，且這些樣本在 55 °C 下孵育 24 h。6 mM H2O2 中的 BPA 溶液樣本則在 HPLC 分析前立即製備 [Source: 1-s2.0-S0731708511003451-main]。
> - 同時觀察到 BPA 在酸性及 FeCl3 溶液中具有穩定性（後者用於測試金屬催化降解 metal-catalysed degradation）[Source: 1-s2.0-S0731708511003451-main]。
> - 所有三種合成雜質（synthetic impurities）在 BPA 存在或不存在的情況下，均可在 0. …（完整內容見 JSON）

---

### ⚠️ Q09 · cross_paper

**問題**：Several papers target LAT1 for cancer therapy. What different therapeutic strategies targeting LAT1 are described across the literature, and how do they differ in mechanism?

- detected_paper：`None`
- 選出論文：['LAT1 ChemComm 2026', '1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 's41421-024-00697-6', 'cas0106-0279']
- gold_papers：['1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 'LAT1 ChemComm 2026', 's41421-024-00697-6']
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 7/7
- translation fidelity：1.0　raw：5/5　judge：`translation_fidelity_v2`
- translation reason：0 material and 0 minor semantic errors
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：57.1%　grounding：1.0
- 延遲：707.2s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> 比較框架：
> - 策略：【1-s2.0-S1347861320300633-main】報告了 JPH203 的抑制作用（預孵育 preincubation + 共孵育 co-incubation），並具有 LAT1 抑制效果的協同增強 [1-s2.0-S1347861320300633-main]。
> - 策略：【1-s2.0-S1936523321002047-main】報告了 JPH203 與 X-irradiation（X 射線照射）的聯合使用，透過 mTOR 下調 (downregulation) 增強細胞衰老 [1-s2.0-S1936523321002047-main]。
> - 策略：【LAT1 ChemComm 2026】報告了 LffVLKK-4Phe 自組裝肽 (self-assembling peptide) 處理，可抑制 LAT1 介導的氨基酸運輸並抑制增殖 [LAT1 ChemComm 2026]。
> - 策略依據：這可能與顯示 JPH203 作為非底物阻斷劑 (non-substrate blocker) 的報告相關，儘管作者表示其抑制模式是競爭性的 (competitive) [1-s2.0-S1347861320300633-main]。
> - 策略依據：我們證明了 X-irradiation（X 射線照射）增加了 A549 和 MIA Paca-2 細胞中經由 LAT1 的細胞中性氨基酸攝取，且 LAT1 抑制劑 JPH203 在最低毒性濃度下，能顯著增加癌細胞對輻射的敏感性 [1-s2.0-S1936523321002047-main]。
> - 策略依據：我們設計了一種偶聯至 L-phenylalanine (Phe) 基序的自組裝肽，作為 LAT1 配體 (ligand) (LffVLKK-4Phe) [LAT1 ChemComm 2026]。
> -  …（完整內容見 JSON）

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
- 延遲：506.5s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> 比較框架：
> - 策略：【Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation】報導將三個 boronic acid (硼酸) 單元併入一個柔性大環 (flexible macrocycle) 中（intramolecular boroxine formation，分子內 boroxine 形成），從而提高了 boroxines 對於水解 (hydrolysis) 的穩定性 【Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation】。
> - 策略：【41467_2024_Article_45464】報導 2-hydroxyphenylboronic acid (2-羥基苯硼酸) 的自發性脫水 (spontaneous dehydration)，隨後暴露於水中，導致形成具有優異 pH 穩定性和相容於水的動態共價鍵 (water-compatible dynamic covalent bonds) 的 boroxine 結構 【41467_2024_Article_45464】。
> - 機制：【Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation】報導透過 intramolecular boroxine formation (分子內 boroxine 形成) 產生的 Tricyclic boroxine 2  …（完整內容見 JSON）

---
