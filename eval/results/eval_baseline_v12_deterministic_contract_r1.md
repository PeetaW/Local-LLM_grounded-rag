# Eval Report — `baseline_v12_deterministic_contract_r1`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-08-05 20:02
- 題數：3

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 1.0 |
| Correctness judge 覆蓋 | 3/3（N/A 0） |
| 平均翻譯忠實度（LLM-judge） | 0.833 |
| Translation judge 覆蓋 | 3/3（N/A 0） |
| 平均 grounding 分數 | 0.891 |
| 平均論文選擇命中率 | 100.0% |
| 平均 retriever candidate 覆蓋率 | 100.0% |
| 平均 Stage 2 evidence 覆蓋率 | 77.4% |
| 平均總延遲 | 546.3s |
| 平均 planning 延遲 | 28.8s |
| 平均 retrieval 延遲 | 5.7s |
| └ Phase A embed/vector/BM25 | 5.7s |
| └ Phase B 子答案生成 | 0.0s |
| 平均 grounding 延遲 | 20.2s |
| └ 其中 NLI | 1.7s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | correctness | 翻譯忠實度 | 選擇命中 | candidate recall | Stage 2 recall | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|-------------|------------|---------|------------------|----------------|-----------|------|------|
| ✅ | Q04 | table_lookup | 1.0 | 1.0 | 100.0% | 100.0% | 100.0% | 1.0 | 383.2s | C0/U0 |
| ⚠️ | Q09 | cross_paper | 1.0 | 1.0 | 100.0% | 100.0% | 57.1% | 0.923 | 697.1s | C0/U1 |
| ⚠️ | Q10 | contradiction_check | 1.0 | 0.5 | 100.0% | 100.0% | 75.0% | 0.75 | 558.5s | C0/U1 |

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
- 延遲：383.2s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
>
> - BPA 與 tyrosine 的分離在 230 nm 處進行定量，其保留時間（retention times）分別為 5.3 和 4.5 min [Source: 1-s2.0-S0731708511003451-main]。
> - 在樣本製備與降解分析（degradation assays）中，BPA 藥品是透過將 BPA (100 mg/ml) 與 mannitol (110 mg/ml) 進行冷凍乾燥（freeze drying）製得，在 WFI 中將 pH 調節至 8.0 ± 0.1，樣本瓶在黑暗中於 4, 25 和 40 ◦C 下孵育數月 [Source: 1-s2.0-S0731708511003451-main]。
> - 關於降解（Degradation），BPA 作為原粉（raw powder）是穩定的，在 55 ◦C 下儲存 6 個月或在 40 ◦C 下儲存 12 個月時，未檢測到降解 [Source: 1-s2.0-S0731708511003451-main]。
> - BPA 強制降解試驗（forced degradation tests）使用溶解於 100 mM NaOH、100 mM HCl 或 5% FeCl3 中的 BPA 進行，且這些樣本在 55 ◦C 下孵育 24 h。6 mM H2O2 的 BPA 溶液樣本則在 HPLC 分析前立即製備 [Source: 1-s2.0-S0731708511003451-main]。
> - 同時觀察到 BPA 在酸性溶液和 FeCl3 溶液中（後者用於測試金屬催化降解 metal-catalysed degradation）是穩定的 [Source: 1-s2.0-S0731708511003451-main]。
> - 所有三種合成雜質（synthetic impurities）在 B …（完整內容見 JSON）

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
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：57.1%　grounding：0.923
- 延遲：697.1s　問題標記：{'conflicts': 0, 'unsupported': 1}

**答案預覽**：

> 比較框架：
> - 策略：【1-s2.0-S1347861320300633-main】報告了 JPH203 的預孵育（preincubation）與共同孵育（co-incubation）抑制作用，並具有協同抑制 LAT1 功能的效果 【1-s2.0-S1347861320300633-main】。
> - 策略：【1-s2.0-S1936523321002047-main】報告了 JPH203 與放射治療（radiation therapy）的聯合應用，透過 mTOR 下調（downregulation）與細胞衰老（cellular senescence）增強放射敏感性 【1-s2.0-S1936523321002047-main】。
> - 策略：【LAT1 ChemComm 2026】報告了一種針對 LAT1 的 LffVLKK-4Phe 自組裝肽（self-assembling peptide），透過多價交互作用（multivalent interactions）與細胞膜破壞抑制癌細胞增殖 【LAT1 ChemComm 2026】。
> - 策略依據：這可能與報告顯示 JPH203 作為非底物阻斷劑（non-substrate blocker）相關，儘管作者表示其抑制模式為競爭性（competitive） 【1-s2.0-S1347861320300633-main】。
> - 策略依據：我們證明了 X 射線照射（X-irradiation）增加了 A549 和 MIA Paca-2 細胞中經由 LAT1 的細胞中性氨基酸攝取，且在最低毒性濃度下，LAT1 抑制劑 JPH203 能顯著提高癌細胞對放射線的敏感性 【1-s2.0-S1936523321002047-main】。
> - 策略依據：我們設計了一種偶聯至 L-苯丙氨酸（L-phenylalanine, Phe）基序的自組裝肽，作 …（完整內容見 JSON）

---

### ⚠️ Q10 · contradiction_check

**問題**：Do the papers agree on the hydrolytic stability of boroxine structures? Describe any agreements or differences between them.

- detected_paper：`None`
- 選出論文：['Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation', '41467_2024_Article_45464', 'a-method-for-the-deprotection-of-alkylpinacolyl-boronate-esters']
- gold_papers：['41467_2024_Article_45464', 'Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation']
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 4/4
- translation fidelity：0.5　raw：3/5　judge：`translation_fidelity_v2`
- translation reason：1 material and 0 minor semantic errors; The source explicitly states the results 'clearly show' (清楚顯示), but this emphasis is omitted in T1 which only says 'reports that'. Additionally, S2 mentions a specific transformation process involving spontaneous dehydration followed by water exposure; while T2 captures parts of it, the nuance of 'd
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：75.0%　grounding：0.75
- 延遲：558.5s　問題標記：{'conflicts': 0, 'unsupported': 1}

**答案預覽**：

> 比較框架：
> - 策略：【Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation】報告指出，將三個硼酸 (boronic acid) 單位整合至一個柔性大環 (flexible macrocycle) 中，是提高 boroxines 對於水解穩定性的可行方法 【Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation】。
> - 策略：【41467_2024_Article_45464】報告指出，透過 2-hydroxyphenylboronic acid 的自發脫水形成二聚體 (dimer)，隨後在接觸水時發生轉化，該二聚體迅速轉化為具有優異 pH 穩定性和水兼容動態共價鍵 (water-compatible dynamic covalent bonds) 的 boroxine 結構 【41467_2024_Article_45464】。
> - 機制：【Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation】報告指出，由分子內 boroxine 形成 (intramolecular boroxine formation) 的三環 boroxine 2，比由分子間 boroxine 形成 (intermolecular boroxine formation) 的 boroxine 4 具有更高的熵穩定性 (ent …（完整內容見 JSON）

---
