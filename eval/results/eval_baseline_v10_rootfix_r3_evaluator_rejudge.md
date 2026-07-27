# Eval Report — `baseline_v10_rootfix_r3_evaluator_rejudge`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-27 21:44
- 題數：2

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.875 |
| Correctness judge 覆蓋 | 2/2（N/A 0） |
| 平均翻譯忠實度（LLM-judge） | 0.75 |
| Translation judge 覆蓋 | 2/2（N/A 0） |
| 平均 grounding 分數 | 1.0 |
| 平均論文選擇命中率 | 100.0% |
| 平均 retriever candidate 覆蓋率 | 83.4% |
| 平均 Stage 2 evidence 覆蓋率 | 53.5% |
| 平均總延遲 | 571.2s |
| 平均 planning 延遲 | 27.4s |
| 平均 retrieval 延遲 | 6.2s |
| └ Phase A embed/vector/BM25 | 6.2s |
| └ Phase B 子答案生成 | 0.1s |
| 平均 grounding 延遲 | 18.2s |
| └ 其中 NLI | 0.9s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | correctness | 翻譯忠實度 | 選擇命中 | candidate recall | Stage 2 recall | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|-------------|------------|---------|------------------|----------------|-----------|------|------|
| ⚠️ | Q07 | figure_dependent | 1.0 | 0.5 | 100.0% | 66.7% | 50.0% | 1.0 | 448.7s | C0/U0 |
| ⚠️ | Q09 | cross_paper | 0.75 | 1.0 | 100.0% | 100.0% | 57.1% | 1.0 | 693.7s | C0/U0 |

## 逐題細節

### ⚠️ Q07 · figure_dependent

**問題**：What is the water-stable boroxine structure reported, and what role do the dynamic covalent bonds play in its fluoride binding and hydrogel formation?

- detected_paper：`None`
- 選出論文：['41467_2024_Article_45464']
- gold_papers：['41467_2024_Article_45464']
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 8/8
- translation fidelity：0.5　raw：3/5　judge：`translation_fidelity_scalar_fallback`
- translation reason：The translation contains a material semantic error in the fourth bullet point where "we achieves" (grammatically corrected to "achieve") is mistranslated as "我們實現了" (past tense) instead of reflecting the present/future intent or correcting the subject-verb agreement, and more critically, it omits th
- 論文選擇命中率：100.0%　candidate 覆蓋率：66.7%　Stage 2 evidence 覆蓋率：50.0%　grounding：1.0
- 延遲：448.7s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
>
> - 有趣的是，在接觸水後，二聚體迅速轉化為具有優異 pH 穩定性和水兼容動態共價鍵 (dynamic covalent bonds) 的 boroxine 結構 【Source: 41467_2024_Article_45464】。
> - 更令人感興趣的是，在室溫下接觸水後，HO‑PBA 二聚體 (dimers) 會迅速轉化為 HO‑PBA 三聚體-H2O 複合物 (trimer–H2O complexes)（圖 1b），其中包含水穩定的 boroxine 結構，這一點已通過核磁共振 (nuclear magnetic resonance, NMR) 光譜法、質譜 (mass spectrometry, MS)、UV–Raman 光譜法以及理論計算 (theoretical calculations) 證實 【Source: 41467_2024_Article_45464】。
> - 在此，我們報導了一種從 2-羥基苯硼酸 (2-hydroxyphenylboronic acid) 中發現的水穩定 boroxine 結構 【Source: 41467_2024_Article_45464】。
> - 憑藉這一發現，我們實現了 boroxines 在水相介質 (aqueous media) 中對 F− 的選擇性識別，其親和力顯著強於廣泛使用的 F− 受體 PBA，這為開發用於 F− 檢測與分離的先進材料和技術提供了途徑 【Source: 41467_2024_Article_45464】。
> - 這些發現使得 boroxines 在水相介質 (aqueous media) 中能選擇性地識別氟離子 (fluoride ions, F−)，其親和力顯著強於廣泛使用的 F− 受體苯硼酸 (phenylboronic acid, PBA)25–27 【Source:  …（完整內容見 JSON）

---

### ⚠️ Q09 · cross_paper

**問題**：Several papers target LAT1 for cancer therapy. What different therapeutic strategies targeting LAT1 are described across the literature, and how do they differ in mechanism?

- detected_paper：`None`
- 選出論文：['LAT1 ChemComm 2026', '1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 's41421-024-00697-6', 'cas0106-0279']
- gold_papers：['1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 'LAT1 ChemComm 2026', 's41421-024-00697-6']
- correctness candidate：`answer_for_judge`
- correctness：0.75　raw：4/5　judge：`structured_fact_audit_v1`
- judge reason：covered 6/7; missing F5
- translation fidelity：1.0　raw：5/5　judge：`translation_fidelity_v2`
- translation reason：0 material and 0 minor semantic errors
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：57.1%　grounding：1.0
- 延遲：693.7s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> 比較框架：
> - 策略：【1-s2.0-S1347861320300633-main】報告了 JPH203 對 LAT1 功能的抑制，表現為對 HT-29 細胞中 Na+-independent leucine uptake activity（不依賴於 $\text{Na}^+$ 的leucine攝取活性）的抑制 [1-s2.0-S1347861320300633-main]。
> - 策略：【1-s2.0-S1936523321002047-main】報告了 JPH203 與 X-irradiation（X射線照射）的聯合應用，透過 mTOR downregulation（mTOR 下調）增強 cellular senescence（細胞衰老） [1-s2.0-S1936523321002047-main]。
> - 策略：【LAT1 ChemComm 2026】報告了 LffVLKK-4Phe self-assembling peptide（自組裝肽）處理，表現為早期抑制 LAT1-mediated amino acid transport（LAT1 介導的氨基酸運輸），隨後在後期導致 cell membrane disruption（細胞膜破裂） [LAT1 ChemComm 2026]。
> - 策略：【cas0106-0279】報告了使用 BPA 的 Boron Neutron Capture Therapy (BNCT)（硼中子捕獲治療），透過 nuclear capture and fission reactions（核捕獲與裂變反應）選擇性地殺死含有 10B 的惡性細胞 [cas0106-0279]。
> - 機制：【s41421-024-00697-6】報告了與 JX075 相比，JPH203 誘導了不同的 conformational changes（構象變化），包括 T …（完整內容見 JSON）

---
