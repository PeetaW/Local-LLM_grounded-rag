# Eval Report — `baseline_v10_contract_rootfix_r5`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-28 00:16
- 題數：2

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 1.0 |
| Correctness judge 覆蓋 | 2/2（N/A 0） |
| 平均翻譯忠實度（LLM-judge） | 1.0 |
| Translation judge 覆蓋 | 2/2（N/A 0） |
| 平均 grounding 分數 | 0.907 |
| 平均論文選擇命中率 | 100.0% |
| 平均 retriever candidate 覆蓋率 | 83.4% |
| 平均 Stage 2 evidence 覆蓋率 | 53.5% |
| 平均總延遲 | 818.3s |
| 平均 planning 延遲 | 27.9s |
| 平均 retrieval 延遲 | 6.3s |
| └ Phase A embed/vector/BM25 | 6.3s |
| └ Phase B 子答案生成 | 0.1s |
| 平均 grounding 延遲 | 28.8s |
| └ 其中 NLI | 4.8s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | correctness | 翻譯忠實度 | 選擇命中 | candidate recall | Stage 2 recall | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|-------------|------------|---------|------------------|----------------|-----------|------|------|
| ⚠️ | Q07 | figure_dependent | 1.0 | 1.0 | 100.0% | 66.7% | 50.0% | 1.0 | 475.4s | C0/U0 |
| ⚠️ | Q09 | cross_paper | 1.0 | 1.0 | 100.0% | 100.0% | 57.1% | 0.815 | 1161.2s | C0/U5 |

## 逐題細節

### ⚠️ Q07 · figure_dependent

**問題**：What is the water-stable boroxine structure reported, and what role do the dynamic covalent bonds play in its fluoride binding and hydrogel formation?

- detected_paper：`None`
- 選出論文：['41467_2024_Article_45464']
- gold_papers：['41467_2024_Article_45464']
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 8/8
- translation fidelity：1.0　raw：5/5　judge：`translation_fidelity_v2`
- translation reason：0 material and 0 minor semantic errors
- 論文選擇命中率：100.0%　candidate 覆蓋率：66.7%　Stage 2 evidence 覆蓋率：50.0%　grounding：1.0
- 延遲：475.4s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
>
> - 有趣的是，在接觸水後，二聚體 (dimer) 迅速轉化為具有優異 pH 穩定性和水兼容動態共價鍵 (water-compatible dynamic covalent bonds) 的 boroxine 結構 【Source: 41467_2024_Article_45464】。
> - 更令人感興趣的是，在室溫下接觸水後，HO‑PBA 二聚體 (dimers) 會迅速轉化為 HO‑PBA 三聚體–H2O 複合物 (HO‑PBA trimer–H2O complexes)（圖 1b），其中包含水穩定的 boroxine 結構，這已通過核磁共振光譜 (nuclear magnetic resonance (NMR) spectroscopy)、質譜 (mass spectrometry (MS))、UV-Raman 光譜 (UV–Raman spectrometry) 以及理論計算 (theoretical calculations) 證實 【Source: 41467_2024_Article_45464】。
> - 這一推測得到了以下事實的支持：如圖 7b 所示，該水凝膠 (hydrogel) 在酸性 (pH=2) 和鹼性 (pH=10) 溶液中均保持穩定 【Source: 41467_2024_Article_45464】。
> - 這些發現使得 boroxines 能在水相介質 (aqueous media) 中選擇性地識別氟離子 (fluoride ions, F−)，其親和力顯著強於廣泛使用的 F− 受體 phenylboronic acid (PBA) 【Source: 41467_2024_Article_45464】。
> - 水穩定 boroxine 結構中 B–O 鍵的動態特性 (dynamic nature) 通過各種 HO‑PBA 三聚 …（完整內容見 JSON）

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
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：57.1%　grounding：0.815
- 延遲：1161.2s　問題標記：{'conflicts': 0, 'unsupported': 5}

**答案預覽**：

> 根據提供的文獻，描述了三種針對 LAT1 的癌症治療策略：小分子抑制 (small molecule inhibition, JPH203)、自組裝肽抑制 (self-assembling peptide inhibition, L$\text{ff}$VLKK-4Phe)，以及利用 LAT1 介導的 p-硼苯丙氨酸 (p-Boronophenylalanine, BPA) 運輸的硼中子捕獲治療 (Boron Neutron Capture Therapy, BNCT)。這些策略在於其目標是阻斷轉運蛋白功能，還是利用該功能進行藥物遞送，有著根本性的區別。
>
> ### 小分子抑制策略 (Small Molecule Inhibition Strategy)
> 第一種策略涉及使用 JPH203，這是一種旨在直接抑制 LAT1 活性的小分子抑制劑 【1-s2.0-S1347861320300633-main】。JPH203 被鑑定為一種針對 LAT1 用於治療各種侵襲性癌症的首創 (first-in-class) 抗癌藥劑 【1-s2.0-S1347861320300633-main】。其機制涉及結合在 LAT1 的傳統底物結合口袋 (substrate-binding pocket) 內，利用親水性頭部基團 (hydrophilic head group, 苯丙氨酸骨架 phenylalanine backbone) 和疏水性尾部基團 (hydrophobic tail group, 5-amino-2-(3-aminophenyl) benzoxazole 骨架) 【s41421-024-00697-6】。具體的分子交互作用包括透過頭部基團與 TM1 和 TM6 形成氫鍵網絡 (hydrogen bond network)，透過氯原子 (chloride atom) 與 Tyr2 …（完整內容見 JSON）

---
