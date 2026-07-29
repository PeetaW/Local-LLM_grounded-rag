# Eval Report — `q07_q08_q09_rootfix_stability_r1`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-28 21:22
- 題數：3

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 1.0 |
| Correctness judge 覆蓋 | 3/3（N/A 0） |
| 平均翻譯忠實度（LLM-judge） | 1.0 |
| Translation judge 覆蓋 | 3/3（N/A 0） |
| 平均 grounding 分數 | 0.849 |
| 平均論文選擇命中率 | 100.0% |
| 平均 retriever candidate 覆蓋率 | 88.9% |
| 平均 Stage 2 evidence 覆蓋率 | 57.9% |
| 平均總延遲 | 966.3s |
| 平均 planning 延遲 | 29.7s |
| 平均 retrieval 延遲 | 6.7s |
| └ Phase A embed/vector/BM25 | 6.6s |
| └ Phase B 子答案生成 | 0.1s |
| 平均 grounding 延遲 | 24.0s |
| └ 其中 NLI | 1.8s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | correctness | 翻譯忠實度 | 選擇命中 | candidate recall | Stage 2 recall | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|-------------|------------|---------|------------------|----------------|-----------|------|------|
| ⚠️ | Q07 | figure_dependent | 1.0 | 1.0 | 100.0% | 66.7% | 50.0% | 1.0 | 521.9s | C0/U0 |
| ⚠️ | Q08 | cross_paper | 1.0 | 1.0 | 100.0% | 100.0% | 66.7% | 0.833 | 513.6s | C0/U1 |
| ⚠️ | Q09 | cross_paper | 1.0 | 1.0 | 100.0% | 100.0% | 57.1% | 0.714 | 1863.3s | C0/U2 |

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
- 延遲：521.9s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
>
> - 令人好奇的是，在接觸水後，二聚體 (dimer) 迅速轉化為具有優異 pH 穩定性和水兼容動態共價鍵 (water-compatible dynamic covalent bonds) 的 boroxine 結構【Source: 41467_2024_Article_45464】。
> - 更有趣的是，在室溫下接觸水後，HO‑PBA 二聚體 (dimers) 會迅速轉化為 HO‑PBA 三聚體–H2O 複合物 (trimer–H2O complexes)（圖 1b），其中包含水穩定的 boroxine 結構，這一點已通過核磁共振 (NMR) 光譜、質譜 (MS)、UV–Raman 光譜以及理論計算得到證實【Source: 41467_2024_Article_45464】。
> - 這一推測得到了以下事實的支持：如圖 7b 所示，該水凝膠 (hydrogel) 在酸性 (pH=2) 和鹼性 (pH=10) 溶液中均保持穩定【Source: 41467_2024_Article_45464】。
> - 這些發現使得 boroxine 能在水相介質中選擇性地識別氟離子 (F−)，其親和力顯著強於廣泛使用的 F− 受體 phenylboronic acid (PBA)25–27【Source: 41467_2024_Article_45464】。
> - 水穩定 boroxine 結構中 B–O 鍵的動態特性 (dynamic nature) 可由各種 HO‑PBA 三聚體結構之間的快速交換來證明【Source: 41467_2024_Article_45464】。
> - 此外，我們開發了一種由水穩定 boroxine 交聯的水凝膠 (hydrogel)，由於 boroxine 結構具有優異的 pH 穩定性和動態特性，該水凝膠表現出高酸鹼穩定性和可逆的凝膠-溶膠轉變  …（完整內容見 JSON）

---

### ⚠️ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 7/7
- translation fidelity：1.0　raw：5/5　judge：`translation_fidelity_v2`
- translation reason：0 material and 0 minor semantic errors
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：66.7%　grounding：0.833
- 延遲：513.6s　問題標記：{'conflicts': 0, 'unsupported': 1}

**答案預覽**：

> ⚠️ **資料可能不足**：檢索內容僅部分涵蓋此問題，以下回答可能不完整，請謹慎參考並自行查證。
>
> 比較框架：
> - 路線：【bbb0683】報導了對映選擇性烷基化 (enantioselective alkylation) 以及隨後的酶促水解 (enzymatic hydrolysis)，從而產生光學純的 L-BPA 【bbb0683】。
> - 回顧/比較來源：【CMDC-20-e202500059】報導了 Snippet 3, Snippet 4 【CMDC-20-e202500059】。
> - 回顧維度：該回顧強調了每種方法在成本效益 (cost-effectiveness)、可擴展性 (scalability) 和安全性 (safety) 方面的局限性 【CMDC-20-e202500059】。
>
> 核心權衡（高純度/同位素富集與可擴展性及成本效益）：
> - 高純度/同位素富集：在 L-BPA 合成中，生產高純度、同位素富集的材料仍然是一個挑戰 【CMDC-20-e202500059】。
> - 可擴展性：在大規模生產中使用任何氧化劑 (oxidant) 本質上都存在製程安全風險 【CMDC-20-e202500059】。
> - 成本效益：在製備同位素富集化合物時，主要成本通常來自於同位素起始原料 (isotope starting material) 【CMDC-20-e202500059】。
>
> ---
> 📋 **答案品質報告**
>
> ✅ **整體論文依據率**：83.3%　高（答案高度忠實於論文內容）
>
> ⚠️  **以下 1 個陳述未找到明確論文依據，請謹慎參考：**
>
> - Review/comparison source: `CMDC-20-e202500059` reports that Snippet 3, Snippet 4 [CMDC-20-e202500059].（信心 …（完整內容見 JSON）

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
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：57.1%　grounding：0.714
- 延遲：1863.3s　問題標記：{'conflicts': 0, 'unsupported': 2}

**答案預覽**：

> 比較框架：
> - 策略：【1-s2.0-S1347861320300633-main】報告了 JPH203 單一療法 (monotherapy)（共孵育 co-incubation 與預孵育 preincubation），其中預孵育協同增強了共孵育的抑制效果 [1-s2.0-S1347861320300633-main]。
> - 策略：【1-s2.0-S1936523321002047-main】報告了 JPH203 與放射治療 (radiation therapy) 的聯合應用，透過 mTOR 下調 (downregulation) 誘導細胞衰老 (cellular senescence)，從而增強放射敏感性 (radiosensitivity) [1-s2.0-S1936523321002047-main]。
> - 策略：【LAT1 ChemComm 2026】報告了自組裝肽 (self-assembling peptide) LffVLKK-4Phe 治療，透過多價交互作用 (multivalent interactions) 抑制 LAT1 介導的氨基酸運輸，隨後導致細胞膜破裂 (cell membrane disruption)，從而抑制增殖 [LAT1 ChemComm 2026]。
> - 策略：【cas0106-0279】報告了用於 BNCT 的 p-Boronophenylalanine (BPA) 遞送，透過 LAT1 及其他轉運蛋白使 10B 在惡性細胞中選擇性累積，隨後通過中子照射 (neutron irradiation) 產生 alpha 粒子與鋰核 (lithium nuclei) [cas0106-0279]。
> - 機制：【s41421-024-00697-6】報告了 JPH203 的結構結合涉及一個親水性頭部 (hydrophilic head) 與 …（完整內容見 JSON）

---
