# Eval Report — `selfcorrect`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-06-23 11:26
- 題數：6

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.625 |
| 平均 grounding 分數 | 0.732 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 95.8% |
| 平均總延遲 | 1152.0s |
| 平均 planning 延遲 | 7.8s |
| 平均 retrieval 延遲 | 427.8s |
| 平均 grounding 延遲 | 121.5s |
| └ 其中 NLI | 32.7s |
| └ 其中 gemma4 | 142.2s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ⚠️ | Q04 | table_lookup | 100.0% | 100.0% | 0.64 | 787.7s | C0/U13 |
| ✅ | Q05 | single_paper | 100.0% | 100.0% | 0.875 | 604.3s | C0/U6 |
| ✅ | Q06 | multi_chunk | 100.0% | 75.0% | 0.812 | 1165.5s | C0/U4 |
| ⚠️ | Q07 | figure_dependent | 100.0% | 100.0% | 0.545 | 1328.8s | C0/U19 |
| ✅ | Q09 | cross_paper | 100.0% | 100.0% | 1.0 | 1834.1s | C0/U0 |
| ⚠️ | Q10 | contradiction_check | 100.0% | 100.0% | 0.52 | 1191.5s | C0/U15 |

## 逐題細節

### ⚠️ Q04 · table_lookup

**問題**：What are the main degradation products and impurities of BPA identified by HPLC, and under which storage conditions do they form?

- detected_paper：`None`
- 選出論文：['1-s2.0-S0731708511003451-main', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water']
- gold_papers：['1-s2.0-S0731708511003451-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.64
- 延遲：787.7s　問題標記：{'conflicts': 0, 'unsupported': 13}

**答案預覽**：

> ## 【論文直接依據】
> 
> **降解產物與條件**
> * BPA 在鹼性條件下會降解為酪胺酸 (Tyrosine) [Chunk 1, Chunk 2]。
> * BPA 在氧化條件下會降解為酪胺酸 (Tyrosine) [Chunk 1, Chunk 2]。
> * BPA 在氧化條件下降解為酪胺酸 (Tyrosine) 的反應速度非常快 [Chunk 1, Chunk 2]。
> * 在鹼性水溶液中，BPA 降解為酪胺酸 (Tyrosine) 的過程會因接觸空氣而加速 [Chunk 2]。
> * 當 BPA 溶解於 $100\text{ mM NaOH}$ 並在 $55^\circ\text{C}$ 下儲存 24 小時後，可觀察到酪胺酸 (Tyrosine) 的生成 [Chunk 2]。
> * 在使用 $6\text{ mM H}_2\text{O}_2$ 的氧化條件下，BPA 降解為酪胺酸 (Tyrosine) 的速度非常快 [Chunk 2]。
> * 在鹼性水溶液條件下，BPA 通常會發生原脫硼反應 (protodeborylation) 以形成母體芳環化合物 (parent arene) [Chunk 2]。
> * 在 BPA/甘露醇 (mannitol) 凍乾藥物中，BPA 會降解為苯丙胺酸 (Phenylalanine) [Chunk 1, Chunk 2]。
> * 在 BPA/甘露醇 (mannitol) 凍乾藥物中，BPA 降解為苯丙胺酸 (Phenylalanine) 的速度較慢 [Chunk 1, Chunk 2]。
> * 在 BPA/甘露醇 (mannitol) 凍乾藥物中，BPA 降解為苯丙胺酸 (Phenylalanine) 與溫度相關 [Chunk 1, Chunk 2]。
> * 在 $40^\circ\text{C}$ 下儲存 6 個月的 BPA/甘露醇 (mannito …（完整內容見 JSON）

---

### ✅ Q05 · single_paper

**問題**：What is the solvent-free green protocol for N-Boc protection of amines using picric acid as a catalyst, and what reaction conditions are used?

- detected_paper：`None`
- 選出論文：['1-s2.0-S2773223124000268-main']
- gold_papers：['1-s2.0-S2773223124000268-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.875
- 延遲：604.3s　問題標記：{'conflicts': 0, 'unsupported': 6}

**答案預覽**：

> ## 【論文直接依據】
> 
> * 苦味酸被用作 Brønsted 酸催化劑 [1-s2.0-S2773223124000268-main]。
> * 所使用的苦味酸催化劑量為 2 mol% [1-s2.0-S2773223124000268-main]。
> * 或者，所使用的苦味酸催化劑量為 0.2 equivalents [1-s2.0-S2773223124000268-main]。
> * 所使用的保護劑為二叔丁基二碳酸酯 ((Boc)₂O / Boc anhydride) [1-s2.0-S2773223124000268-main]。
> * 反應在無溶劑 (neat) 條件下進行 [1-s2.0-S2773223124000268-main]。
> * 反應在室溫 (rt) 下進行 [1-s2.0-S2773223124000268-main]。
> * 優化後的反應時間為 60 分鐘 [1-s2.0-S2773223124000268-main]。
> * 此方法達到的最高產率為 98% [1-s2.0-S2773223124000268-main]。
> * 在室溫下使用 0.2 equivalents 的催化劑並反應 60 分鐘時，可獲得 95% 的產率 [1-s2.0-S2773223124000268-main]。
> * 適用底物包括芳香胺 [1-s2.0-S2773223124000268-main]。
> * 適用底物包括雜環芳香胺 [1-s2.0-S2773223124000268-main]。
> * 適用底物包括開鏈脂肪胺 [1-s2.0-S2773223124000268-main]。
> * 適用底物包括環狀脂肪胺 [1-s2.0-S2773223124000268-main]。
> * 反應機制始於苦味酸活化 Boc anhydride 的羰基氧 [1-s2.0-S27732231240 …（完整內容見 JSON）

---

### ✅ Q06 · multi_chunk

**問題**：How does poly(vinyl alcohol) enhance the therapeutic effect of 4-L-boronophenylalanine in neutron capture therapy of thoracic tumors? Explain the proposed mechanism with supporting data.

- detected_paper：`None`
- 選出論文：['1-s2.0-S0378517325007926-main', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'CMDC-20-e202500059']
- gold_papers：['1-s2.0-S0378517325007926-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：75.0%　grounding：0.812
- 延遲：1165.5s　問題標記：{'conflicts': 0, 'unsupported': 4}

**答案預覽**：

> ## 【論文直接依據】
> * 聚乙烯醇 (PVA) 通過硼酸酯與多個 4-L-硼苯丙氨酸 (BPA) 分子反應，形成稱為 PVA-BPA 的複合物 【1-s2.0-S0378517325007926-main】。
> * PVA-BPA 複合物保留了類苯丙氨酸結構，這對於與 L 型氨基酸轉運蛋白 1 (LAT1) 的相互作用至關重要 【1-s2.0-S0378517325007926-main】。
> * PVA-BPA 通過 LAT1 介導的胞吞作用被癌細胞吸收 【1-s2.0-S0378517325007926-main】。
> * PVA-BPA 的吸收機制與傳統 BPA 所使用的轉運方式不同 【1-s2.0-S0378517325007926-main】。
> * 進入癌細胞後，PVA-BPA 定位於內體/溶酶體 (endo-/lysosomes) 中 【1-s2.0-S0378517325007926-main】。
> * PVA-BPA 的吸收機制增加了細胞吸收 BPA 的總量 【1-s2.0-S0378517325007926-main】。
> * PVA-BPA 的吸收機制延緩了 BPA 從細胞質中的外排 【1-s2.0-S0378517325007926-main】。
> * 外排的延緩延長了 BPA 在腫瘤內的滯留時間 【1-s2.0-S0378517325007926-main】。
> * PVA-BPA 通過對 LAT1 的主動靶向，在 1 hour 內實現了腫瘤的高效累積與滯留 【1-s2.0-S0378517325007926-main】。[待確認]
> * PVA-BPA 在正常器官中的累積有限 【1-s2.0-S0378517325007926-main】。
> * 使用山梨醇 (sorbitol) 作為增溶劑，開發了一種稱為 "PVA-sorbitol-BPA" 的優化配方 【1- …（完整內容見 JSON）

---

### ⚠️ Q07 · figure_dependent

**問題**：What is the water-stable boroxine structure reported, and what role do the dynamic covalent bonds play in its fluoride binding and hydrogel formation?

- detected_paper：`None`
- 選出論文：['41467_2024_Article_45464', 's41467-024-45464-z (1)']
- gold_papers：['41467_2024_Article_45464']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.545
- 延遲：1328.8s　問題標記：{'conflicts': 0, 'unsupported': 19}

**答案預覽**：

> ## 【論文直接依據】
> 
> **水穩定硼氧環結構及其形成**
> * 報導的水穩定硼氧環結構為 HO-PBA 三聚體 $-\text{H}_2\text{O}$ 複合物 【s41467-024-45464-z】。
> * 2-羥基苯硼酸 (HO-PBA) 在室溫下會自發脫水形成二聚體 【s41467-024-45464-z】。
> * HO-PBA 二聚體在室溫下接觸水後，會迅速轉化為三聚體硼氧環結構 【s41467-024-45464-z】。
> * 向含有 HO-PBA 二聚體的 $\text{THF-}d_8$ 溶液中加入水會觸發此轉化 【s41467-024-45464-z】。
> * 水在二聚體轉化為三聚體的過程中起催化作用，這一點已通過使用甲醇 $-\text{H}_2^{18}\text{O}$ 溶液的同位素標記得到證實 【s41467-024-45464-z】。
> * 在檢測此轉化的實驗中，HO-PBA 二聚體在室溫下以 $1.7\text{ mg mL}^{-1}$ 的濃度溶解於甲醇-水混合物（體積比 5:1）中 【s41467-024-45464-z】。
> * 硼氧環結構通過核磁共振光譜 (NMR) 得到證實 【s41467-024-45464-z】。
> * 硼氧環結構通過質譜 (MS) 得到證實 【s41467-024-45464-z】。
> * 硼氧環結構通過 UV-Raman 光譜得到證實 【s41467-024-45464-z】。
> * 硼氧環結構通過密度泛函理論 (DFT) 計算得到證實 【s41467-024-45464-z】。
> * HO-PBA 三聚體的 pH 穩定範圍為 $2.4 < \text{pH} < 9$ 【s41467-024-45464-z】。
> * $\text{CH}_3\text{-HO-PBA}$ 三聚體的 pH 穩定範圍為 $2.4 < \text{p …（完整內容見 JSON）

---

### ✅ Q09 · cross_paper

**問題**：Several papers target LAT1 for cancer therapy. What different therapeutic strategies targeting LAT1 are described across the literature, and how do they differ in mechanism?

- detected_paper：`None`
- 選出論文：['LAT1 ChemComm 2026', '1-s2.0-S0378517325007926-main', '1-s2.0-S1347861320300633-main', 's41421-024-00697-6', '1-s2.0-S1936523321002047-main']
- gold_papers：['1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 'LAT1 ChemComm 2026', 's41421-024-00697-6']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：1.0
- 延遲：1834.1s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
> 
> **基因干預策略**
> * 降低 LAT1 mRNA 表達（mRNA 敲低/knock-down）可顯著減緩前列腺癌細胞的增殖、侵襲和遷移 [1-s2.0-S1347861320300633-main]。
> * 多功能 CD98/LAT1 複合體的基因破壞表明，必需氨基酸運輸對於控制腫瘤生長和 mTORC1 至關重要 [1-s2.0-S1936523321002047-main]。
> 
> **小分子抑制劑 (JPH203)**
> * JPH203 是一種酪氨酸衍生物，以競爭方式抑制 LAT1 及其底物 [1-s2.0-S1347861320300633-main]。
> * JPH203 的 $\text{IC}_{50}$ 值為 $60\text{--}140\text{ nM}$ [1-s2.0-S1347861320300633-main]。
> * JPH203 在多種癌細胞和腫瘤負荷動物模型中表現出抗增殖效果 [1-s2.0-S1347861320300633-main]。
> * 初步的人體臨床試驗結果顯示，JPH203 對於部分膽道癌和結腸癌患者具有耐受性且有效 [1-s2.0-S1347861320300633-main]。
> * JPH203 正在進行針對晚期實體腫瘤的 I 期和 II 期臨床試驗 [1-s2.0-S1936523321002047-main]。
> * JPH203 可誘導 YD-38 人類口腔癌細胞凋亡 [1-s2.0-S1936523321002047-main]。
> * JPH203 在完全免疫功能正常的鼠類模型中可減少未分化甲狀腺癌的生長 [1-s2.0-S1936523321002047-main]。
> * JPH203 在體外和體內均能抑制 MIA Paca-2 和 A549 非小細胞肺癌 (NSCLC) 細胞系的腫瘤生長 [1-s …（完整內容見 JSON）

---

### ⚠️ Q10 · contradiction_check

**問題**：Do the papers agree on the hydrolytic stability of boroxine structures? Describe any agreements or differences between them.

- detected_paper：`None`
- 選出論文：['Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation', '41467_2024_Article_45464', 's41467-024-45464-z (1)', 'a-method-for-the-deprotection-of-alkylpinacolyl-boronate-esters']
- gold_papers：['41467_2024_Article_45464', 'Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.52
- 延遲：1191.5s　問題標記：{'conflicts': 0, 'unsupported': 15}

**答案預覽**：

> ## 【論文直接依據】
> 
> **來自【s41467-024-45464-z (1)】：**
> * 傳統認知認為硼氧環（boroxines）在水解方面是不穩定的 [s41467-024-45464-z (1)]。
> * 傳統上，硼氧環在接觸水或潮濕環境時會迅速水解為硼酸，導致功能喪失 [s41467-024-45464-z (1)]。
> * 提高穩定性的傳統策略包括引入電子捐贈基團 [s41467-024-45464-z (1)]。
> * 提高穩定性的傳統策略包括添加大尺寸（大體積）基團 [s41467-024-45464-z (1)]。
> * 提高穩定性的傳統策略包括與 N-供體配體形成加合物 [s41467-024-45464-z (1)]。
> * 提高穩定性的傳統策略包括通過將三個硼酸單元整合到一個柔性大環中來實現熵穩定化 [s41467-024-45464-z (1)]。
> * 這些傳統方法使硼氧環對水解更具魯棒性，但未能從根本上解決不穩定問題 [s41467-024-45464-z (1)]。
> * 由 2-羥基苯硼酸 (HO-PBA) 組成的硼氧環結構具有水穩定性 [s41467-024-45464-z (1)]。
> * HO-PBA 在室溫下會自發脫水形成二聚體 [s41467-024-45464-z (1)]。
> * 接觸水後，HO-PBA 二聚體迅速轉化為包含水穩定硼氧環結構的 $\text{HO-PBA}$ 三聚體–$\text{H}_2\text{O}$ 複合物 [s41467-024-45464-z (1)]。
> * HO-PBA 三聚體的穩定範圍為 $2.4 < \text{pH} < 9$ [s41467-024-45464-z (1)]。
> * $\text{CH}_3\text{-HO-PBA}$ 三聚體的穩定範圍為 $2.4 < \text{pH} < 10$ [s4 …（完整內容見 JSON）

---
