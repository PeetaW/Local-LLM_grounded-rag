# Eval Report — `baseline_v3`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-06-23 01:22
- 題數：12

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.688 |
| 平均 grounding 分數 | 0.698 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 97.5% |
| 平均總延遲 | 1117.6s |
| 平均 planning 延遲 | 7.1s |
| 平均 retrieval 延遲 | 455.0s |
| 平均 grounding 延遲 | 49.7s |
| └ 其中 NLI | 32.5s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ✅ | Q01 | single_paper | 100.0% | 100.0% | 0.941 | 1188.9s | C0/U2 |
| ⚠️ | Q02 | single_paper | 100.0% | 100.0% | 0.725 | 1132.6s | C0/U17 |
| ✅ | Q03 | figure_dependent | 100.0% | 100.0% | 1.0 | 1091.7s | C0/U0 |
| ✅ | Q04 | table_lookup | 100.0% | 100.0% | 0.889 | 640.3s | C0/U4 |
| ❌ | Q05 | single_paper | 100.0% | 100.0% | 0.259 | 617.7s | C0/U24 |
| ⚠️ | Q06 | multi_chunk | 100.0% | 75.0% | 0.706 | 1869.7s | C0/U6 |
| ✅ | Q07 | figure_dependent | 100.0% | 100.0% | 0.826 | 1003.0s | C0/U5 |
| ✅ | Q08 | cross_paper | 100.0% | 100.0% | 0.833 | 1665.4s | C0/U9 |
| ✅ | Q09 | cross_paper | 100.0% | 100.0% | 1.0 | 1893.5s | C0/U0 |
| ⚠️ | Q10 | contradiction_check | 100.0% | 100.0% | 0.5 | 1306.9s | C0/U24 |
| ✅ | Q11 | out_of_scope | N/A | N/A | — | 344.5s | C0/U0 |
| ❌ | Q12 | false_premise | N/A | N/A | 0.0 | 657.0s | C0/U3 |

## 逐題細節

### ✅ Q01 · single_paper

**問題**：In the study of JPH203's preincubation inhibitory effect on LAT1, how does preincubation change JPH203's inhibitory potency? Give the reported values.

- detected_paper：`None`
- 選出論文：['1-s2.0-S1347861320300633-main', '1-s2.0-S0021979710012403-main', '1-s2.0-S1936523321002047-main', 's41421-024-00697-6']
- gold_papers：['1-s2.0-S1347861320300633-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.941
- 延遲：1188.9s　問題標記：{'conflicts': 0, 'unsupported': 2}

**答案預覽**：

> ## 【論文直接依據】
> 
> * 預孵育與共同孵育具有協同作用，能顯著增強 JPH203 的抑制效力 【1-s2.0-S1347861320300633-main】。
> * 單純共同孵育的 $\text{IC}_{50}$ 值為 $99.2 \pm 11.0\text{ nM}$ 【1-s2.0-S1347861320300633-main】。
> * 預孵育與共同孵育組合的 $\text{IC}_{50}$ 值為 $34.2 \pm 3.6\text{ nM}$ 【1-s2.0-S1347861320300633-main】。
> * 共同孵育抑制分析的 $\text{Ki}$ 值為 $0.46\text{ mM}$ 【1-s2.0-S1347861320300633-main】。
> * 預孵育與共同孵育組合的 $\text{Ki}$ 值為 $0.37\text{ mM}$ 【1-s2.0-S1347861320300633-main】。
> * 預孵育與共同孵育的協同作用導致 $\text{IC}_{50}$ 值降低了三倍 【1-s2.0-S1347861320300633-main】。
> * 預孵育與共同孵育的協同作用導致 $\text{Ki}$ 值降低了 20% 【1-s2.0-S1347861320300633-main】。
> * JPH203 預孵育的抑制效果是暫時性的 【1-s2.0-S1347861320300633-main】。
> * 使用 $10\text{ mM}$ JPH203 對 HT-29 細胞預孵育 30 分鐘，將亮氨酸（leucine）攝取活性降低至對照組的 42% 【1-s2.0-S1347861320300633-main】。
> * 使用 $10\text{ mM}$ JPH203 對 HT-29 細胞預孵育 60 分鐘，將亮氨酸攝取活性降低至對照組的 32% 【1 …（完整內容見 JSON）

---

### ⚠️ Q02 · single_paper

**問題**：What is the hybrid process used for the asymmetric synthesis of optically pure L-p-boronophenylalanine, and what are its key steps?

- detected_paper：`None`
- 選出論文：['bbb0683', 'CMDC-20-e202500059']
- gold_papers：['bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.725
- 延遲：1132.6s　問題標記：{'conflicts': 0, 'unsupported': 17}

**答案預覽**：

> ## 【論文直接依據】
> 
> **一般流程概述**
> * L-p-硼苯丙氨酸 (L-BPA) 是一種腫瘤靶向氨基酸 【bbb0683】。
> * L-BPA 是硼中子捕獲治療 (BNCT) 中最有效的硼載體之一 【bbb0683】。
> * 合成光學純 L-BPA 的混合流程結合了「不對稱烷基化」以及隨後的「酶水解」 【bbb0683】。
> * 與先前的方法相比，此方法所需的反應步驟較少 【bbb0683】。
> * 此方法在操作和後處理方面比先前的方法更容易 【bbb0683】。
> * 先前的方法需要額外的分離與純化，或使用 $\alpha$-benzamidocinnamic acid 衍生物作為底物 【bbb0683】。
> 
> **步驟 1：保護反應**
> * 起始原料為 4-bromomethylphenylboronic acid (1) 【bbb0683】。
> * 所使用的試劑為 2,2-dimethyl-1,3-propanediol 【bbb0683】。
> * 使用的溶劑為乾燥 THF 【bbb0683】。
> * 反應溫度為室溫 【bbb0683】。
> * 攪拌時間為 10 minutes 【bbb0683】。
> * 產物為環狀硼酸酯/化合物 2 (2-(4-Bromomethylphenyl)-5,5-dimethyl-1,3,2-dioxaborane) 【bbb0683】。
> * 此步驟的產率為 79% 【bbb0683】。
> 
> **步驟 2：對映選擇性烷基化**
> * 此步驟合成中間體 4 ((2S,5R)-4) 【bbb0683】。
> * 反應在氬氣氛圍下進行 【bbb0683】。
> * 使用的溶劑為乾燥 THF 【bbb0683】。
> * 反應溫度為 -78°C 【bbb0683】。
> * 試劑 $(2R)-(-)$-2,5-dihydro-2-isopropyl-3,6-dimethoxypyr …（完整內容見 JSON）

---

### ✅ Q03 · figure_dependent

**問題**：According to the cryo-EM structure, how does JPH203 bind to and inhibit the LAT1-4F2hc complex?

- detected_paper：`None`
- 選出論文：['s41421-024-00697-6', '1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 's41467-024-45464-z (1)', 'LAT1 ChemComm 2026']
- gold_papers：['s41421-024-00697-6']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：1.0
- 延遲：1091.7s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
> 
> * JPH203 結合於 LAT1 的傳統底物結合口袋中 【s41421-024-00697-6】。
> * JPH203 的分子結構包含一個具有苯丙氨酸（phenylalanine）骨架的親水性頭部基團 【s41421-024-00697-6】。
> * JPH203 的分子結構包含一個具有 5-amino-2-(3-aminophenyl) benzoxazole 骨架的疏水性尾部基團 【s41421-024-00697-6】。
> * JPH203 結合的 LAT1-4F2hc 複合物之冷凍電子顯微鏡（cryo-EM）結構處於向外開放（outward-facing）構象 【s41421-024-00697-6】。
> * cryo-EM 結構的整體解析度為 3.30 Å 【s41421-024-00697-6】。
> * 結構中 JPH203 部分的聚焦精修（focused refinement）解析度達到 3.25 Å 【s41421-024-00697-6】。
> * 結構分析的實驗條件包括使用 500 μM JPH203 【s41421-024-00697-6】。
> * 結構分析的實驗條件包括尺寸排除色譜（size exclusion chromatography）洗脫 【s41421-024-00697-6】。
> * JPH203 親水性頭部的 $\alpha$-氨基與 TM1 和 TM6 解捲繞/去折疊（uncoiled/unfolded）區域的主鏈原子形成氫鍵網絡 【s41421-024-00697-6】。
> * JPH203 親水性頭部的 $\alpha$-羧基與 TM1 和 TM6 解捲繞/去折疊（uncoiled/unfolded）區域的主鏈原子形成氫鍵網絡 【s41421-024-00697-6】。
> * JPH203 二鹵酪胺酸（dihalo-tyro …（完整內容見 JSON）

---

### ✅ Q04 · table_lookup

**問題**：What are the main degradation products and impurities of BPA identified by HPLC, and under which storage conditions do they form?

- detected_paper：`None`
- 選出論文：['1-s2.0-S0731708511003451-main', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water']
- gold_papers：['1-s2.0-S0731708511003451-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.889
- 延遲：640.3s　問題標記：{'conflicts': 0, 'unsupported': 4}

**答案預覽**：

> ## 【論文直接依據】
> 
> **分解產物與條件**
> * BPA 在鹼性條件下產生分解產物酪胺酸 (tyrosine) 【1-s2.0-S0731708511003451-main】。
> * BPA 在氧化條件下產生分解產物酪胺酸 (tyrosine) 【1-s2.0-S0731708511003451-main】。
> * 在氧化條件下，BPA 分解為酪胺酸的速度非常快 【1-s2.0-S0731708511003451-main】。
> * 在氧化條件下，BPA 分解為酪胺酸的過程符合質量平衡 (mass balance) 【1-s2.0-S0731708511003451-main】。
> * 在 BPA/甘露醇 (mannitol) 凍乾藥物中，會形成分解產物苯丙胺酸 (phenylalanine) 【1-s2.0-S0731708511003451-main】。
> * BPA/甘露醇凍乾藥物中苯丙胺酸的形成是一個緩慢的過程 【1-s2.0-S0731708511003451-main】。
> * BPA/甘露醇凍乾藥物中苯丙胺酸的形成取決於溫度 【1-s2.0-S0731708511003451-main】。
> * 將 BPA/甘露醇凍乾藥物在 40 °C 下儲存 6 個月，會產生約 1%（相對於 BPA 質量）的苯丙胺酸 【1-s2.0-S0731708511003451-main】。
> 
> **合成雜質**
> * BrPD (2-(4-Bromophenyl)-[1,3]dioxane) 是 BPA 的合成雜質 【1-s2.0-S0731708511003451-main】。
> * FBBA (4-Formylbenzeneboronic acid) 是 BPA 的合成雜質 【1-s2.0-S0731708511003451-main】。
> * BDPA (2-tert-Butoxycarbon …（完整內容見 JSON）

---

### ❌ Q05 · single_paper

**問題**：What is the solvent-free green protocol for N-Boc protection of amines using picric acid as a catalyst, and what reaction conditions are used?

- detected_paper：`None`
- 選出論文：['1-s2.0-S2773223124000268-main']
- gold_papers：['1-s2.0-S2773223124000268-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.259
- 延遲：617.7s　問題標記：{'conflicts': 0, 'unsupported': 24}

**答案預覽**：

> ## 【論文直接依據】
> 
> * 苦味酸被用作布朗斯特酸 (Brønsted acid) 或有機催化劑，用於胺的 N-Boc 保護 [Chunk 1], [Chunk 2]。
> * 所使用的催化劑量為 2 mol% [Chunk 1], [Chunk 2]。
> * 所使用的催化劑量為 0.2 當量 (equivalents) [Chunk 1], [Chunk 2]。
> * 使用的保護試劑為二叔丁基二碳酸酯 ((Boc)₂O / Boc anhydride) [Chunk 1]。
> * 反應在無溶劑 (solvent-free/neat) 條件下進行 [Chunk 1], [Chunk 2]。
> * 反應在室溫 (rt) 下進行 [Chunk 1], [Chunk 2]。
> * 最佳反應時間為 60 分鐘 [Chunk 1], [Chunk 2]。
> * 此方法達到的最高產率為 98% [Chunk 1], [Chunk 2]。
> * 對於芳香族一級胺，使用的催化劑量為 0.2 當量 [Chunk 1], [Chunk 2]。
> * 對於芳香族一級胺，反應在室溫下進行 [Chunk 1], [Chunk 2]。
> * 對於芳香族一級胺，反應時間為 60 分鐘 [Chunk 1], [Chunk 2]。
> * 對於芳香族一級胺，產率為 95% (Table 1, Entry 13) [Chunk 1], [Chunk 2]。
> * 適用底物包括芳香族胺，例如 2-aminophenol [Chunk 1]。
> * 適用底物包括雜環芳香族胺，例如 quinolin-4-amine [Chunk 1]。
> * 適用底物包括雜環芳香族胺，例如 benzo[d]thiazol-2-amine [Chunk 1]。
> * 適用底物包括雜環芳香族胺，例如 1H-imidazole [Chunk 1]。
> * 適用底物包括開 …（完整內容見 JSON）

---

### ⚠️ Q06 · multi_chunk

**問題**：How does poly(vinyl alcohol) enhance the therapeutic effect of 4-L-boronophenylalanine in neutron capture therapy of thoracic tumors? Explain the proposed mechanism with supporting data.

- detected_paper：`None`
- 選出論文：['1-s2.0-S0378517325007926-main', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'CMDC-20-e202500059']
- gold_papers：['1-s2.0-S0378517325007926-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：75.0%　grounding：0.706
- 延遲：1869.7s　問題標記：{'conflicts': 0, 'unsupported': 6}

**答案預覽**：

> ## 【論文直接依據】
> * 聚乙烯醇 (PVA) 通過硼酸酯與多個 BPA 分子反應，形成名為 PVA-BPA 的複合物 [1-s2.0-S0378517325007926-main]。
> * PVA-BPA 複合物保留了與 LAT1（L 型氨基酸轉運蛋白 1）相互作用至關重要的類苯丙氨酸結構 [1-s2.0-S0378517325007926-main]。
> * PVA-BPA 與 LAT1 相互作用 [1-s2.0-S0378517325007926-main]。
> * PVA-BPA 通過一種稱為「LAT1 介導的胞吞作用」的過程被攝入癌細胞中 [1-s2.0-S0378517325007926-main]。
> * 一旦進入細胞，PVA-BPA 會定位於內體/溶酶體中 [1-s2.0-S0378517325007926-main]。
> * PVA-BPA 的攝取機制增加了細胞對 BPA 的攝取量 [1-s2.0-S0378517325007926-main]。
> * PVA-BPA 的攝取機制延緩了 BPA 從細胞質中的外排 [1-s2.0-S0378517325007926-main]。
> * 延緩的外排延長了 BPA 在腫瘤內的滯留時間 [1-s2.0-S0378517325007926-main]。
> * PVA-BPA 在 1 hour 內能高效地積聚在腫瘤中 [1-s2.0-S0378517325007926-main]。
> * PVA-BPA 在腫瘤中保持較長的滯留時間 [1-s2.0-S0378517325007926-main]。
> * PVA-BPA 限制了在正常器官中的積聚 [1-s2.0-S0378517325007926-main]。
> * 研究人員使用山梨醇作為增溶劑，開發了一種名為「PVA-sorbitol-BPA」的優化配方 [1-s2.0-S03785173 …（完整內容見 JSON）

---

### ✅ Q07 · figure_dependent

**問題**：What is the water-stable boroxine structure reported, and what role do the dynamic covalent bonds play in its fluoride binding and hydrogel formation?

- detected_paper：`None`
- 選出論文：['41467_2024_Article_45464', 's41467-024-45464-z (1)']
- gold_papers：['41467_2024_Article_45464']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.826
- 延遲：1003.0s　問題標記：{'conflicts': 0, 'unsupported': 5}

**答案預覽**：

> ## 【論文直接依據】
> 
> **結構與穩定性**
> * 所報導的水穩定硼氧環（boroxine）結構源自於 2-羥基苯硼酸 (HO-PBA) 【s41467-024-45464-z】。
> * HO-PBA 三聚體的 pH 穩定範圍為 $2.4 < \text{pH} < 9$ 【s41467-024-45464-z】。
> * $\text{CH}_3\text{-HO-PBA}$ 三聚體的 pH 穩定範圍為 $2.4 < \text{pH} < 10$ 【s41467-024-45464-z】。
> * HO-PBA 在室溫下會自發性脫水形成二聚體 【s41467-024-45464-z】。
> * 在室溫下接觸水後，二聚體會迅速轉化為硼氧環結構 ($\text{HO-PBA}$ 三聚體-$\text{H}_2\text{O}$ 複合物) 【s41467-024-45464-z】。
> 
> **動態共價鍵 (DCBs)**
> * 該結構包含 B–O 動態共價鍵，在室溫的水溶液中可快速交換 【s41467-024-45464-z】。
> * 傳統的硼氧環交換反應通常需要在甲苯溶液中加熱至 $60\text{ }^\circ\text{C}$ 並持續 8 小時 【s41467-024-45464-z】。
> 
> **氟化物結合**
> * HO-PBA 三聚體在水溶液中能選擇性地識別氟離子 ($\text{F}^-$) 【s41467-024-45464-z】。
> * HO-PBA 三聚體對 $\text{F}^-$ 的結合親和力顯著高於常用的苯硼酸 (PBA) 受體 【s41467-024-45464-z】。
> * 在 $^1\text{H NMR}$ 光譜中，加入等莫耳量的 $\text{NaF}$ 或 $\text{F}^-$ 後，HO-PBA 三聚體的四組原始峰會增加至十個以上的峰 【s41467-024-45 …（完整內容見 JSON）

---

### ✅ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.833
- 延遲：1665.4s　問題標記：{'conflicts': 0, 'unsupported': 9}

**答案預覽**：

> ## 【論文直接依據】
> 
> **同位素富集與成本效益**
> * $^{10}\text{B}$ 的價格極高，超過普通硼酸的 1000 倍以上 【CMDC-20-e202500059】。
> * 合成路徑的成本效益主要由 $^{10}\text{B}$ 原子的利用率決定 【CMDC-20-e202500059】。
> * 依賴 $\text{B}_2\text{pin}_2$ 的 L-BPA 合成路徑因成本高昂且浪費 $^{10}\text{B}$ 核心而不受歡迎 【CMDC-20-e202500059】。
> 
> **通用合成策略**
> * L-BPA 的合成策略分為「手性優先 (Chirality First)」和「硼優先 (Boron First)」 【CMDC-20-e202500059】。
> * 在過去十年中，二鹵代苯的修飾方法大多被直接氨基酸官能化路徑所取代，因為前者涉及過多步驟 【CMDC-20-e202500059】。
> 
> **Pd 催化路徑**
> * Pd 催化的交叉耦合使得在存在複雜氨基酸官能基的情況下，直接硼化變得可行 【CMDC-20-e202500059】。
> * 若使用 Pd 催化氫化作為最後一步，僅靠過濾可能無法完全去除重金屬殘留 【CMDC-20-e202500059】。
> * FDA 對於非口服攝入 Pd 的限制為 $10\ \mu\text{g day}^{-1}$ 【CMDC-20-e202500059】。
> * 在 Pd 催化路徑中，$\text{B}_2\text{pin}_2/\text{DMSO}/\text{KOAc}$ 的組合比 $\text{HBpin}/1,4\text{-dioxane}/\text{NEt}_3$ 更有效 【CMDC-20-e202500059】。
> * $\text{HBpin}/1,4\text{-dioxane}/\text{NE …（完整內容見 JSON）

---

### ✅ Q09 · cross_paper

**問題**：Several papers target LAT1 for cancer therapy. What different therapeutic strategies targeting LAT1 are described across the literature, and how do they differ in mechanism?

- detected_paper：`None`
- 選出論文：['LAT1 ChemComm 2026', '1-s2.0-S0378517325007926-main', '1-s2.0-S1347861320300633-main', 's41421-024-00697-6', '1-s2.0-S1936523321002047-main']
- gold_papers：['1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 'LAT1 ChemComm 2026', 's41421-024-00697-6']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：1.0
- 延遲：1893.5s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
> 
> **通用策略**
> * LAT1 的治療策略包括使用 BPA（p-boronophenylalanine）及其製劑 [1-s2.0-S0378517325007926-main, 1-s2.0-S1347861320300633-main, 1-s2.0-S1936523321002047-main, 【LAT1 ChemComm 2026】, s41421-024-00697-6]。
> * LAT1 的治療策略包括使用 PVA-BPA 複合物 [1-s2.0-S0378517325007926-main, 1-s2.0-S1347861320300633-main, 1-s2.0-S1936523321002047-main, 【LAT1 ChemComm 2026】, s41421-024-00697-6]。
> * LAT1 的治療策略包括 mRNA 敲低 (knock-down) [1-s2.0-S0378517325007926-main, 1-s2.0-S1347861320300633-main, 1-s2.0-S1936523321002047-main, 【LAT1 ChemComm 2026】, s41421-024-00697-6]。
> * LAT1 的治療策略包括基因破壞 (genetic disruption) [1-s2.0-S0378517325007926-main, 1-s2.0-S1347861320300633-main, 1-s2.0-S1936523321002047-main, 【LAT1 ChemComm 2026】, s41421-024-00697-6]。
> * LAT1 的治療策略包括藥理學抑制或敲除 (knock out) [1-s2.0-S0378517325007926-main, 1-s2.0-S13 …（完整內容見 JSON）

---

### ⚠️ Q10 · contradiction_check

**問題**：Do the papers agree on the hydrolytic stability of boroxine structures? Describe any agreements or differences between them.

- detected_paper：`None`
- 選出論文：['Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation', '41467_2024_Article_45464', 's41467-024-45464-z (1)', 'a-method-for-the-deprotection-of-alkylpinacolyl-boronate-esters']
- gold_papers：['41467_2024_Article_45464', 'Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.5
- 延遲：1306.9s　問題標記：{'conflicts': 0, 'unsupported': 24}

**答案預覽**：

> ## 【論文直接依據】
> 
> **一般穩定性與傳統策略**
> * Boroxine（硼氧環）在水相介質中通常被認為具有水解不穩定性 [s41467-024-45464-z]。
> * 與水接觸通常會導致 boroxine 迅速水解為 boronic acids（硼酸），進而導致功能喪失 [s41467-024-45464-z]。
> * 傳統的提高穩定性策略包括引入電子捐贈基團 (electron donating groups) [s41467-024-45464-z]。
> * 傳統的提高穩定性策略包括引入大體積基團 (bulky groups) [s41467-024-45464-z]。
> * 傳統的提高穩定性策略包括與 N-供體配體 (N-donor ligands) 形成加合物 [s41467-024-45464-z]。
> * 傳統的提高穩定性策略包括將三個硼酸單元整合到柔性大環結構中，以實現熵穩定化 (entropic stabilization) [s41467-024-45464-z]。
> * 這些傳統方法尚未從根本上解決水解不穩定性的問題 [s41467-024-45464-z]。
> 
> **HO-PBA Boroxine 穩定性**
> * 源自 2-羥基苯硼酸 (HO-PBA) 的 boroxine 結構在水中是穩定的 [s41467-024-45464-z]。
> * HO-PBA 結構從根本上解決了水解不穩定性的問題 [s41467-024-45464-z]。
> * HO-PBA 在室溫下會自發脫水形成二聚體 (dimer) [s41467-024-45464-z]。
> * 與水接觸後，HO-PBA 二聚體在室溫下會迅速轉化為 HO-PBA 三聚體 (trimer)–$\text{H}_2\text{O}$ 複合物 [s41467-024-45464-z]。
> * HO-PBA 三聚體在室 …（完整內容見 JSON）

---

### ✅ Q11 · out_of_scope

**問題**：What are the phase III clinical trial overall survival results for BNCT in glioblastoma patients?

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', '1-s2.0-S0378517325007926-main', 'cas0106-0279']
- gold_papers：[]
- 論文選擇命中率：N/A　檢索覆蓋率：N/A　grounding：—
- 延遲：344.5s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ⚠️ **資料來源說明**：本地學術文獻資料庫中未找到與此問題直接相關的內容。以下回答來自模型自身知識，非論文原文，請謹慎參考並自行查證。
> 
> **此為模型推測，建議查閱原始文獻確認。**
> 
> 針對您詢問關於硼中子捕獲治療（Boron Neutron Capture Therapy, BNCT）在膠質母細胞瘤（Glioblastoma, GBM）患者中的第三期臨床試驗總生存期（Overall Survival, OS）結果，說明如下：
> 
> **此為模型推測，建議查閱原始文獻確認。** 根據目前的學術認知，BNCT 在治療膠質母細胞瘤的領域中，尚未有被廣泛認可且已完成的大規模、隨機對照第三期臨床試驗（Phase III Clinical Trial）所公布的定論性總生存期數據。
> 
> **此為模型推測，建議查閱原始文獻確認。** 該治療方法目前的研究進展主要集中在第一期與第二期臨床試驗，其核心目的在於評估治療的安全性、耐受性以及初步的療效（如局部控制率或短期生存趨勢）。由於 BNCT 需要特殊的中子源設備（如加速器型中子源）以及高效的硼載體藥物，其臨床推廣面臨較高的技術門檻與樣本量限制，因此尚未進入到能產生標準化第三期 OS 數據的階段。
> 
> **此為模型推測，建議查閱原始文獻確認。** 因此，若您在尋找具體的第三期試驗統計數值（如中位生存月數或風險比），目前在主流學術文獻中可能並不存在這些數據。目前的臨床證據多屬於小規模隊列研究或探索性研究，不足以作為標準治療指南中的定論數據。
> 
> **總結：您所詢問的「第三期臨床試驗總生存期結果」目前在學術界尚未確立或不存在對應的公開數據。**

---

### ❌ Q12 · false_premise

**問題**：Since BPA is administered orally in BNCT, what oral bioavailability values are reported in these papers?

- detected_paper：`None`
- 選出論文：['cas0106-0279', 'Chem Biol Drug Des - 2013 - Achilli - Folic Acid‐Conjugated 4‐Amino‐Phenylboronate  a Boron‐Containing Compound Designed']
- gold_papers：[]
- 論文選擇命中率：N/A　檢索覆蓋率：N/A　grounding：0.0
- 延遲：657.0s　問題標記：{'conflicts': 0, 'unsupported': 3}

**答案預覽**：

> ## 【論文直接依據】
> 
> * 提供之文本中未報告 BPA 的口服生物利用度數值。【Chem Biol Drug Des - 2013; cas0106-0279】
> 
> ## 【跨文獻推論】
> 
> * 文獻依據不足：提供之事實提到研究 cas0106-0279 旨在分析 BPA 的轉運動力學參數【cas0106-0279】，但提取的文本中未包含關於口服生物利用度的具體結果或數值。
> 
> ## 【知識延伸與推測】
> 
> * 模型推測（文獻依據不足）：雖然提供的文本未提及 BPA 的給藥途徑，但在臨床 BNCT 實踐中，p-boronophenylalanine (BPA) 通常透過靜脈輸注而非口服給藥，以確保對血硼濃度的精確控制。這些特定論文中缺乏口服生物利用度數據，可能是因為研究重點在於細胞轉運機制（使用 *Xenopus oocytes*）和生物相容性，而非口服劑量的藥代動力學分析。
> 
> ---
> 📋 **答案品質報告**
> 
> 📊 **分段論文依據率：**
> 
>   ❌ 【論文直接依據】：0.0%（0/1 句）
>   ❌ 【跨文獻推論】：0.0%（0/1 句）  ← 跨論文推論，低分為預期範圍
>   ❌ 【知識延伸推測】：0.0%（0/1 句）  ← 知識延伸推測，低分為預期範圍
> 
> ❌ **直引依據率**：0.0%　低（建議縮小問題範圍）
> 
> ⚠️  **以下 3 個陳述未找到明確論文依據，請謹慎參考：**
> 
> - * No oral bioavailability values for BPA are reported in the provided text. [Chem Biol Drug Des - 2013; cas0106-0279]（信心度：34.1%，最近似來源：cas0106-0279-d7164215）
> - * Insufficient literature basis: The pr …（完整內容見 JSON）

---
