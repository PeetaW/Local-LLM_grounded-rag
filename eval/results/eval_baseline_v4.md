# Eval Report — `baseline_v4`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-06-23 17:14
- 題數：12

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.667 |
| 平均 grounding 分數 | 0.727 |
| 平均論文選擇命中率 | 97.5% |
| 平均檢索覆蓋率 | 97.5% |
| 平均總延遲 | 1011.4s |
| 平均 planning 延遲 | 7.0s |
| 平均 retrieval 延遲 | 400.0s |
| 平均 grounding 延遲 | 46.5s |
| └ 其中 NLI | 29.8s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ✅ | Q01 | single_paper | 100.0% | 100.0% | 0.95 | 885.7s | C0/U2 |
| ⚠️ | Q02 | single_paper | 100.0% | 100.0% | 0.766 | 956.9s | C0/U14 |
| ✅ | Q03 | figure_dependent | 100.0% | 100.0% | 0.969 | 1002.8s | C0/U1 |
| ⚠️ | Q04 | table_lookup | 100.0% | 100.0% | 0.655 | 739.8s | C0/U13 |
| ⚠️ | Q05 | single_paper | 100.0% | 100.0% | 0.444 | 633.0s | C0/U13 |
| ⚠️ | Q06 | multi_chunk | 100.0% | 75.0% | 0.75 | 1012.7s | C0/U6 |
| ⚠️ | Q07 | figure_dependent | 100.0% | 100.0% | 0.4 | 1036.1s | C0/U19 |
| ✅ | Q08 | cross_paper | 100.0% | 100.0% | 0.833 | 1561.9s | C0/U8 |
| ⚠️ | Q09 | cross_paper | 75.0% | 100.0% | 1.0 | 2057.3s | C0/U0 |
| ⚠️ | Q10 | contradiction_check | 100.0% | 100.0% | 0.6 | 1170.4s | C0/U15 |
| ✅ | Q11 | out_of_scope | N/A | N/A | — | 361.6s | C0/U0 |
| ⚠️ | Q12 | false_premise | N/A | N/A | 0.625 | 719.0s | C0/U5 |

## 逐題細節

### ✅ Q01 · single_paper

**問題**：In the study of JPH203's preincubation inhibitory effect on LAT1, how does preincubation change JPH203's inhibitory potency? Give the reported values.

- detected_paper：`None`
- 選出論文：['1-s2.0-S1347861320300633-main', '1-s2.0-S0021979710012403-main', '1-s2.0-S1936523321002047-main', 's41421-024-00697-6']
- gold_papers：['1-s2.0-S1347861320300633-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.95
- 延遲：885.7s　問題標記：{'conflicts': 0, 'unsupported': 2}

**答案預覽**：

> ## 【論文直接依據】
> 
> * 預孵育（Preincubation）在隨後進行共同孵育（co-incubation）時，能顯著增強 JPH203 的抑制效力【1-s2.0-S1347861320300633-main】。
> * 單獨進行共同孵育的 $\text{IC}_{50}$ 值為 $99.2 \pm 11.0\text{ nM}$【1-s2.0-S1347861320300633-main】。
> * 預孵育結合共同孵育的 $\text{IC}_{50}$ 值為 $34.2 \pm 3.6\text{ nM}$【1-s2.0-S1347861320300633-main】。
> * 預孵育與共同孵育之間的協同作用導致 $\text{IC}_{50}$ 值降低了三倍【1-s2.0-S1347861320300633-main】。
> * 單獨進行共同孵育的 $\text{K}_i$ 值為 $0.46\text{ mM}$【1-s2.0-S1347861320300633-main】。
> * 預孵育結合共同孵育的 $\text{K}_i$ 值為 $0.37\text{ mM}$【1-s2.0-S1347861320300633-main】。
> * 預孵育與共同孵育之間的協同作用導致 $\text{K}_i$ 值降低了 20%【1-s2.0-S1347861320300633-main】。
> * 單獨預孵育抑制效應的 $\text{IC}_{50}$ 為 $193 \pm 50\text{ nM}$【1-s2.0-S1347861320300633-main】。
> * 在 HT-29 細胞中，濃度為 $0.03\text{ mM}$ 的 JPH203 預孵育 120 分鐘後，將亮氨酸（leucine）的攝取量降低至對照組的 63%【1-s2.0-S1347861320300633-main】。
> * …（完整內容見 JSON）

---

### ⚠️ Q02 · single_paper

**問題**：What is the hybrid process used for the asymmetric synthesis of optically pure L-p-boronophenylalanine, and what are its key steps?

- detected_paper：`None`
- 選出論文：['bbb0683', 'CMDC-20-e202500059']
- gold_papers：['bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.766
- 延遲：956.9s　問題標記：{'conflicts': 0, 'unsupported': 14}

**答案預覽**：

> ## 【論文直接依據】
> 
> **一般流程概述**
> * 合成光學純 L-p-硼苯丙氨酸 (L-BPA) 的混合流程結合了不對稱烷基化與隨後的酶促水解 (bbb0683)。
> * 與之前的合成方法相比，此混合流程的反應步驟較少 (bbb0683)。
> * 此混合流程比之前的合成方法提供更簡單的操作和後處理 (bbb0683)。
> 
> **第一步：起始原料保護**
> * 第一步涉及使 4-溴甲基苯硼酸 (1) 與 2,2-二甲基-1,3-丙二醇反應 (bbb0683)。
> * 此反應在乾燥 THF 中進行 (bbb0683)。
> * 反應在室溫（環境溫度）下發生 (bbb0683)。
> * 攪拌時間為 10 分鐘 (bbb0683)。
> * 生成的產物為環狀硼酸酯（化合物 2）(bbb0683)。
> * 此步驟的產率為 79% (bbb0683)。
> 
> **第二步：不對稱烷基化**
> * 此步驟使用手性輔助劑 (2R)-(-)-2,5-二氫-2-異丙基-3,6-二甲氧基吡嗪 (3) (bbb0683)。
> * 使用的試劑為濃度 1.68M 的 n-BuLi，溶劑為己烷 (bbb0683)。
> * 反應環境為氬氣氛圍 (bbb0683)。
> * 使用的溶劑為乾燥 THF (bbb0683)。
> * 溫度維持在 -78°C (bbb0683)。
> * n-BuLi 在 -78°C 下於 30 分鐘內加入 (bbb0683)。
> * 加入 n-BuLi 後需等待 1 h (bbb0683)。
> * 然後在 30 分鐘內加入保護後的 4-溴甲基苯硼酸衍生物 (2) 的溶液 (bbb0683)。
> * 在 -78°C 下繼續攪拌 15 h (bbb0683)。
> * 後處理使用 pH 7.0 的磷酸鹽緩衝液 (bbb0683)。
> * 生成的加成產物 4 具有 2S 構型 (bbb0683)。
> * 此步驟的立體選擇性為 74% e.e …（完整內容見 JSON）

---

### ✅ Q03 · figure_dependent

**問題**：According to the cryo-EM structure, how does JPH203 bind to and inhibit the LAT1-4F2hc complex?

- detected_paper：`None`
- 選出論文：['s41421-024-00697-6', '1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 's41467-024-45464-z (1)', 'LAT1 ChemComm 2026']
- gold_papers：['s41421-024-00697-6']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.969
- 延遲：1002.8s　問題標記：{'conflicts': 0, 'unsupported': 1}

**答案預覽**：

> ## 【論文直接依據】
> 
> **結合位點與分子結構**
> * JPH203 結合於 LAT1 的傳統底物結合口袋中 (s41421-024-00697-6)。
> * JPH203 的分子結構包含一個具有苯丙氨酸（phenylalanine）骨架的親水性頭部 (s41421-024-00697-6)。
> * JPH203 的分子結構包含一個具有 5-amino-2-(3-aminophenyl) benzoxazole 骨架的疏水性尾部 (s41421-024-00697-6)。
> 
> **分子交互作用**
> * JPH203 親水性頭部的 $\alpha$-氨基與 LAT1 中 TM1 和 TM6 解旋區域的主鏈原子形成氫鍵網絡 (s41421-024-00697-6)。
> * JPH203 親水性頭部的 $\alpha$-羧基與 LAT1 中 TM1 和 TM6 解旋區域的主鏈原子形成氫鍵網絡 (s41421-024-00697-6)。
> * JPH203 二鹵酪胺酸（dihalo-tyrosine）核心上的氯原子與 Tyr259 形成鹵鍵 (halogen bonds) (s41421-024-00697-6)。
> * JPH203 的疏水性尾部嵌入由 TM1 (Ile63, Ile64, Ile68)、TM3 (Ile140, Ile147, Ile148)、TM6 (Phe252, Gly255) 和 TM10 (Ile397, Phe400, Phe403) 組成的疏水性口袋中 (s41421-024-00697-6)。
> * JPH203 尾部的氨基與 TM10 上 Phe400 的主鏈形成氫鍵 (s41421-024-00697-6)。
> * JPH203 尾部的苯環與 Phe252 的側鏈發生 T 型 $\pi$-$\pi$ 交互作用 (s41421-024-00697-6)。
> 
>  …（完整內容見 JSON）

---

### ⚠️ Q04 · table_lookup

**問題**：What are the main degradation products and impurities of BPA identified by HPLC, and under which storage conditions do they form?

- detected_paper：`None`
- 選出論文：['1-s2.0-S0731708511003451-main', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water']
- gold_papers：['1-s2.0-S0731708511003451-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.655
- 延遲：739.8s　問題標記：{'conflicts': 0, 'unsupported': 13}

**答案預覽**：

> ## 【論文直接依據】
> 
> **降解產物與條件**
> * 硼苯丙氨酸 (BPA) 在鹼性條件下會降解為酪胺酸 (tyrosine) 【1-s2.0-S0731708511003451-main】。
> * BPA 在氧化條件下會降解為酪胺酸 【1-s2.0-S0731708511003451-main】。
> * 在水相過氧化氫 ($6\text{ mM H}_2\text{O}_2$) 中，BPA 極其迅速地降解為酪胺酸 【1-s2.0-S0731708511003451-main】。
> * 酪胺酸透過 HPLC 鑑定，其保留時間為 4.5 分鐘 【1-s2.0-S0731708511003451-main】。
> * 酪胺酸在 $230\text{ nm}$ 處進行定量 【1-s2.0-S0731708511003451-main】。
> * 在水相鹼性溶液中使用 $100\text{ mM NaOH}$ 可使 BPA 強制降解為酪胺酸 【1-s2.0-S0731708511003451-main】。
> * 此強制降解是將溶液在 $55^\circ\text{C}$ 下培養 24 小時而完成 【1-s2.0-S0731708511003451-main】。
> * 在含有 BPA 和甘露醇 (mannitol) 的冷凍乾燥藥品中，BPA 會降解為苯丙氨酸 (phenylalanine) 【1-s2.0-S0731708511003451-main】。
> * BPA 在冷凍乾燥狀態下向苯丙氨酸的降解速度較慢 【1-s2.0-S0731708511003451-main】。
> * BPA 在冷凍乾燥狀態下向苯丙氨酸的降解隨溫度而異 【1-s2.0-S0731708511003451-main】。
> * 冷凍乾燥的 BPA/甘露醇藥品在 $40^\circ\text{C}$ 下儲存 6 個月後，產生約 1%（ …（完整內容見 JSON）

---

### ⚠️ Q05 · single_paper

**問題**：What is the solvent-free green protocol for N-Boc protection of amines using picric acid as a catalyst, and what reaction conditions are used?

- detected_paper：`None`
- 選出論文：['1-s2.0-S2773223124000268-main']
- gold_papers：['1-s2.0-S2773223124000268-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.444
- 延遲：633.0s　問題標記：{'conflicts': 0, 'unsupported': 13}

**答案預覽**：

> ## 【論文直接依據】
> 
> * 苦味酸被用作 Brønsted 酸催化劑 (1-s2.0-S2773223124000268-main)。
> * 所使用的苦味酸量為 0.2 當量 (1-s2.0-S2773223124000268-main)。
> * 所使用的苦味酸量為 2 mol% (1-s2.0-S2773223124000268-main)。
> * 二叔丁基二碳酸酯 ((Boc)₂O / Boc anhydride) 被用作保護劑 (1-s2.0-S2773223124000268-main)。
> * 反應在無溶劑條件下進行 (1-s2.0-S2773223124000268-main)。
> * 反應在室溫下進行 (1-s2.0-S2773223124000268-main)。
> * 最適反應時間為 60 分鐘 (1-s2.0-S2773223124000268-main)。
> * 此方法達到的最高產率為 98% (1-s2.0-S2773223124000268-main)。
> * 在優化條件下，芳香族一級胺的產率為 95% (1-s2.0-S2773223124000268-main)。
> * 此方法適用於芳香族胺 (1-s2.0-S2773223124000268-main)。
> * 此方法適用於雜環芳香族胺 (1-s2.0-S2773223124000268-main)。
> * 此方法適用於開鏈脂肪族胺 (1-s2.0-S2773223124000268-main)。
> * 此方法適用於環狀脂肪族胺 (1-s2.0-S2773223124000268-main)。
> * 在反應機制中，苦味酸首先活化 Boc anhydride 中羰基的氧原子 (1-s2.0-S2773223124000268-main)。
> * 隨後，胺的氮原子提供孤對電子攻擊 Boc anhydride 的羰基碳 (1 …（完整內容見 JSON）

---

### ⚠️ Q06 · multi_chunk

**問題**：How does poly(vinyl alcohol) enhance the therapeutic effect of 4-L-boronophenylalanine in neutron capture therapy of thoracic tumors? Explain the proposed mechanism with supporting data.

- detected_paper：`None`
- 選出論文：['1-s2.0-S0378517325007926-main', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'CMDC-20-e202500059']
- gold_papers：['1-s2.0-S0378517325007926-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：75.0%　grounding：0.75
- 延遲：1012.7s　問題標記：{'conflicts': 0, 'unsupported': 6}

**答案預覽**：

> ## 【論文直接依據】
> * 聚乙烯醇 (PVA) 透過硼酸酯與多個 4-L-硼苯丙氨酸 (BPA) 分子形成複合物 (1-s2.0-S0378517325007926-main)。
> * PVA-BPA 複合物保留了與 L 型氨基酸轉運蛋白 1 (LAT1) 相互作用所需的類苯丙氨酸結構 (1-s2.0-S0378517325007926-main)。
> * PVA-BPA 與 LAT1 相互作用 (1-s2.0-S0378517325007926-main)。
> * PVA-BPA 透過一種稱為「LAT1 介導的胞吞作用 (LAT1-mediated endocytosis)」的過程被攝入癌細胞中 (1-s2.0-S0378517325007926-main)。
> * 進入癌細胞後，PVA-BPA 定位於內體/溶酶體 (endo-/lysosomes) 中 (1-s2.0-S0378517325007926-main)。
> * PVA-BPA 的攝取機制增加了細胞攝取的 BPA 量 (1-s2.0-S0378517325007926-main)。
> * PVA-BPA 的攝取機制延緩了 BPA 從細胞質中的外排 (efflux) (1-s2.0-S0378517325007926-main)。
> * 外排的延緩延長了 BPA 在腫瘤內的滯留時間 (1-s2.0-S0378517325007926-main)。
> * PVA-BPA 透過對 LAT1 的主動靶向，在 1 h 內實現高效的腫瘤累積與滯留 (1-s2.0-S0378517325007926-main)。
> * PVA-BPA 在正常器官中的累積有限 (1-s2.0-S0378517325007926-main)。
> * 當細胞外 BPA 濃度降低時，傳統 BPA 會因與細胞外酪胺酸 (tyrosine) 交換而從細胞中外排 (1- …（完整內容見 JSON）

---

### ⚠️ Q07 · figure_dependent

**問題**：What is the water-stable boroxine structure reported, and what role do the dynamic covalent bonds play in its fluoride binding and hydrogel formation?

- detected_paper：`None`
- 選出論文：['41467_2024_Article_45464', 's41467-024-45464-z (1)']
- gold_papers：['41467_2024_Article_45464']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.4
- 延遲：1036.1s　問題標記：{'conflicts': 0, 'unsupported': 19}

**答案預覽**：

> ## 【論文直接依據】
> 
> **水穩定硼氧環結構**
> * 在 2-羥基苯硼酸 (HO-PBA) 中發現了一種水穩定的硼氧環結構 (s41467-024-45464-z)。
> * HO-PBA 三聚體的 pH 穩定範圍為 $2.4 < \text{pH} < 9$ (s41467-024-45464-z)。
> * $\text{CH}_3\text{-HO-PBA}$ 三聚體的 pH 穩定範圍為 $2.4 < \text{pH} < 10$ (s41467-024-45464-z)。
> * 此結構中的 B–O 鍵在室溫的水介質中會發生快速交換 (s41467-024-45464-z)。
> * 傳統的硼氧環結構需要甲苯溶液才能進行交換反應 (s41467-024-45464-z)。
> * 傳統的硼氧環結構需要加熱至 $60^\circ\text{C}$ 才能進行交換反應 (s41467-024-45464-z)。
> * 傳統的硼氧環結構需要 8 小時才能維持交換反應 (s41467-024-45464-z)。
> 
> **氟化物結合**
> * HO-PBA 三聚體在水溶液中能選擇性地識別 $\text{F}^-$ (s41467-024-45464-z)。
> * 加入 $\text{F}^-$ 後，HO-PBA 三聚體的 $^1\text{H NMR}$ 譜圖顯示峰組數從四組增加到十組以上 (s41467-024-45464-z)。
> * 加入 $\text{Cl}^-$, $\text{Br}^-$, 或 $\text{I}^-$ 對 HO-PBA 三聚體的 $^1\text{H NMR}$ 譜圖沒有影響 (s41467-024-45464-z)。
> * 與苯硼酸 (PBA) 相比，硼氧環結構在水溶液中對 $\text{F}^-$ 表現出顯著更高的結合親和力 (s41467-024-45464-z)。
>  …（完整內容見 JSON）

---

### ✅ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.833
- 延遲：1561.9s　問題標記：{'conflicts': 0, 'unsupported': 8}

**答案預覽**：

> ## 【論文直接依據】
> 
> **Pd 催化直接硼化路徑**
> * 使用 $\text{B}_2\text{pin}_2$ 搭配 $\text{DMSO}$ 和 $\text{KOAc}$ 比使用 $\text{HBpin}$ 搭配 $1,4\text{-dioxane}$ 和 $\text{NEt}_3$ 更有效 (【CMDC-20-e2500059】)。
> * 使用 $\text{HBpin}$ 路徑的交叉耦合與脫保護總產率不超過 $48\%$ (【CMDC-20-e2500059】)。
> * Pd 催化路徑存在重金屬污染的風險 (【CMDC-20-e2500059】)。
> 
> **非 Pd n-BuLi 路徑**
> * 此路徑在 $\text{THF}$ 中使用 $\text{n-BuLi}$ 將 $\text{B}(\text{OBu})_3$ 與 Boc 保護的 4-碘苯丙氨酸耦合 (【CMDC-20-e2500059】)。
> * 此路徑的兩步總產率為 $53\text{--}58\%$ (【CMDC-20-e2500059】)。
> * 當使用 $^{10}\text{B}(\text{OBu})_3$ 時，第一步耦合產率為 $65\%$ (【CMDC-20-e2500059】)。
> * 當使用 $^{10}\text{B}(\text{OBu})_3$ 時，脫保護產率為 $95\%$ (【CMDC-20-e2500059】)。
> * 當使用 $^{10}\text{B}(\text{OBu})_3$ 時，總產率為 $62\%$ (【CMDC-20-e2500059】)。
> 
> **非 Pd 格氏試劑路徑**
> * 使用 $\text{i-PrMgCl}$ 搭配 $\text{B}(\text{OBu})_3$ 的耦合步驟產率為 $83\%$ (【CMDC-20-e2500059】)。
> * 使 …（完整內容見 JSON）

---

### ⚠️ Q09 · cross_paper

**問題**：Several papers target LAT1 for cancer therapy. What different therapeutic strategies targeting LAT1 are described across the literature, and how do they differ in mechanism?

- detected_paper：`None`
- 選出論文：['LAT1 ChemComm 2026', '1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 'LAT1 ChemComm 2026SI', 'cas0106-0279']
- gold_papers：['1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 'LAT1 ChemComm 2026', 's41421-024-00697-6']
- 論文選擇命中率：75.0%　檢索覆蓋率：100.0%　grounding：1.0
- 延遲：2057.3s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> 根據提供的學術事實，針對 LAT1 的癌症治療策略可分為三種主要方法：遞送硼劑以進行中子捕獲治療、透過小分子進行競爭性抑制，以及透過自組裝肽進行抑制。
> 
> ## 【論文直接依據】
> 
> ### 策略 1：硼遞送劑 (BNCT)
> * 針對 LAT1 的治療策略包括使用 L-BPA、PVA-BPA 和 PVA-sorbitol-BPA (1-s2.0-S0378517325007926-main)。
> * L-BPA 透過 LAT1 從細胞外空間運輸至細胞質，並伴隨細胞內氨基酸（如麩醯胺酸）的流出 (1-s2.0-S0378517325007926-main)。
> * 當細胞外 BPA 濃度降低時，細胞內 BPA 會與細胞外酪胺酸進行交換 (1-s2.0-S0378517325007926-main)。
> * L-BPA 在熱中子照射前或照射期間需要持續輸注，通常為 30-60 分鐘 (1-s2.0-S0378517325007926-main)。
> * 對於需要約 100 分鐘照射的軀幹腫瘤，維持穩定的 L-BPA 輸注在技術上較為困難 (1-s2.0-S0378517325007926-main)。
> * PVA-BPA 由多個 BPA 分子透過硼酸酯與 PVA 連結而成，同時保留類苯丙氨酸結構以針對 LAT1 (1-s2.0-S0378517325007926-main)。
> * PVA-BPA 透過 LAT1 介導的內吞作用進入癌細胞 (1-s2.0-S0378517325007926-main)。
> * PVA-BPA 定位於內體/溶酶體中 (1-s2.0-S0378517325007926-main)。
> * PVA-BPA 增加了細胞攝取並延緩了 BPA 從細胞質的流出 (1-s2.0-S0378517325007926-main)。
> * PVA-BPA 在 1 小時內可在腫瘤中實現高 …（完整內容見 JSON）

---

### ⚠️ Q10 · contradiction_check

**問題**：Do the papers agree on the hydrolytic stability of boroxine structures? Describe any agreements or differences between them.

- detected_paper：`None`
- 選出論文：['Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation', '41467_2024_Article_45464', 's41467-024-45464-z (1)', 'a-method-for-the-deprotection-of-alkylpinacolyl-boronate-esters']
- gold_papers：['41467_2024_Article_45464', 'Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.6
- 延遲：1170.4s　問題標記：{'conflicts': 0, 'unsupported': 15}

**答案預覽**：

> ## 【論文直接依據】
> 
> **關於傳統 boroxine 的穩定性及其先前的改良嘗試：**
> * 傳統共識認為 boroxines 在水環境中具有水解不穩定性 (41467_2024_Article_45464, s41467-024-45464-z (1))。
> * 與水接觸通常會導致傳統 boroxines 迅速水解為硼酸 (boronic acids) 並失去功能 (41467_2024_Article_45464, s41467-024-45464-z (1))。
> * 先前提高抗水解能力的嘗試包括引入電子捐贈基團 (electron-donating groups) (41467_2024_Article_45464, s41467-024-45464-z (1))。
> * 先前提高抗水解能力的嘗試包括引入大體積基團 (bulky groups) (41467_2024_Article_45464, s41467-024-45464-z (1))。
> * 先前提高抗水解能力的嘗試包括與 N-供體配體 (N-donor ligands) 形成加合物 (41467_2024_Article_45464, s41467-024-45464-z (1))。
> * 先前提高抗水解能力的嘗試包括透過將三個硼酸單元整合到柔性大環 (flexible macrocycles) 中來採取熵穩定策略 (entropy stabilization strategies) (41467_2024_Article_45464, s41467-024-45464-z (1))。
> * 這些先前的方法未能從根本上解決水解不穩定性的問題 (41467_2024_Article_45464, s41467-024-45464-z (1))。
> 
> **關於源自 HO-PBA 的 boroxines：**
> * 源自  …（完整內容見 JSON）

---

### ✅ Q11 · out_of_scope

**問題**：What are the phase III clinical trial overall survival results for BNCT in glioblastoma patients?

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', '1-s2.0-S0378517325007926-main', 'cas0106-0279']
- gold_papers：[]
- 論文選擇命中率：N/A　檢索覆蓋率：N/A　grounding：—
- 延遲：361.6s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ⚠️ **資料來源說明**：本地學術文獻資料庫中未找到與此問題直接相關的內容。以下回答來自模型自身知識，非論文原文，請謹慎參考並自行查證。
> 
> （此為模型推測，建議查閱原始文獻確認）
> 
> 關於您詢問的「硼中子捕獲治療（BNCT）在膠質母細胞瘤（Glioblastoma, GBM）患者中的第三期臨床試驗總生存期（Overall Survival, OS）結果」，其回答如下：
> 
> **目前在主流學術文獻中，尚未見到針對 BNCT 治療膠質母細胞瘤且已完成並公布結果的大規模、隨機對照第三期（Phase III）臨床試驗數據。**
> 
> 以下為定性分析說明：
> 
> 1.  **研究階段之現況**：
>     目前 BNCT 在腦腫瘤領域的臨床研究大多集中在第一期（安全性評估）與第二期（初步療效探索）試驗。雖然有許多單中心或小規模的病例系列報告顯示出潛在的生存獲益，但這些數據缺乏第三期試驗所需的隨機對照組（Randomized Controlled Trial, RCT）以及足夠的樣本數來建立統計學上的定論。
> 
> 2.  **缺乏第三期數據的原因**：
>     *   **設備普及率低**：BNCT 需要特定的中子源（如加速器或核反應爐），全球可運作的中心數量有限，難以支持大規模的多中心第三期試驗。
>     *   **硼載體藥物限制**：目前使用的硼載體（如 BPA 或 BSH）在腫瘤蓄積量與分佈的個體差異較大，這增加了標準化治療方案的困難度。
> 
> 3.  **結論**：
>     由於缺乏正式的第三期臨床試驗結果，因此無法提供具體的總生存期（OS）統計數值。若您在某些文獻中看到存活率或生存月數，該數據通常來自於第二期試驗或回顧性分析，而非金標準的第三期隨機對照試驗。
> 
> （此為模型推測，建議查閱原始文獻確認）

---

### ⚠️ Q12 · false_premise

**問題**：Since BPA is administered orally in BNCT, what oral bioavailability values are reported in these papers?

- detected_paper：`None`
- 選出論文：['cas0106-0279', 'Chem Biol Drug Des - 2013 - Achilli - Folic Acid‐Conjugated 4‐Amino‐Phenylboronate  a Boron‐Containing Compound Designed']
- gold_papers：[]
- 論文選擇命中率：N/A　檢索覆蓋率：N/A　grounding：0.625
- 延遲：719.0s　問題標記：{'conflicts': 0, 'unsupported': 5}

**答案預覽**：

> ## 【論文直接依據】
> 
> * 提供之文本中未報告 BPA 的口服生物利用度數值 (cas0106-0279, Chem Biol Drug Des - 2013 - Achilli)。
> * $\text{ATB}_{0,+}$、$\text{LAT1}$ 與 $\text{LAT2}$ 能夠運輸 BPA (cas0106-0279)。
> * $\text{ATB}_{0,+}$ 運輸 BPA 的 $K_m$ 值為 $137.4 \pm 11.7\ \mu\text{M}$ (cas0106-0279)。
> * $\text{LAT1}$ 運輸 BPA 的 $K_m$ 值為 $20.3 \pm 0.8\ \mu\text{M}$ (cas0106-0279)。
> * $\text{LAT2}$ 運輸 BPA 的 $K_m$ 值為 $88.3 \pm 5.6\ \mu\text{M}$ (cas0106-0279)。
> * 在 BPA 濃度為 $100\ \mu\text{M}$ 時，$\text{LAT1}$ 蛋白質的量是決定 BPA 攝取的首要因素 (cas0106-0279)。
> * 在 BPA 濃度為 $1000\ \mu\text{M}$ 時，$\text{ATB}_{0,+}$ 約貢獻了 MCF-7 乳腺癌細胞中總 BPA 攝取的 20–25% (cas0106-0279)。
> * BPA 對於血細胞和造血前驅細胞表現出高生物相容性 (Chem Biol Drug Des - 2013 - Achilli)。
> 
> ## 【跨文獻推論】
> 
> * 文獻依據不足：雖然論文提供了促進 BPA 攝取的各種運輸蛋白（$\text{ATB}_{0,+}$、$\text{LAT1}$、$\text{LAT2}$）的 $K_m$ 值，但這些是細胞動力學參數，不能用於推導或計算全身性口服生物利用度數值 …（完整內容見 JSON）

---
