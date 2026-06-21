# Eval Report — `tier1_purereport`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-06-21 22:53
- 題數：12

## 彙總

| 指標 | 值 |
|------|-----|
| 平均 grounding 分數 | 0.727 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 97.5% |
| 平均總延遲 | 1609.6s |
| 平均 planning 延遲 | 7.7s |
| 平均 retrieval 延遲 | 379.0s |
| 平均 grounding 延遲 | 42.9s |
| └ 其中 NLI | 26.6s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ✅ | Q01 | single_paper | 100.0% | 100.0% | 1.0 | 775.8s | C0/U0 |
| ⚠️ | Q02 | single_paper | 100.0% | 100.0% | 0.7 | 931.8s | C0/U23 |
| ✅ | Q03 | figure_dependent | 100.0% | 100.0% | 1.0 | 5570.7s | C0/U1 |
| ⚠️ | Q04 | table_lookup | 100.0% | 100.0% | 0.75 | 539.2s | C0/U6 |
| ⚠️ | Q05 | single_paper | 100.0% | 100.0% | 0.4 | 1356.9s | C0/U17 |
| ⚠️ | Q06 | multi_chunk | 100.0% | 75.0% | 0.45 | 2582.2s | C0/U14 |
| ⚠️ | Q07 | figure_dependent | 100.0% | 100.0% | 0.727 | 779.2s | C0/U10 |
| ⚠️ | Q08 | cross_paper | 100.0% | 100.0% | 0.757 | 2898.6s | C0/U11 |
| ✅ | Q09 | cross_paper | 100.0% | 100.0% | 0.97 | 1739.1s | C0/U1 |
| ⚠️ | Q10 | contradiction_check | 100.0% | 100.0% | 0.571 | 1151.1s | C0/U17 |
| ✅ | Q11 | out_of_scope | N/A | N/A | — | 337.5s | C0/U0 |
| ⚠️ | Q12 | false_premise | N/A | N/A | 0.667 | 653.2s | C0/U3 |

## 逐題細節

### ✅ Q01 · single_paper

**問題**：In the study of JPH203's preincubation inhibitory effect on LAT1, how does preincubation change JPH203's inhibitory potency? Give the reported values.

- detected_paper：`None`
- 選出論文：['1-s2.0-S1347861320300633-main', '1-s2.0-S0021979710012403-main', '1-s2.0-S1936523321002047-main', 's41421-024-00697-6']
- gold_papers：['1-s2.0-S1347861320300633-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：1.0
- 延遲：775.8s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
> * 預孵育（Preincubation）在與共同孵育（co-incubation）結合時，能顯著增強 JPH203 的抑制效力 【1-s2.0-S1347861320300633-main】。
> * 單獨進行共同孵育時，JPH203 的 $\text{IC}_{50}$ 值為 $99.2 \pm 11.0\text{ nM}$ 【1-s2.0-S1347861320300633-main】。
> * 當在共同孵育中加入預孵育時，JPH203 的 $\text{IC}_{50}$ 值為 $34.2 \pm 3.6\text{ nM}$ 【1-s2.0-S1347861320300633-main】。
> * 預孵育與共同孵育之間的協同作用導致 $\text{IC}_{50}$ 值降低了三倍 【1-s2.0-S1347861320300633-main】。
> * 共同孵育期間，JPH203 的 $K_i$ 值為 $0.46\text{ mM}$ 【1-s2.0-S1347861320300633-main】。
> * 當在共同孵育中加入預孵育時，JPH203 的 $K_i$ 值為 $0.37\text{ mM}$ 【1-s2.0-S1347861320300633-main】。
> * 預孵育與共同孵育之間的協同作用導致 $K_i$ 值降低了 20% 【1-s2.0-S1347861320300633-main】。
> 
> ## 【跨文獻推論】
> * 跨論文推論（基於 【1-s2.0-S1347861320300633-main】）：與立即暴露（僅共同孵育）相比，JPH203 在模擬長期暴露（預孵育 + 共同孵育）的條件下抑制效果更強，這表明該藥物的效力可能具有時間依賴性，或需要一段時間才能與 LAT1 轉運蛋白達到平衡。
> 
> ## 【知識延伸與推測】
> * 模型推測（文獻依據不足）： …（完整內容見 JSON）

---

### ⚠️ Q02 · single_paper

**問題**：What is the hybrid process used for the asymmetric synthesis of optically pure L-p-boronophenylalanine, and what are its key steps?

- detected_paper：`None`
- 選出論文：['bbb0683', 'CMDC-20-e202500059']
- gold_papers：['bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.7
- 延遲：931.8s　問題標記：{'conflicts': 0, 'unsupported': 23}

**答案預覽**：

> ## 【論文直接依據】
> 
> **一般流程與優勢**
> * L-p-硼苯丙氨酸 (L-BPA) 是一種腫瘤靶向氨基酸 【bbb0683】。
> * L-BPA 是硼中子捕獲治療 (BNCT) 最有效的硼載體之一 【bbb0683】。
> * 合成光學純 L-BPA 的複合製程結合了「對稱烷基化」以及隨後的「酵素水解」【bbb0683】。
> * 此製程使用 (2R)-(-)-2,5-dihydro-2-isopropyl-3,6-dimethoxypyrazine 作為對稱烷基化的商業可用手性輔助劑 【bbb0683】。
> * 此製程使用胰蛋白酶 (chymotrypsin) 進行水解 【bbb0683】。
> * 與使用手性膦的對稱氫化反應相比，該複合製程所需的反應步驟較少 【bbb0683】。
> * 與使用手性膦的對稱氫化反應相比，該複合製程的操作和後處理更簡單 【bbb0683】。
> * 與消旋前體的酵素解析 (enzymatic resolution) 相比，該複合製程的操作和後處理更簡單 【bbb0683】。
> * 透過胰蛋白酶水解，該製程將中間體 4 的立體選擇性 (74% e.e.) 提升至 L-BPA 的最終光學純度 100% e.e. 【bbb0683】。
> 
> **步驟 1：保護反應**
> * 第一步是保護反應，用以製備環狀硼酸酯 (2) 【bbb0683】。
> * 起始原料為 4-溴甲基苯硼酸 (1) 【bbb0683】。
> * 使用的試劑為 2,2-dimethyl-1,3-propanediol 【bbb0683】。
> * 使用的溶劑為乾燥 THF 【bbb0683】。
> * 反應在室溫下進行 【bbb0683】。
> * 攪拌時間為 10 分鐘 【bbb0683】。
> * 環狀硼酸酯 (2) 的產率為 79% 【bbb0683】。
> * 使用的起始原料量為 5.99 g (27.9 mmol) 【 …（完整內容見 JSON）

---

### ✅ Q03 · figure_dependent

**問題**：According to the cryo-EM structure, how does JPH203 bind to and inhibit the LAT1-4F2hc complex?

- detected_paper：`None`
- 選出論文：['s41421-024-00697-6', '1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 's41467-024-45464-z (1)', 'LAT1 ChemComm 2026']
- gold_papers：['s41421-024-00697-6']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：1.0
- 延遲：5570.7s　問題標記：{'conflicts': 0, 'unsupported': 1}

**答案預覽**：

> ## 【論文直接依據】
> * JPH203 結合於 LAT1 的傳統底物結合口袋中 [s41421-024-00697-6]。
> * JPH203 的分子結構包含一個具有苯丙氨酸（phenylalanine）骨架的親水頭部 [s41421-024-00697-6]。
> * JPH203 的分子結構包含一個具有 5-amino-2-(3-aminophenyl) benzoxazole 骨架的疏水尾部 [s41421-024-00697-6]。
> * 親水頭部的 $\alpha$-氨基與 TM1 和 TM6 非螺旋區域的主鏈原子形成氫鍵 [s41421-024-00697-6]。
> * 親水頭部的 $\alpha$-羧基與 TM1 和 TM6 非螺旋區域的主鏈原子形成氫鍵 [s41421-024-00697-6]。
> * JPH203 的二鹵酪胺酸（dihalo-tyrosine）核心上的氯原子與 Tyr259 形成鹵鍵（halogen bond）[s41421-024-00697-6]。
> * 疏水尾部嵌入由 TM1（Ile63, Ile64, Ile68）殘基組成的口袋中 [s41421-024-00697-6]。
> * 疏水尾部嵌入由 TM3（Ile140, Ile147, Ile148）殘基組成的口袋中 [s41421-024-00697-6]。
> * 疏水尾部嵌入由 TM6（Phe252, Gly255）殘基組成的口袋中 [s41421-024-00697-6]。
> * 疏水尾部嵌入由 TM10（Ile397, Phe400, Phe403）殘基組成的口袋中 [s41421-024-00697-6]。
> * JPH203 尾部的氨基與 TM10 上 Phe400 的主鏈形成氫鍵 [s41421-024-00697-6]。
> * JPH203 尾部的苯環與 Phe252 的側鏈產生 T 型  …（完整內容見 JSON）

---

### ⚠️ Q04 · table_lookup

**問題**：What are the main degradation products and impurities of BPA identified by HPLC, and under which storage conditions do they form?

- detected_paper：`None`
- 選出論文：['1-s2.0-S0731708511003451-main', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water']
- gold_papers：['1-s2.0-S0731708511003451-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.75
- 延遲：539.2s　問題標記：{'conflicts': 0, 'unsupported': 6}

**答案預覽**：

> ## 【論文直接依據】
> 
> **降解產物與條件**
> * Boronphenylalanine (BPA) 在鹼性和氧化條件下會降解為 Tyrosine 【Chunk 1, Chunk 2】。
> * 在氧化條件下，BPA 降解為 Tyrosine 的速度非常快 【Chunk 1, Chunk 2】。
> * 凍乾的 BPA/mannitol 藥物會以溫度依賴的方式緩慢降解為 Phenylalanine 【Chunk 1, Chunk 2】。
> * 當凍乾的 BPA/mannitol 在 40 °C 下儲存 6 個月時，會產生約 1% 的 Phenylalanine（相對於 BPA 質量） 【Chunk 1, Chunk 2】。
> * Boronic acids 在水性鹼性條件下會發生原脫硼反應 (protodeborylation) 以形成母體芳環 (parent arene) 【Chunk 2】。
> * 在 55 °C 下儲存 6 個月的 BPA 原料粉末中未檢測到降解 【Chunk 2】。
> * 在 40 °C 下儲存 12 個月的 BPA 原料粉末中未檢測到降解 【Chunk 2】。
> 
> **合成雜質**
> * BrPD (2-(4-Bromophenyl)-[1,3]dioxane) 是 BPA 的合成雜質 【Chunk 1】。
> * FBBA (4-Formylbenzeneboronic acid) 是 BPA 的合成雜質 【Chunk 1】。
> * BDPA (2-tert-Butoxycarbonylamino-3-[4-(5,5-dimethyl-[1,3,2]dioxaborinan-2-yl)-phenyl]acrylic acid methyl ester) 是 BPA 的合成雜質 【Chunk 1】。
> 
> **HPLC 鑑定數據**
> * Tyrosine 的 HPLC 保留時 …（完整內容見 JSON）

---

### ⚠️ Q05 · single_paper

**問題**：What is the solvent-free green protocol for N-Boc protection of amines using picric acid as a catalyst, and what reaction conditions are used?

- detected_paper：`None`
- 選出論文：['1-s2.0-S2773223124000268-main']
- gold_papers：['1-s2.0-S2773223124000268-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.4
- 延遲：1356.9s　問題標記：{'conflicts': 0, 'unsupported': 17}

**答案預覽**：

> ## 【論文直接依據】
> * 苦味酸被用作 Brønsted 酸催化劑 [Chunk 1, Chunk 2]。
> * 所使用的苦味酸量為 0.2 當量或 2 mol% [Chunk 1, Chunk 2]。
> * 二叔丁基二碳酸酯 ($\text{(Boc)}_2\text{O}$)，亦稱為 Boc 酐，被用作胺類保護的試劑 [Chunk 1]。
> * 反應在無溶劑（neat）條件下進行 [Chunk 1, Chunk 2]。
> * 反應在室溫 (rt) 下進行 [Chunk 1, Chunk 2]。
> * 最佳反應時間為 60 分鐘 [Chunk 1, Chunk 2]。
> * 此方法達到的最高產率為 98% [Chunk 1]。
> * 在最佳條件下，芳香族初級胺的產率可達 95%（表 1，Entry 13）[Chunk 1]。
> * 適用底物包括芳香族胺 [Chunk 1]。
> * 適用底物包括雜環芳香族胺 [Chunk 1]。
> * 適用底物包括開鏈脂肪族胺 [Chunk 1]。
> * 適用底物包括環狀脂肪族胺 [Chunk 1]。
> * 在提出的機理中，苦味酸活化 $\text{Boc}$ 酐的羰基氧原子 [Chunk 1]。
> * 在提出的機理中，胺的氮原子攻擊羰基碳 [Chunk 1]。
> * 機理的最後一步涉及質子取代以及二氧化碳 ($\text{CO}_2$) 和叔丁醇 (t-BuOH) 的消除，以形成 N-Boc 保護化合物 [Chunk 1]。
> * 對於 2-氨基苯酚的保護，使用了 1.2 當量的 Boc 酐 [Chunk 2]。
> * 在 0 °C 和 −10 °C 的溫度下未觀察到產物 [Chunk 2]。
> * 在 10 °C 的溫度下，觀察到產率為 20% [Chunk 2]。
> * 在沒有催化劑的情況下，反應需要 48 小時才能達到 55% 的產率 [Chunk 2]。
> * 本研 …（完整內容見 JSON）

---

### ⚠️ Q06 · multi_chunk

**問題**：How does poly(vinyl alcohol) enhance the therapeutic effect of 4-L-boronophenylalanine in neutron capture therapy of thoracic tumors? Explain the proposed mechanism with supporting data.

- detected_paper：`None`
- 選出論文：['1-s2.0-S0378517325007926-main', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'CMDC-20-e202500059']
- gold_papers：['1-s2.0-S0378517325007926-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：75.0%　grounding：0.45
- 延遲：2582.2s　問題標記：{'conflicts': 0, 'unsupported': 14}

**答案預覽**：

> ## 【論文直接依據】
> 
> * PVA 通過硼酸酯與 BPA 形成複合物 (PVA-BPA) [1-s2.0-S0378517325007926-main]。
> * PVA-BPA 複合物保留了針對 LAT1（大中性氨基酸轉運蛋白 1）所需的分別似苯丙氨酸結構 [1-s2.0-S0378517325007926-main]。
> * 傳統的 BPA 依賴 LAT1 從細胞外空間進入細胞，並與細胞內氨基酸（如谷氨酰胺）進行交換 [1-s2.0-S0378517325007926-main]。
> * 當細胞外濃度降低時，傳統的 BPA 通過與細胞外酪氨酸交換而流出細胞 [1-s2.0-S0378517325007926-main]。
> * PVA-BPA 是通過「LAT1 介導的內吞作用」而非傳統的交換方式被攝入癌細胞中 [1-s2.0-S0378517325007926-main]。
> * 進入細胞後，PVA-BPA 定位於內體/溶酶體 (endo-/lysosomes) 中 [1-s2.0-S0378517325007926-main]。
> * 在內體/溶酶體中的定位增加了細胞對 BPA 的攝取量 [1-s2.0-S0378517325007926-main]。
> * 在內體/溶酶體中的定位降低了 BPA 從細胞質流出的速度 [1-s2.0-S0378517325007926-main]。
> * 流出速度的降低延長了硼載體在腫瘤內的滯留時間 [1-s2.0-S0378517325007926-main]。
> * PVA-BPA 通過對 LAT1 的主動靶向，在 1 hour 內實現了在腫瘤中的高效累積與滯留 [1-s2.0-S0378517325007926-main]。
> * PVA-BPA 在正常器官中的累積有限 [1-s2.0-S0378517325007926-main]。
> * 使用山梨醇 …（完整內容見 JSON）

---

### ⚠️ Q07 · figure_dependent

**問題**：What is the water-stable boroxine structure reported, and what role do the dynamic covalent bonds play in its fluoride binding and hydrogel formation?

- detected_paper：`None`
- 選出論文：['41467_2024_Article_45464', 's41467-024-45464-z (1)']
- gold_papers：['41467_2024_Article_45464']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.727
- 延遲：779.2s　問題標記：{'conflicts': 0, 'unsupported': 10}

**答案預覽**：

> ## 【論文直接依據】
> 
> **水穩定的硼氧環（Boroxine）結構**
> * 報導的水穩定硼氧環結構源自 2-羥基苯硼酸 (HO-PBA) 【s41467-024-45464-z】。
> * HO-PBA 在室溫下會自發脫水形成二聚體 【s41467-024-45464-z】。
> * 該二聚體在室溫下接觸水後，會迅速轉化為水穩定的硼氧環三聚體 【s41467-024-45464-z】。
> * HO-PBA 三聚體的 pH 穩定範圍為 $2.4 < \text{pH} < 9$ 【s41467-024-45464-z】。
> * $\text{CH}_3\text{-HO-PBA}$ 三聚體的 pH 穩定範圍為 $2.4 < \text{pH} < 10$ 【s41467-024-45464-z】。
> 
> **動態共價鍵 (DCBs) 的作用**
> * 硼氧環結構包含與水兼容的動態共價鍵 (DCBs) 【s41467-024-45464-z】。
> * 這些 DCBs 中的 B–O 鍵在室溫的水環境中可以進行快速交換 【s41467-024-45464-z】。
> * 傳統的硼氧環結構需要在甲苯中於 $60\text{ °C}$ 加熱 8 小時才能進行交換反應 【s41467-024-45464-z】。
> 
> **氟離子結合**
> * HO-PBA 三聚體在水溶液中能選擇性地識別 $\text{F}^-$ 【s41467-024-45464-z】。
> * HO-PBA 三聚體對 $\text{F}^-$ 的結合親和力顯著高於苯硼酸 (PBA) 【s41467-024-45464-z】。
> * 在 $^1\text{H NMR}$ 譜圖中，向 HO-PBA 三聚體添加 $\text{F}^-$ 會使峰組的數量從四個增加到十個以上 【s41467-024-45464-z】。
> * 添加 $\text{Cl}^-$,  …（完整內容見 JSON）

---

### ⚠️ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.757
- 延遲：2898.6s　問題標記：{'conflicts': 0, 'unsupported': 11}

**答案預覽**：

> ## 【論文直接依據】
> 
> **同位素富集與成本效益**
> * $^{10}\text{B}$ 的價格比普通硼酸高出 1000 倍以上【CMDC-20-e202500059】。
> * 對於 $^{10}\text{B}$-L-BPA 合成路徑的評估，重點在於總步驟數以及引入硼原子後的產率【CMDC-20-e202500059】。
> * 使用 $\text{B}_2\text{pin}_2$ 的路徑被認為是不理想的，因為它們會浪費試劑中昂貴的 $^{10}\text{B}$ 原子【CMDC-20-e202500059】。
> * 使用三烷基硼酸酯的非 Pd 催化路徑在成本和安全性方面更具吸引力【CMDC-20-e202500059】。
> * 在 Pd 交叉耦合中，$\text{B}_2\text{pin}_2$ 結合 DMSO 和 KOAc 的效果優於 HBpin 結合 1,4-dioxane 和 $\text{NEt}_3$【CMDC-20-e202500059】。
> * $\text{B}_2\text{pin}_2$/DMSO/KOAc 的 Pd 交叉耦合組合總產率不超過 48%【CMDC-20-e202500059】。
> 
> **合成路徑的產率與效率**
> * 使用二鹵代苯的合成路徑在過去十年中因步驟過多而被大部分放棄【CMDC-20-e202500059】。
> * $\alpha$-苯甲醯胺基肉桂酸衍生物的不對稱氫化產率 e.e. 值為 76-88%【bbb0683】。
> * $\alpha$-苯甲醯胺基肉桂酸衍生物的不對稱氫化方法被認為是不理想的，因為底物難以獲取【bbb0683】。
> * 不對稱氫化方法需要額外的拆分步驟才能獲得光學純的 L-BPA【bbb0683】。
> * 結合對映選擇性烷基化和酶水解的「混合製程」（hybrid process），涉及使用 2,2-dimethyl-1,3 …（完整內容見 JSON）

---

### ✅ Q09 · cross_paper

**問題**：Several papers target LAT1 for cancer therapy. What different therapeutic strategies targeting LAT1 are described across the literature, and how do they differ in mechanism?

- detected_paper：`None`
- 選出論文：['LAT1 ChemComm 2026', '1-s2.0-S0378517325007926-main', '1-s2.0-S1347861320300633-main', 's41421-024-00697-6', '1-s2.0-S1936523321002047-main']
- gold_papers：['1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 'LAT1 ChemComm 2026', 's41421-024-00697-6']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.97
- 延遲：1739.1s　問題標記：{'conflicts': 0, 'unsupported': 1}

**答案預覽**：

> ## 【論文直接依據】
> 
> **藥理學抑制（阻斷運輸活性）**
> * LAT1 的治療策略包括使用抑制劑來阻斷其運輸活性的藥理學抑制【s41421-024-00697-6, 1-s2.0-S1347861320300633-main, 1-s2.0-S1936523321002047-main】。
> * JPH203 是一種選擇性且強效的酪胺酸衍生物抑制劑，能抑制多種癌症細胞系的腫瘤生長【1-s2.0-S1347861320300633-main, 1-s2.0-S1936523321002047-main】。
> * JPH203 以競爭方式抑制 LAT1，其 $\text{IC}_{50}$ 為 $60\text{--}140\text{ nM}$【1-s2.0-S1347861320300633-main】。
> * JPH203 具有預孵育（pre-incubation）抑制效果，能協同增強共同孵育（co-incubation）的抑制作用【1-s2.0-S1347861320300633-main】。
> * JPH203 的預孵育效果是暫時性的，在藥物移除後活性可恢復至 $95\text{--}97\%$【1-s2.0-S1347861320300633-main】。
> * JPH203 不會降低 LAT1 的 mRNA 或蛋白質表達水平【1-s2.0-S1347861320300633-main】。
> * JPH203 預孵育效果的潛在機制包括：黏附於亮氨酸（leucine）運輸路徑的 $\text{cis}$ 側，導致細胞質結合位點的 $\text{trans}$ 抑制，或促進 LAT1 的內吞作用（internalization）【1-s2.0-S1347861320300633-main】。
> * JPH203 透過其親水頭部（$\alpha$-氨基和 $\alpha$-羧基）與 …（完整內容見 JSON）

---

### ⚠️ Q10 · contradiction_check

**問題**：Do the papers agree on the hydrolytic stability of boroxine structures? Describe any agreements or differences between them.

- detected_paper：`None`
- 選出論文：['Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation', '41467_2024_Article_45464', 's41467-024-45464-z (1)', 'a-method-for-the-deprotection-of-alkylpinacolyl-boronate-esters']
- gold_papers：['41467_2024_Article_45464', 'Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.571
- 延遲：1151.1s　問題標記：{'conflicts': 0, 'unsupported': 17}

**答案預覽**：

> ## 【論文直接依據】
> 
> **來自 【s41467-024-45464-z (1)】：**
> * 傳統認知認為硼氧環（boroxines）在水解方面是不穩定的 [s41467-024-45464-z (1)]。
> * 傳統的硼氧環在接觸水或潮濕環境時會迅速水解為硼酸 [s41467-024-45464-z (1)]。
> * 快速水解導致傳統硼氧環的功能喪失 [s41467-024-45464-z (1)]。
> * 過去增強水解穩定性的策略包括引入電子捐贈基團 [s41467-024-45464-z (1)]。
> * 過去的策略包括引入龐大基團 [s41467-024-45464-z (1)]。
> * 過去的策略包括與 N-供體配位基形成加合物 [s41467-024-45464-z (1)]。
> * 過去的策略包括通過柔性大環進行熵穩定化，以降低路易斯酸性硼位點的親電性 [s41467-024-45464-z (1)]。
> * 這些過去的策略未能從根本上解決水解不穩定問題 [s41467-024-45464-z (1)]。
> * 2-羥基苯硼酸 (HO-PBA) 可以形成在水中穩定的硼氧環結構 [s41467-024-45464-z (1)]。
> * HO-PBA 在室溫下會自發脫水形成二聚體 [s41467-024-45464-z (1)]。
> * 接觸水後，HO-PBA 二聚體在室溫下迅速轉化為 HO-PBA 三聚體–H$_2$O 複合物 [s41467-024-45464-z (1)]。
> * HO-PBA 三聚體的穩定範圍為 $2.4 < \text{pH} < 9$ [s41467-024-45464-z (1)]。
> * $\text{CH}_3\text{-HO-PBA}$ 三聚體的穩定範圍為 $2.4 < \text{pH} < 10$ [s41467-024-45464-z (1) …（完整內容見 JSON）

---

### ✅ Q11 · out_of_scope

**問題**：What are the phase III clinical trial overall survival results for BNCT in glioblastoma patients?

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', '1-s2.0-S0378517325007926-main', 'cas0106-0279']
- gold_papers：[]
- 論文選擇命中率：N/A　檢索覆蓋率：N/A　grounding：—
- 延遲：337.5s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ⚠️ **資料來源說明**：本地學術文獻資料庫中未找到與此問題直接相關的內容。以下回答來自模型自身知識，非論文原文，請謹慎參考並自行查證。
> 
> 針對您提出的關於「硼中子捕獲治療（Boron Neutron Capture Therapy, BNCT）在膠質母細胞瘤（Glioblastoma, GBM）患者中的第三期臨床試驗整體生存期（Overall Survival, OS）結果」之問題，以下是基於目前學術知識的詳細分析與回答。
> 
> ### 核心結論：缺乏大規模隨機對照第三期臨床試驗
> 
> 首先必須明確指出：**目前在國際主流醫學文獻中，針對 BNCT 治療膠質母細胞瘤的大規模、隨機對照（Randomized Controlled Trial, RCT）第三期臨床試驗結果仍然非常稀缺。**
> 
> 大多數關於 BNCT 的研究仍處於第一期（安全性評估）或第二期（初步療效評估）階段。這主要是由於 BNCT 對於中子源（Neutron Source）的依賴性極高，過去需依賴核反應爐，而近年才逐漸轉向加速器-BNCT（Accelerator-BNCT, AB-BNCT），導致能進行大規模多中心試驗的醫療設施有限。
> 
> ### 現有研究之生存期趨勢分析（主要為 I/II 期數據）
> 
> 雖然缺乏標準的第三期試驗結果，但根據已發表的第二期或小規模前瞻性研究，BNCT 在 GBM 患者中的 OS 表現呈現以下特點：
> 
> 1.  **復發性膠質母細胞瘤（Recurrent GBM）：**
>     在針對復發性 GBM 的研究中，BNCT 被視為一種挽救療法（Salvage Therapy）。部分研究顯示，接受 BNCT 的患者其中位整體生存期（Median OS）較歷史對照組有所延長。某些小規模研究報告的 OS 可能落在 12 至 20 個月之間，但這類數據通常缺乏對照組的嚴格隨機化。
> 
> 2.  **初診/新 …（完整內容見 JSON）

---

### ⚠️ Q12 · false_premise

**問題**：Since BPA is administered orally in BNCT, what oral bioavailability values are reported in these papers?

- detected_paper：`None`
- 選出論文：['cas0106-0279', 'Chem Biol Drug Des - 2013 - Achilli - Folic Acid‐Conjugated 4‐Amino‐Phenylboronate  a Boron‐Containing Compound Designed']
- gold_papers：[]
- 論文選擇命中率：N/A　檢索覆蓋率：N/A　grounding：0.667
- 延遲：653.2s　問題標記：{'conflicts': 0, 'unsupported': 3}

**答案預覽**：

> ## 【論文直接依據】
> * BNCT 載體需要透過靜脈注射給藥 【Chem Biol Drug Des - 2013 - Achilli】。
> * 論文 【cas0106-0279】 分析了利用芳香族氨基酸轉運蛋白之 p-硼苯丙氨酸 (BPA) 的轉運動力學與參數 【cas0106-0279】。
> * BPA 對於血細胞和造血前驅細胞表現出高生物相容性，且未觀察到對中性粒細胞和血小板有影響 【Chem Biol Drug Des - 2013 - Achilli】。
> 
> ## 【跨文獻推論】
> * 文獻依據不足：提供的論文未報告任何關於 BPA 的口服生物利用度數值。此外，BPA 採取口服給藥的假設與 [Fact 3] 相矛盾，後者指出 BNCT 載體是透過靜脈注射給藥。
> 
> ## 【知識延伸與推測】
> * 模型推測（文獻依據不足）：雖然提供的文本未提及口服給藥，但 【cas0106-0279】 中對芳香族氨基酸轉運蛋白的研究表明，BPA 進入細胞的過程是由特定的轉運蛋白介導的。在假設的口服給藥情境中，BPA 的生物利用度將取決於腸上皮細胞中這些相同轉運蛋白（如 LAT1）的表達與效率，以促進其從腸道吸收進入血液循環。然而，由於提供的文本中缺乏藥代動力學數據，口服生物利用度的任何具體數值仍然未知。
> 
> ---
> 📋 **答案品質報告**
> 
> 📊 **分段論文依據率：**
> 
>   ⚠️ 【論文直接依據】：66.7%（2/3 句）
>   ❌ 【跨文獻推論】：0.0%（0/1 句）  ← 跨論文推論，低分為預期範圍
>   ❌ 【知識延伸推測】：0.0%（0/1 句）  ← 知識延伸推測，低分為預期範圍
> 
> ⚠️ **直引依據率**：66.7%　中（部分直引陳述需確認）
> 
> ⚠️  **以下 3 個陳述未找到明確論文依據，請謹慎參考：**
> 
> - * BPA exhibits high biocompatibi …（完整內容見 JSON）

---
