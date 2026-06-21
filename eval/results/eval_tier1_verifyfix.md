# Eval Report — `tier1_verifyfix`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-06-19 19:54
- 題數：12

## 彙總

| 指標 | 值 |
|------|-----|
| 平均 grounding 分數 | 0.729 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 97.5% |
| 平均總延遲 | 2734.6s |
| 平均 planning 延遲 | 7.3s |
| 平均 retrieval 延遲 | 353.1s |
| 平均 grounding 延遲 | 726.9s |
| └ 其中 NLI | 35.1s |
| └ 其中 gemma4 | 674.7s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ✅ | Q01 | single_paper | 100.0% | 100.0% | 1.0 | 2448.9s | C0/U0 |
| ✅ | Q02 | single_paper | 100.0% | 100.0% | 0.875 | 2431.2s | C0/U6 |
| ✅ | Q03 | figure_dependent | 100.0% | 100.0% | 1.0 | 2586.6s | C0/U0 |
| ✅ | Q04 | table_lookup | 100.0% | 100.0% | 0.92 | 1977.9s | C0/U4 |
| ⚠️ | Q05 | single_paper | 100.0% | 100.0% | 0.389 | 3072.1s | C0/U15 |
| ⚠️ | Q06 | multi_chunk | 100.0% | 75.0% | 0.615 | 2538.6s | C0/U8 |
| ⚠️ | Q07 | figure_dependent | 100.0% | 100.0% | 0.68 | 4219.4s | C0/U11 |
| ✅ | Q08 | cross_paper | 100.0% | 100.0% | 0.868 | 3433.2s | C0/U6 |
| ✅ | Q09 | cross_paper | 100.0% | 100.0% | 1.0 | 3030.5s | C0/U0 |
| ⚠️ | Q10 | contradiction_check | 100.0% | 100.0% | 0.667 | 4292.0s | C0/U12 |
| ✅ | Q11 | out_of_scope | N/A | N/A | — | 316.9s | C0/U0 |
| ❌ | Q12 | false_premise | N/A | N/A | 0.0 | 2468.2s | C0/U3 |

## 逐題細節

### ✅ Q01 · single_paper

**問題**：In the study of JPH203's preincubation inhibitory effect on LAT1, how does preincubation change JPH203's inhibitory potency? Give the reported values.

- detected_paper：`None`
- 選出論文：['1-s2.0-S1347861320300633-main', '1-s2.0-S0021979710012403-main', '1-s2.0-S1936523321002047-main', 's41421-024-00697-6']
- gold_papers：['1-s2.0-S1347861320300633-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：1.0
- 延遲：2448.9s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
> * JPH203 可透過預孵育效應（preincubation effect）抑制 LAT1 功能 [Chunk 2]。
> * JPH203 的預孵育效應與共同孵育（co-incubation）效應具有協同作用，以增強抑制效果 [Chunk 2]。
> * 當預孵育與共同孵育結合時，可顯著提高抑制效能 [Chunk 1, Chunk 3]。
> * 單獨共同孵育的 $\text{IC}_{50}$ 值為 $99.2 \pm 11.0\text{ nM}$ [Chunk 1, Chunk 3]。
> * 預孵育與共同孵育結合的 $\text{IC}_{50}$ 值為 $34.2 \pm 3.6\text{ nM}$ [Chunk 1, Chunk 3]。
> * 單獨共同孵育的 $K_i$ 值為 $0.46\text{ mM}$ [Chunk 1, Chunk 3]。
> * 預孵育與共同孵育結合的 $K_i$ 值為 $0.37\text{ mM}$ [Chunk 1, Chunk 3]。
> * 預孵育與共同孵育之間的協同作用導致 $\text{IC}_{50}$ 值降低三倍 [Chunk 1]。
> * 預孵育與共同孵育之間的協同作用導致 $K_i$ 值降低 20% [Chunk 1]。
> * 單獨預孵育的 $\text{IC}_{50}$ 值為 $193 \pm 50\text{ nM}$ [Chunk 3]。
> * 在 HT-29 細胞中，JPH203 的預孵育效應具有濃度依賴性 [Chunk 3]。
> * 使用 $0.03\text{ mM}$ JPH203 處理 120 分鐘，使亮氨酸（leucine）攝取量降低至對照組的 63% [Chunk 3]。
> * 使用 $0.1\text{ mM}$ JPH203 處理 120 分鐘，使亮氨酸攝取量降低至對照組的 51% [Chun …（完整內容見 JSON）

---

### ✅ Q02 · single_paper

**問題**：What is the hybrid process used for the asymmetric synthesis of optically pure L-p-boronophenylalanine, and what are its key steps?

- detected_paper：`None`
- 選出論文：['bbb0683', 'CMDC-20-e202500059']
- gold_papers：['bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.875
- 延遲：2431.2s　問題標記：{'conflicts': 0, 'unsupported': 6}

**答案預覽**：

> ## 【論文直接依據】
> * L-p-硼苯丙氨酸 (L-BPA) 是一種腫瘤靶向氨基酸 【bbb0683】。
> * L-BPA 是硼中子捕獲治療 (BNCT) 最有效的硼載體之一 【bbb0683】。
> * 合成光學純 L-BPA 的混合製程結合了「不對稱烷基化」以及隨後的「酶促水解」 【bbb0683】。
> * 第一步是保護反應 【bbb0683】。
> * 此保護反應使用 4-溴甲基苯硼酸 (1) 作為起始原料 【bbb0683】。
> * 在此步驟中，使用了 2,2-二甲基-1,3-丙二醇 【bbb0683】。
> * 反應在室溫下的乾燥 THF 中進行 【bbb0683】。
> * 保護反應的攪拌時間為 10 minutes 【bbb0683】。
> * 此步驟產生環狀硼酸酯 (2)，產率為 79% 【bbb0683】。
> * 第二步是不對稱/立體選擇性烷基化 【bbb0683】。
> * 此步驟在 -78°C 的 THF 中進行 【bbb0683】。
> * 使用 n-BuLi 對 (2R)-(-)-2,5-二氫-2-異丙基-3,6-二甲氧基吡嗪 (3) 進行鋰化 【bbb0683】。
> * 鋰化產物與溴化物 (2) 反應，生成加成產物 4 【bbb0683】。
> * 此不對稱烷基化的立體選擇性為 74% e.e. 或 72% 非對映異構體過量 【bbb0683】。
> * 第三步涉及酸處理和脫保護，以合成 L-BPA 甲基酯 (6) 【bbb0683】。
> * 首先，在 5°C 下使用 THF 中的 0.1 N 鹽酸 (HCl) 【bbb0683】。
> * 然後，使用 3% 氨水將 pH 調節至 8.5 以獲得粗產物 5 【bbb0683】。
> * 在獲取粗產物 5 的過程中，必須通過蒸餾除去 D-缬氨酸甲基酯 (D-Val-OMe) 【bbb0683】。
> * 然後將粗產物 5 與 1 N HCl 和 THF 在 …（完整內容見 JSON）

---

### ✅ Q03 · figure_dependent

**問題**：According to the cryo-EM structure, how does JPH203 bind to and inhibit the LAT1-4F2hc complex?

- detected_paper：`None`
- 選出論文：['s41421-024-00697-6', '1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 's41467-024-45464-z (1)', 'LAT1 ChemComm 2026']
- gold_papers：['s41421-024-00697-6']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：1.0
- 延遲：2586.6s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
> * JPH203 結合於 LAT1 的傳統底物結合口袋中 【s41421-024-00697-6】。
> * JPH203 的分子結構包含一個具有苯丙氨酸（phenylalanine）骨架的親水性頭部基團 【s41421-024-00697-6】。
> * JPH203 的分子結構包含一個具有 5-amino-2-(3-aminophenyl) benzoxazole 骨架的疏水性尾部基團 【s41421-024-00697-6】。
> * JPH203 親水性頭部的 $\alpha$-氨基與 LAT1 之 TM1 和 TM6 去螺旋區域（uncoiled regions）的主鏈原子形成氫鍵網絡 【s41421-024-00697-6】。
> * JPH203 親水性頭部的 $\alpha$-羧基與 LAT1 之 TM1 和 TM6 去螺旋區域的主鏈原子形成氫鍵網絡 【s41421-024-00697-6】。
> * JPH203 二鹵酪胺酸（dihalo-tyrosine）核心上的氯原子與 Tyr259 形成鹵鍵 【s41421-024-00697-6】。
> * JPH203 的疏水性尾部嵌入一個疏水性口袋中 【s41421-024-00697-6】。
> * 此疏水性口袋由 TM1 (Ile63, Ile64, Ile68) 組成 【s41421-024-00697-6】。
> * 此疏水性口袋由 TM3 (Ile140, Ile147, Ile148) 組成 【s41421-024-00697-6】。
> * 此疏水性口袋由 TM6 (Phe252, Gly255) 組成 【s41421-024-00697-6】。
> * 此疏水性口袋由 TM10 (Ile397, Phe400, Phe403) 組成 【s41421-024-00697-6】。
> * JPH203 尾部的氨基與 T …（完整內容見 JSON）

---

### ✅ Q04 · table_lookup

**問題**：What are the main degradation products and impurities of BPA identified by HPLC, and under which storage conditions do they form?

- detected_paper：`None`
- 選出論文：['1-s2.0-S0731708511003451-main', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water']
- gold_papers：['1-s2.0-S0731708511003451-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.92
- 延遲：1977.9s　問題標記：{'conflicts': 0, 'unsupported': 4}

**答案預覽**：

> ## 【論文直接依據】
> 
> **降解產物與條件**
> * 在鹼性條件下會產生酪胺酸 (Tyrosine) [1-s2.0-S0731708511003451-main]。
> * 在氧化條件下會產生酪胺酸 (Tyrosine) [1-s2.0-S0731708511003451-main]。
> * 酪胺酸在氧化條件下降解速度極快 [1-s2.0-S0731708511003451-main]。
> * 在冷凍乾燥的 BPA/甘露醇藥物中會產生苯丙胺酸 (Phenylalanine) [1-s2.0-S0731708511003451-main]。
> * 苯丙胺酸的產生是一個緩慢的過程 [1-s2.0-S0731708511003451-main]。
> * 苯丙胺酸的產生具有溫度依賴性 [1-s2.0-S0731708511003451-main]。
> * 將冷凍乾燥的 BPA/甘露醇藥物在 $40^\circ\text{C}$ 下儲存 6 個月，會產生約 1%（相對於 BPA 質量）的苯丙胺酸 [1-s2.0-S0731708511003451-main]。
> 
> **合成雜質**
> * BrPD (2-(4-Bromophenyl)-[1,3]dioxane) 是 BPA 的合成雜質 [1-s2.0-S0731708511003451-main]。
> * FBBA (4-Formylbenzeneboronic acid) 是 BPA 的合成雜質 [1-s2.0-S0731708511003451-main]。
> * BDPA (2-tert-Butoxycarbonylamino-3-[4-(5,5-dimethyl-[1,3,2]dioxaborinan-2-yl)-phenyl]acrylic acid methyl ester) 是 BPA 的合成雜質 [1-s2.0-S07317085110 …（完整內容見 JSON）

---

### ⚠️ Q05 · single_paper

**問題**：What is the solvent-free green protocol for N-Boc protection of amines using picric acid as a catalyst, and what reaction conditions are used?

- detected_paper：`None`
- 選出論文：['1-s2.0-S2773223124000268-main']
- gold_papers：['1-s2.0-S2773223124000268-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.389
- 延遲：3072.1s　問題標記：{'conflicts': 0, 'unsupported': 15}

**答案預覽**：

> ## 【論文直接依據】
> * 苦味酸被用作 Brønsted 酸或有機催化劑，用於胺的 N-Boc 保護 [Chunk 1, Chunk 2]。
> * 所使用的苦味酸量為 0.2 equivalents [Chunk 1, Chunk 2]。
> * 或者，所使用的苦味酸量為 2 mol% [Chunk 1, Chunk 2]。
> * 反應所使用的試劑為二叔丁基二碳酸酯（也稱為 $(\text{Boc})_2\text{O}$ 或 Boc 酸酐）[Chunk 1]。
> * 反應在無溶劑（neat）條件下進行 [Chunk 1, Chunk 2]。
> * 反應在室溫下進行 [Chunk 1, Chunk 2]。
> * 最佳反應時間為 60 分鐘 [Chunk 1, Chunk 2]。
> * 此方法適用於芳香族胺 [Chunk 1]。
> * 此方法適用於雜環芳香族胺 [Chunk 1]。
> * 此方法適用於開鏈脂肪族胺 [Chunk 1]。
> * 此方法適用於環狀脂肪族胺 [Chunk 1]。
> * 最高產率可達 98% [Chunk 1, Chunk 2]。
> * 在最佳條件下（0.2 eq 催化劑，室溫，60 分鐘），芳香族一級胺的產率可達 95% [Chunk 1, Chunk 2]。
> * 在反應機制中，苦味酸首先活化 Boc 酸酐的羰基氧原子 [Chunk 1]。
> * 隨後，胺的氮原子提供一對孤對電子攻擊 Boc 酸酐的羰基碳 [Chunk 1]。
> * 此過程以質子取代以及消除二氧化碳 ($\text{CO}_2$) 作為副產物而結束 [Chunk 1]。
> * 此過程以消除叔丁醇 (t-BuOH) 作為副產物而結束 [Chunk 1]。
> * 這些步驟最終形成了 N-Boc 保護化合物 [Chunk 1]。
> 
> ## 【跨文獻推論】
> * 跨論文推論（基於 [Chunk 1] 和 [Chunk 2]）： …（完整內容見 JSON）

---

### ⚠️ Q06 · multi_chunk

**問題**：How does poly(vinyl alcohol) enhance the therapeutic effect of 4-L-boronophenylalanine in neutron capture therapy of thoracic tumors? Explain the proposed mechanism with supporting data.

- detected_paper：`None`
- 選出論文：['1-s2.0-S0378517325007926-main', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'CMDC-20-e202500059']
- gold_papers：['1-s2.0-S0378517325007926-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：75.0%　grounding：0.615
- 延遲：2538.6s　問題標記：{'conflicts': 0, 'unsupported': 8}

**答案預覽**：

> ## 【論文直接依據】
> * 聚乙烯醇 (PVA) 與多個 4-L-硼苯丙氨酸 (BPA) 分子通過硼酸酯形成稱為 PVA-BPA 的複合物 [1-s2.0-S0378517325007926-main]。
> * PVA-BPA 複合物保留了對於 LAT1（L 型氨基酸轉運蛋白 1）至關重要的類苯丙氨酸結構 [1-s2.0-S0378517325007926-main]。
> * PVA-BPA 與 LAT1 相互作用 [1-s2.0-S0378517325007926-main]。
> * PVA-BPA 通過 LAT1 介導的內吞作用被攝入癌細胞中 [1-s2.0-S0378517325007926-main]。
> * PVA-BPA 定位於細胞內的內體/溶酶體 (endo-/lysosomes) 中 [1-s2.0-S0378517325007926-main]。
> * PVA-BPA 的攝取路徑增加了 BPA 的細胞攝取量 [1-s2.0-S0378517325007926-main]。
> * PVA-BPA 的攝取路徑延緩了 BPA 從細胞質中的外排 (efflux) [1-s2.0-S0378517325007926-main]。
> * 外排的延緩延長了 BPA 在腫瘤中的滯留時間 [1-s2.0-S0378517325007926-main]。
> * 通過對 LAT1 的主動靶向，PVA-BPA 在 1 hour 內高效地積聚並滯留在腫瘤中 [1-s2.0-S0378517325007926-main]。
> * PVA-BPA 在正常器官中的積聚有限 [1-s2.0-S0378517325007926-main]。
> * 研究人員使用山梨醇 (sorbitol) 作為增溶劑，開發了一種優化的 PVA-sorbitol-BPA 配方 [1-s2.0-S0378517325007926-m …（完整內容見 JSON）

---

### ⚠️ Q07 · figure_dependent

**問題**：What is the water-stable boroxine structure reported, and what role do the dynamic covalent bonds play in its fluoride binding and hydrogel formation?

- detected_paper：`None`
- 選出論文：['41467_2024_Article_45464', 's41467-024-45464-z (1)']
- gold_papers：['41467_2024_Article_45464']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.68
- 延遲：4219.4s　問題標記：{'conflicts': 0, 'unsupported': 11}

**答案預覽**：

> ## 【論文直接依據】
> 
> **水穩定硼氧環結構與性質**
> * 報導的水穩定硼氧環結構為 $\text{HO-PBA}$ 三聚體–$\text{H}_2\text{O}$ 錯合物 [s41467-024-45464-z]。
> * 當 $\text{HO-PBA}$ 二聚體在室溫下接觸水時，會迅速轉化為三聚體，從而形成此結構 [s41467-024-45464-z]。
> * $\text{HO-PBA}$ 三聚體的穩定範圍為 $2.4 < \text{pH} < 9$ [s41467-024-45464-z]。
> * $\text{HO-PBA}$ 三聚體在室溫下可保持穩定長達 7 天 [s41467-024-45464-z]。
> * 此穩定性已透過 $^1\text{H NMR}$ 量測驗證 [s41467-024-45464-z]。
> * 此穩定性已透過 $\text{ESI-Q-TOF MS}$ 量測驗證 [s41467-024-45464-z]。
> * 此穩定性已透過 $\text{UV-Raman}$ 量測驗證 [s41467-024-45464-z]。
> 
> **動態共價鍵 (DCBs) 的作用**
> * 水穩定硼氧環結構包含與水兼容的 B–O 動態共價鍵 (DCBs) [s41467-024-45464-z]。
> * 這些 B–O DCBs 在室溫下可進行快速交換 [s41467-024-45464-z]。
> * 相比之下，傳統硼氧環的 B–O 鍵交換需要在甲苯中加熱至 $60\text{ °C}$ 並持續 8 小時 [s41467-024-45464-z]。
> 
> **氟化物結合**
> * $\text{HO-PBA}$ 三聚體能選擇性地識別水溶液中的氟離子 ($\text{F}^-$) [s41467-024-45464-z]。
> * $\text{HO-PBA}$ 三聚體對 $\text …（完整內容見 JSON）

---

### ✅ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.868
- 延遲：3433.2s　問題標記：{'conflicts': 0, 'unsupported': 6}

**答案預覽**：

> ## 【論文直接依據】
> 
> **同位素富集與成本效益**
> * $^{10}\text{B}$ 的價格比普通硼酸高出 1000 倍以上 [CMDC-20-e202500059]。
> * $^{10}\text{B}$ 的原子成本主導了 L-BPA 合成的經濟性 [CMDC-20-e202500059]。
> * 硼引入後的總步驟數和產率是評估 L-BPA 合成路徑成本效益最關鍵的參數 [CMDC-20-e202500059]。
> * 依賴 $\text{B}_2\text{pin}_2$ 的合成路徑較不受青睞，因為它們會浪費 $^{10}\text{B}$ 原子，導致成本極高 [CMDC-20-e202500059]。
> * $^{10}\text{B}$ 通常透過化學交換蒸餾法從 $^{11}\text{B}$ 中分離 [CMDC-20-e202500059]。
> * 分離出的 $^{10}\text{B}$ 隨後被轉化為硼酸酯或水解為 $^{10}\text{B}$ 硼酸 [CMDC-20-e202500059]。
> 
> **Pd 催化與金屬催化路徑**
> * Pd 催化的交叉耦合直接硼化策略具有良好的產率 [CMDC-20-e202500059]。
> * 在 Pd 催化路徑中，$\text{B}_2\text{pin}_2/\text{DMSO}/\text{KOAc}$ 的組合比 $\text{HBpin}$ 更有效，後者的產率 $\le 48\%$ [CMDC-20-e202500059]。
> * 森田製藥公司（由 Kamekawa 等人發明）的方法報告了極高的產率 [CMDC-20-e202500059]。
> * 森田製藥方法的倒數第二步受限於 Pd 催化劑的使用 [CMDC-20-e202500059]。
> * 使用 $[(\text{COD})\text{Rh}(\text{R-Prop …（完整內容見 JSON）

---

### ✅ Q09 · cross_paper

**問題**：Several papers target LAT1 for cancer therapy. What different therapeutic strategies targeting LAT1 are described across the literature, and how do they differ in mechanism?

- detected_paper：`None`
- 選出論文：['LAT1 ChemComm 2026', '1-s2.0-S0378517325007926-main', '1-s2.0-S1347861320300633-main', 's41421-024-00697-6', '1-s2.0-S1936523321002047-main']
- gold_papers：['1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 'LAT1 ChemComm 2026', 's41421-024-00697-6']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：1.0
- 延遲：3030.5s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
> 
> **基因干預策略**
> * 針對 LAT1 的基因方法涉及降低 LAT1 mRNA 表達水平（mRNA 敲低） 【1-s2.0-S1347861320300633-main】。
> * 降低 LAT1 mRNA 表達可顯著減緩前列腺癌細胞的增殖、侵襲與遷移 【1-s2.0-S1347861320300633-main】。
> 
> **小分子抑制劑 (JPH203)**
> * JPH203 是一種用於阻斷 LAT1 功能的高選擇性且強效的抑制劑 【1-s2.0-S1347861320300633-main】。
> * JPH203 的化學名稱為 [(S)-2-amino-3-(4-((5-amino-2-phenylbenzo [d] oxazol-7-yl) methoxy)-3, 5-dichlorophenyl) propanoic acid] 【1-s2.0-S1936523321002047-main】。
> * JPH203 通過以競爭方式與其底物結合來抑制 LAT1 【1-s2.0-S1347861320300633-main】。
> * JPH203 的 $\text{IC}_{50}$ 值在 $60\text{--}140\text{ nM}$ 之間 【1-s2.0-S1347861320300633-main】。
> * JPH203 在多種癌細胞和腫瘤負擔動物模型中顯示出抗增殖效果 【1-s2.0-S1347861320300633-main】。
> * 初次人體臨床試驗的初步結果顯示，JPH203 對於部分結腸癌和膽道癌患者具有耐受性且有效 【1-s2.0-S1347861320300633-main】。
> * JPH203 已進入針對進展期實體腫瘤患者的 I 期和 II 期臨床試驗 【1-s2.0-S1936523321002047-main】。
> * JPH2 …（完整內容見 JSON）

---

### ⚠️ Q10 · contradiction_check

**問題**：Do the papers agree on the hydrolytic stability of boroxine structures? Describe any agreements or differences between them.

- detected_paper：`None`
- 選出論文：['Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation', '41467_2024_Article_45464', 's41467-024-45464-z (1)', 'a-method-for-the-deprotection-of-alkylpinacolyl-boronate-esters']
- gold_papers：['41467_2024_Article_45464', 'Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.667
- 延遲：4292.0s　問題標記：{'conflicts': 0, 'unsupported': 12}

**答案預覽**：

> ## 【論文直接依據】
> 
> **來自【s41467-024-45464-z】：**
> * 傳統認知認為，硼氧環（boroxines）在接觸水或潮濕環境時會迅速水解為硼酸（boronic acids）[s41467-024-45464-z]。
> * 傳統硼氧環的快速水解會導致其功能的喪失 [s41467-024-45464-z]。
> * 提高水解穩定性的其中一種策略是透過引入電子捐贈基團（electron-donating groups）來降低路易斯酸性硼位點（Lewis acidic boron sites）的親電性 [s41467-024-45464-z]。
> * 另一種策略則涉及引入大體積基團以增加穩定性 [s41467-024-45464-z]。
> * 與氮供體配體（N-donor ligands）形成加合物是增強水解穩定性的已知方法 [s41467-024-45464-z]。
> * 將三個硼酸單元整合到柔性大環（flexible macrocycles）中以實現熵穩定化（entropic stabilization）是增加穩定性的方法之一 [s41467-024-45464-z]。
> * 部分文獻指出，將單元整合到柔性大環中未能從根本上解決水解不穩定性的問題 [s41467-024-45464-z]。
> * 源自 2-羥基苯硼酸（HO-PBA）的硼氧環結構在水中具有穩定性 [s41467-024-45464-z]。
> * HO-PBA 衍生物在室溫下會自發脫水形成二聚體（dimers）[s41467-024-45464-z]。
> * 在室溫下接觸水後，HO-PBA 二聚體會迅速轉化為 HO-PBA 三聚體–$\text{H}_2\text{O}$ 複合物 [s41467-024-45464-z]。
> * HO-PBA 三聚體的穩定 pH 範圍為 $2.4 < \text{pH} < 9$  …（完整內容見 JSON）

---

### ✅ Q11 · out_of_scope

**問題**：What are the phase III clinical trial overall survival results for BNCT in glioblastoma patients?

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', '1-s2.0-S0378517325007926-main', 'cas0106-0279']
- gold_papers：[]
- 論文選擇命中率：N/A　檢索覆蓋率：N/A　grounding：—
- 延遲：316.9s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ⚠️ **資料來源說明**：本地學術文獻資料庫中未找到與此問題直接相關的內容。以下回答來自模型自身知識，非論文原文，請謹慎參考並自行查證。
> 
> 針對您提出的關於硼中子捕獲治療（Boron Neutron Capture Therapy, BNCT）在膠質母細胞瘤（Glioblastoma, GBM）患者中第三期臨床試驗總生存期（Overall Survival, OS）結果的問題，以下是基於目前學術知識的詳細分析。
> 
> ### 核心結論：缺乏大規模第三期隨機對照試驗 (RCT) 的定論
> 
> 首先必須明確指出：**目前在國際主流醫學文獻中，尚未見到針對膠質母細胞瘤（GBM）且具有高度權威性、大規模樣本量的第三期臨床試驗（Phase III Clinical Trial）之總生存期（OS）最終結果。**
> 
> 大多數關於 BNCT 治療 GBM 的研究仍處於第一期（安全性評估）或第二期（初步療效評估）階段。BNCT 的臨床推進面臨較大的挑戰，主因在於中子源（Neutron Source）的可用性（早期依賴核反應爐，近期才逐漸轉向加速器 BNCT, AB-BNCT）以及患者篩選的複雜性。
> 
> ### 現有研究（第一、二期）的生存趨勢分析
> 
> 雖然缺乏第三期試驗數據，但根據已發表的第二期臨床研究或回顧性分析，BNCT 在 GBM 患者中展現出一定的潛力。以下為其特點：
> 
> 1.  **對比歷史對照組（Historical Controls）：**
>     部分研究顯示，接受 BNCT 治療的患者在總生存期上優於僅接受標準治療（如 Stupp Protocol：手術 + 放療 + 替莫唑胺 TMZ）的歷史數據。然而，由於缺乏隨機分組，這類結果存在較大的選擇性偏差（Selection Bias）。
> 
> 2.  **復發性 GBM (Recurrent GBM) 的表現：**
>     BNCT 在復發性  …（完整內容見 JSON）

---

### ❌ Q12 · false_premise

**問題**：Since BPA is administered orally in BNCT, what oral bioavailability values are reported in these papers?

- detected_paper：`None`
- 選出論文：['cas0106-0279', 'Chem Biol Drug Des - 2013 - Achilli - Folic Acid‐Conjugated 4‐Amino‐Phenylboronate  a Boron‐Containing Compound Designed']
- gold_papers：[]
- 論文選擇命中率：N/A　檢索覆蓋率：N/A　grounding：0.0
- 延遲：2468.2s　問題標記：{'conflicts': 0, 'unsupported': 3}

**答案預覽**：

> ## 【論文直接依據】
> 
> * 提供之文本中未報告 BPA 的口服生物利用度數值。
> * 根據 [Fact 4]，BNCT 硼載體必須透過靜脈注射給藥。
> 
> ## 【跨文獻推論】
> 
> * [Insufficient Evidence] 缺乏足夠的文獻依據來確認關於 BPA 口服給藥的藥代動力學數值或推論，因為提供的事實中不包含其口服生物利用度的數據。
> 
> ## 【知識延伸與推測】
> 
> * [Unverified] 關於使用靜脈注射以避開肝臟首過效應（first-pass metabolism）並確保血漿濃度受控的主張，在已知事實清單中缺乏支持的前題。
> * [Unverified] 推論口服生物利用度會因胃腸道吸收限制而較低或波動較大，已超出可用證據之範圍，因為提供的事實僅討論了癌細胞中透過芳香族氨基酸轉運蛋白對 BPA 的攝取 [Fact 6, Fact 7]。
> 
> ---
> 📋 **答案品質報告**
> 
> 📊 **分段論文依據率：**
> 
>   ❌ 【論文直接依據】：0.0%（0/2 句）
>   ❌ 【跨文獻推論】：0.0%（0/1 句）  ← 跨論文推論，低分為預期範圍
>   ✅ 【知識延伸推測】：100.0%（2/2 句）  ← 知識延伸推測，低分為預期範圍
> 
> ❌ **直引依據率**：0.0%　低（建議縮小問題範圍）
> 
> ⚠️  **以下 3 個陳述未找到明確論文依據，請謹慎參考：**
> 
> - * No oral bioavailability values for BPA are reported in the provided texts.（信心度：31.5%，最近似來源：cas0106-0279-d7164215）
> - * Based on [Fact 4], BNCT boron carriers must be administered via intravenous infusion.（ …（完整內容見 JSON）

---
