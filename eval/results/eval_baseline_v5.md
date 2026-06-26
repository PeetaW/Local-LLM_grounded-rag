# Eval Report — `baseline_v5`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-06-26 02:30
- 題數：12

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.833 |
| 平均 grounding 分數 | 0.741 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 97.5% |
| 平均總延遲 | 917.9s |
| 平均 planning 延遲 | 7.0s |
| 平均 retrieval 延遲 | 363.5s |
| 平均 grounding 延遲 | 43.4s |
| └ 其中 NLI | 26.7s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ✅ | Q01 | single_paper | 100.0% | 100.0% | 1.0 | 1115.6s | C0/U0 |
| ⚠️ | Q02 | single_paper | 100.0% | 100.0% | 0.662 | 957.9s | C0/U25 |
| ✅ | Q03 | figure_dependent | 100.0% | 100.0% | 1.0 | 977.1s | C0/U0 |
| ⚠️ | Q04 | table_lookup | 100.0% | 100.0% | 0.7 | 585.7s | C0/U7 |
| ⚠️ | Q05 | single_paper | 100.0% | 100.0% | 0.36 | 715.7s | C0/U20 |
| ⚠️ | Q06 | multi_chunk | 100.0% | 75.0% | 0.722 | 1035.2s | C0/U6 |
| ⚠️ | Q07 | figure_dependent | 100.0% | 100.0% | 0.636 | 872.5s | C0/U10 |
| ⚠️ | Q08 | cross_paper | 100.0% | 100.0% | 0.727 | 1032.6s | C0/U6 |
| ✅ | Q09 | cross_paper | 100.0% | 100.0% | 0.951 | 1551.2s | C0/U2 |
| ⚠️ | Q10 | contradiction_check | 100.0% | 100.0% | 0.654 | 1303.0s | C0/U10 |
| ✅ | Q11 | out_of_scope | N/A | N/A | — | 353.5s | C0/U0 |
| ✅ | Q12 | false_premise | N/A | N/A | — | 514.8s | C0/U0 |

## 逐題細節

### ✅ Q01 · single_paper

**問題**：In the study of JPH203's preincubation inhibitory effect on LAT1, how does preincubation change JPH203's inhibitory potency? Give the reported values.

- detected_paper：`None`
- 選出論文：['1-s2.0-S1347861320300633-main', '1-s2.0-S0021979710012403-main', '1-s2.0-S1936523321002047-main', 's41421-024-00697-6']
- gold_papers：['1-s2.0-S1347861320300633-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：1.0
- 延遲：1115.6s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
> 
> * 當預孵育（preincubation）與共同孵育（co-incubation）結合時，能顯著增強 JPH203 的抑制效力【1-s2.0-S1347861320300633-main [Chunk 1, Chunk 3]】。
> * 在單一共同孵育分析中，JPH203 的 $\text{IC}_{50}$ 值為 $99.2 \pm 11.0\text{ nM}$【1-s2.0-S1347861320300633-main [Chunk 1, Chunk 3]】。
> * 當預孵育加入共同孵育時，JPH203 的 $\text{IC}_{50}$ 值為 $34.2 \pm 3.6\text{ nM}$【1-s2.0-S1347861320300633-main [Chunk 1, Chunk 3]】。
> * 預孵育與共同孵育之間的協同作用導致 $\text{IC}_{50}$ 值降低三倍【1-s2.0-S1347861320300633-main [Chunk 1]】。
> * 在共同孵育分析中，JPH203 的 $K_i$ 值為 $0.46\text{ mM}$【1-s2.0-S1347861320300633-main [Chunk 1, Chunk 3]】。
> * 在預孵育加上共同孵育的分析中，JPH203 的 $K_i$ 值為 $0.37\text{ mM}$【1-s2.0-S1347861320300633-main [Chunk 1, Chunk 3]】。
> * 預孵育與共同孵育之間的協同作用導致 $K_i$ 值降低 20%【1-s2.0-S1347861320300633-main [Chunk 1]】。
> * 單獨預孵育時，JPH203 的 $\text{IC}_{50}$ 值為 $193 \pm 50\text{ nM}$【1-s2.0-S1347 …（完整內容見 JSON）

---

### ⚠️ Q02 · single_paper

**問題**：What is the hybrid process used for the asymmetric synthesis of optically pure L-p-boronophenylalanine, and what are its key steps?

- detected_paper：`None`
- 選出論文：['bbb0683', 'CMDC-20-e202500059']
- gold_papers：['bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.662
- 延遲：957.9s　問題標記：{'conflicts': 0, 'unsupported': 25}

**答案預覽**：

> ## 【論文直接依據】
> 
> **混合製程概述**
> * 合成光學純 L-p-硼苯丙氨酸 (L-BPA) 的混合製程結合了對映選擇性烷基化與隨後的酵素水解 【bbb0683】。
> * 此混合製程的特點是反應步驟較少 【bbb0683】。
> * 該混合製程的執行與後處理過程簡單 【bbb0683】。
> * L-BPA 在硼中子捕獲治療 (BNCT) 中可作為有效的腫瘤靶向硼載體 【bbb0683】。
> * 使用手性輔助劑 (2R)-(-)-2,5-dihydro-2-isopropyl-3,6-dimethoxypyrazine 並在 pH 5.0 下使用糜蛋白酶 (chymotrypsin) 進行酵素水解，可製得光學純度為 100% e.e. 的 L-BPA 【bbb0683】。
> 
> **步驟 1：硼酸的保護**
> * 第一步涉及將 4-boronomethylphenylboronic acid (1) 的二羥基硼基保護為環狀硼酸酯，以產生化合物 2 【bbb0683】。
> * 化合物 1 由 4-bromotoluene 製備而來 【bbb0683】。
> * 合成化合物 2 的反應使用 2,2-dimethyl-1,3-propanediol 【bbb0683】。
> * 此反應在乾燥 THF 中進行 【bbb0683】。
> * 反應在室溫下進行 【bbb0683】。
> * 此步驟的攪拌時間為 10 分鐘 【bbb0683】。
> * 化合物 2 的產率為 79% 【bbb0683】。
> * 使用的具體量：5.99 g (27.9 mmol) 的化合物 19（在此步驟中被引用為起始原料） 【bbb0683】。
> * 使用的具體量：2.91 g (27.9 mmol) 的 2,2-dimethyl-1,3-propanediol 【bbb0683】。
> * 使用的具體量：30 ml 的乾燥 THF 【bbb06 …（完整內容見 JSON）

---

### ✅ Q03 · figure_dependent

**問題**：According to the cryo-EM structure, how does JPH203 bind to and inhibit the LAT1-4F2hc complex?

- detected_paper：`None`
- 選出論文：['s41421-024-00697-6', '1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 's41467-024-45464-z (1)', 'LAT1 ChemComm 2026']
- gold_papers：['s41421-024-00697-6']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：1.0
- 延遲：977.1s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
> 
> **結合位點與結構組成**
> * JPH203 結合於 LAT1 的傳統底物結合口袋中 (s41421-024-00697-6)。
> * JPH203 的分子結構包含一個具有苯丙氨酸（phenylalanine）骨架的親水性頭部 (s41421-024-00697-6)。
> * JPH203 的分子結構包含一個具有 5-amino-2-(3-aminophenyl) benzoxazole 骨架的疏水性尾部 (s41421-024-00697-6)。
> 
> **低溫電子顯微鏡（Cryo-EM）實驗細節**
> * Cryo-EM 結構解析了處於向外開放構象（outward-facing conformation）的 JPH203 與 LAT1-4F2hc 複合物 (s41421-024-00697-6)。
> * 該複合物的整體解析度為 3.30 Å (s41421-024-00697-6)。
> * JPH203 的聚焦精修（focused refinement）解析度為 3.25 Å (s41421-024-00697-6)。
> * 實驗中使用了 $500\ \mu\text{M}$ 的 JPH203 (s41421-024-00697-6)。
> 
> **分子間交互作用**
> * 親水性頭部的 $\alpha$-氨基與 TM1 和 TM6 解旋區域（unwound regions）的主鏈原子形成氫鍵網絡 (s41421-024-00697-6)。
> * 親水性頭部的 $\alpha$-羧基與 TM1 和 TM6 解旋區域的主鏈原子形成氫鍵網絡 (s41421-024-00697-6)。
> * JPH203 的二鹵酪胺酸（dihalo-tyrosine）核心上的氯原子與 Tyr259 形成鹵鍵（halogen bond）(s41421-024-00697-6)。
> * 疏水性尾部嵌 …（完整內容見 JSON）

---

### ⚠️ Q04 · table_lookup

**問題**：What are the main degradation products and impurities of BPA identified by HPLC, and under which storage conditions do they form?

- detected_paper：`None`
- 選出論文：['1-s2.0-S0731708511003451-main', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water']
- gold_papers：['1-s2.0-S0731708511003451-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.7
- 延遲：585.7s　問題標記：{'conflicts': 0, 'unsupported': 7}

**答案預覽**：

> ## 【論文直接依據】
> 
> **降解產物與條件**
> * BPA 在鹼性條件下會降解為酪胺酸 (tyrosine) (Chunk 1)。
> * BPA 在氧化條件下會降解為酪胺酸 (tyrosine) (Chunk 1)。
> * 在氧化條件下，BPA 降解為酪胺酸 (tyrosine) 的速度極快 (Chunk 1)。
> * 在水鹼溶液中，使用 $100\text{ mM NaOH}$ 在 $55^\circ\text{C}$ 下處理 24 小時會導致 BPA 強制降解為酪胺酸 (tyrosine) (Chunk 2)。
> * 在含有 $6\text{ mM H}_2\text{O}_2$ 的溶液中，BPA 會極速降解為酪胺酸 (tyrosine) (Chunk 2)。
> * BPA/甘露醇 (mannitol) 凍乾藥物表現出緩慢且與溫度相關的降解，產生苯丙胺酸 (phenylalanine) (Chunk 1)。
> * 當在 $40^\circ\text{C}$ 下儲存 6 個月時，BPA/甘露醇 (mannitol) 凍乾藥物產生的苯丙胺酸 (phenylalanine) 約為 BPA 質量的 1% (Chunk 1, Chunk 2)。
> 
> **HPLC 分析特性**
> * 酪胺酸 (tyrosine) 在 230 nm 處被檢出，保留時間為 4.5 分鐘 (Chunk 1)。
> * 苯丙胺酸 (phenylalanine) 在 256 nm 處被檢出，保留時間為 11.0 分鐘 (Chunk 1)。
> 
> **合成雜質與 HPLC 分析**
> * 合成雜質 BrPD (2-(4-Bromophenyl)-[1,3]dioxane) 在 256 nm 處被檢出，保留時間為 17.3 分鐘 (Chunk 1)。
> * 合成雜質 FBBA (4-Formylbenzeneboronic acid)  …（完整內容見 JSON）

---

### ⚠️ Q05 · single_paper

**問題**：What is the solvent-free green protocol for N-Boc protection of amines using picric acid as a catalyst, and what reaction conditions are used?

- detected_paper：`None`
- 選出論文：['1-s2.0-S2773223124000268-main']
- gold_papers：['1-s2.0-S2773223124000268-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.36
- 延遲：715.7s　問題標記：{'conflicts': 0, 'unsupported': 20}

**答案預覽**：

> ## 【論文直接依據】
> 
> * 苦味酸被用作胺類 N-Boc 保護的催化劑 [Chunk 1]。
> * $\text{(Boc)}_2\text{O}$ (Boc 酐) 被用作反應試劑 [Chunk 1]。
> * 最適反應條件為無溶劑/純物質 (solvent-free/neat) [Chunk 1, Chunk 2]。
> * 最適反應溫度為室溫 [Chunk 1, Chunk 2]。
> * 最適反應時間為 60 分鐘 [Chunk 1, Chunk 2]。
> * 在最適條件下使用的苦味酸催化劑量為 0.2 當量 (equivalents) [Chunk 1, Chunk 2]。
> * 在無溶劑條件下，使用 2 mol% 的苦味酸作為 Brønsted 酸催化劑，最高產率可達 98% [Chunk 1, Chunk 2]。
> * 芳香族一級胺在最適條件下的產率為 95% [Chunk 1, Chunk 2]。
> * 此方法適用於芳香族胺，包括 2-氨基苯酚 (2-aminophenol)、萘-1-胺 (naphthyl-1-amine) 和鹵代芳香族胺 [Chunk 1]。
> * 此方法適用於雜環芳香族胺，包括喹啉-4-胺 (quinolin-4-amine)、苯噻唑-2-胺 (benzothiazole-2-amine) 和 1H-咪唑 (1H-imidazole) [Chunk 1]。
> * 此方法適用於開鏈和環狀脂肪族胺，包括哌啶 (piperidine)、嗎啉 (morpholine)、吡咯烷 (pyrrolidine) 和環己胺 (cyclohexylamine) [Chunk 1]。
> * 在提出的機理中，苦味酸活化 $\text{Boc}$ 酐中羰基的氧原子 [Chunk 1]。
> * 在提出的機理中，胺的氮原子攻擊 $\text{Boc}$ 酐的羰基碳 [Chunk 1]。
> * 提出 …（完整內容見 JSON）

---

### ⚠️ Q06 · multi_chunk

**問題**：How does poly(vinyl alcohol) enhance the therapeutic effect of 4-L-boronophenylalanine in neutron capture therapy of thoracic tumors? Explain the proposed mechanism with supporting data.

- detected_paper：`None`
- 選出論文：['1-s2.0-S0378517325007926-main', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'CMDC-20-e202500059']
- gold_papers：['1-s2.0-S0378517325007926-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：75.0%　grounding：0.722
- 延遲：1035.2s　問題標記：{'conflicts': 0, 'unsupported': 6}

**答案預覽**：

> ## 【論文直接依據】
> * 聚乙烯醇 (PVA) 通過硼酸酯與多個 BPA 分子形成複合物，稱為 PVA-BPA (1-s2.0-S0378517325007926-main)。
> * PVA-BPA 複合物保留了靶向 LAT1 所需的關鍵類苯丙氨酸結構 (1-s2.0-S0378517325007926-main)。
> * 傳統 BPA 通過 LAT1 從細胞外空間運輸至細胞質 (1-s2.0-S0378517325007926-main)。
> * 當細胞外傳統 BPA 濃度降低時，它會通過與細胞外酪胺酸交換而流出細胞 (1-s2.0-S0378517325007926-main)。
> * PVA-BPA 通過 LAT1 介導的內吞作用被攝取到癌細胞中 (1-s2.0-S0378517325007926-main)。
> * 進入細胞後，PVA-BPA 定位於內體/溶酶體 (endo-/lysosomes) 中 (1-s2.0-S0378517325007926-main)。
> * PVA-BPA 的攝取機制增加了細胞攝取量 (1-s2.0-S0378517325007926-main)。
> * PVA-BPA 的攝取機制延緩了 BPA 從細胞質的流出 (1-s2.0-S0378517325007926-main)。
> * 通過對 LAT1 的主動靶向，PVA-BPA 在 1 h 內高效地積聚於腫瘤中 (1-s2.0-S0378517325007926-main)。
> * PVA-BPA 在腫瘤中表現出長期滯留性 (1-s2.0-S0378517325007926-main)。
> * PVA-BPA 限制了在正常器官中的積聚 (1-s2.0-S0378517325007926-main)。
> * 使用山梨醇 (sorbitol) 作為增溶劑，開發了一種優化配方「PVA-sorbitol-BPA」 …（完整內容見 JSON）

---

### ⚠️ Q07 · figure_dependent

**問題**：What is the water-stable boroxine structure reported, and what role do the dynamic covalent bonds play in its fluoride binding and hydrogel formation?

- detected_paper：`None`
- 選出論文：['41467_2024_Article_45464', 's41467-024-45464-z (1)']
- gold_papers：['41467_2024_Article_45464']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.636
- 延遲：872.5s　問題標記：{'conflicts': 0, 'unsupported': 10}

**答案預覽**：

> ⚠️ **資料可能不足**：檢索內容僅部分涵蓋此問題，以下回答可能不完整，請謹慎參考並自行查證。
> 
> ## 【論文直接依據】
> 
> **水穩定硼氧環結構**
> * 所報導的水穩定硼氧環結構源自於 2-羥基苯硼酸 (HO-PBA) 【s41467-024-45464-z (1)】。
> * HO-PBA 在室溫下會自發脫水形成二聚體 【s41467-024-45464-z (1)】。
> * 在室溫下接觸水後，HO-PBA 二聚體會迅速轉化為水穩定的三聚體硼氧環結構 【s41467-024-45464-z (1)】。
> * HO-PBA 三聚體的穩定範圍為 $2.4 < \text{pH} < 9$ 【s41467-024-45464-z (1)】。
> * $\text{CH}_3\text{-HO-PBA}$ 三聚體的穩定範圍為 $2.4 < \text{pH} < 10$ 【s41467-024-45464-z (1)】。
> 
> **動態共價鍵在氟化物結合中的作用**
> * 此水穩定硼氧環結構的 B–O 動態共價鍵在室溫水溶液中可進行快速交換反應 【s41467-024-45464-z (1)】。
> * 相比之下，傳統的硼氧環結構需要在甲苯溶液中加熱至 $60^\circ\text{C}$ 並持續 8 小時才能進行交換 【s41467-024-45464-z (1)】。
> * HO-PBA 三聚體在水溶液中能選擇性地識別氟離子 ($\text{F}^-$) 【s41467-024-45464-z (1)】。
> * 在 $^1\text{H NMR}$ 光譜中，加入 $\text{F}^-$ 會使 HO-PBA 三聚體的峰值從四組增加到十組以上 【s41467-024-45464-z (1)】。
> * 加入 $\text{Cl}^-$、$\text{Br}^-$ 或 $\text{I}^-$ 對 HO-PB …（完整內容見 JSON）

---

### ⚠️ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.727
- 延遲：1032.6s　問題標記：{'conflicts': 0, 'unsupported': 6}

**答案預覽**：

> ⚠️ **資料可能不足**：檢索內容僅部分涵蓋此問題，以下回答可能不完整，請謹慎參考並自行查證。
> 
> 根據提供的文本，關於 4-硼酸-L-苯丙氨酸 (L-BPA) 合成路徑的可用資訊有限。不同論文之間缺乏關於具體成本或工業可擴展性指標的詳細對比數據。
> 
> ## 【論文直接依據】
> 
> **通用 $^{10}\text{B}$ 富集 L-BPA 合成**
> * 目前對於 $^{10}\text{B}$ 富集的 4-硼酸-L-苯丙氨酸 (L-BPA) 的合成路徑尚未達成共識 (【CMDC-20-e202500059】)。
> * 缺乏關於 $^{10}\text{B}$ 富集 L-BPA 合成的學術方法與專利方法之間的詳細對比分析 (【CMDC-20-e202500059】)。
> 
> **「混合製程」合成（用於光學純 L-BPA）**
> * 此路徑利用 (2R)-(-)-2,5-dihydro-2-isopropyl-3,6-dimethoxypyrazine (bislactimether) 與受保護的 4-溴甲基苯硼酸衍生物進行對映選擇性烷基化 (【bbb0683】)。
> * 此路徑採用胰蛋白酶 (trypsin) 在 pH 5.0 下進行酶水解以獲取 L-BPA (【bbb0683】)。
> * 此路徑實現了 100% e.e. 的光學純度 (【bbb0683】)。
> * 與先前的方法相比，此路徑所需的反應步驟較少 (【bbb0683】)。
> * 與先前的方法相比，此路徑的操作更簡單 (【bbb0683】)。
> * 與先前的方法相比，此路徑的後處理更簡單 (【bbb0683】)。
> * 與先前的方法相比，此路徑不需要額外的分離步驟 (【bbb0683】)。
> 
> ## 【跨文獻推論】
> 
> * **可擴展性與成本效益：** 雖然未提供具體的貨幣成本或生產量，但可以推論 (【bbb0683】) 中描述的「混合製程」 …（完整內容見 JSON）

---

### ✅ Q09 · cross_paper

**問題**：Several papers target LAT1 for cancer therapy. What different therapeutic strategies targeting LAT1 are described across the literature, and how do they differ in mechanism?

- detected_paper：`None`
- 選出論文：['LAT1 ChemComm 2026', '1-s2.0-S0378517325007926-main', '1-s2.0-S1347861320300633-main', 's41421-024-00697-6', '1-s2.0-S1936523321002047-main']
- gold_papers：['1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 'LAT1 ChemComm 2026', 's41421-024-00697-6']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.951
- 延遲：1551.2s　問題標記：{'conflicts': 0, 'unsupported': 2}

**答案預覽**：

> 根據提供的文獻，已開發出數種針對 LAT1 的癌症治療策略。這些策略範圍涵蓋基因沉默、小分子抑制、基於肽的干擾以及底物介導的遞送。
> 
> ## 【論文直接依據】
> 
> ### 策略 1：基因沉默（mRNA 敲低/Knock-down）
> *   降低 LAT1 mRNA 的表達可顯著減緩前列腺癌細胞的增殖、侵襲和遷移【1-s2.0-S1347861320300633-main】。
> 
> ### 策略 2：小分子抑制（JPH203）
> *   **機制：**
>     *   JPH203 是一種酪胺酸衍生物，能競爭性地抑制 LAT1【1-s2.0-S1347861320300633-main】。
>     *   其競爭特性表現為 $\text{K}_m$ 顯著增加，而 $\text{V}_{\max}$ 僅有輕微波動【1-s2.0-S1347861320300633-main】。
>     *   JPH203 並非透過降低 LAT1 mRNA 或蛋白質表達水平來實現其抑制效果【1-s2.0-S1347861320300633-main】。
>     *   該抑制作用是暫時性的；在藥物移除後，活性在 120 分鐘時恢復至對照組的 97%，在 240 分鐘時恢復至 95%【1-s2.0-S1347861320300633-main】。
> *   **效力與功效：**
>     *   JPH203 的 $\text{IC}_{50}$ 為 $60\text{--}140\text{ nM}$【1-s2.0-S1347861320300633-main】。
>     *   在 HT-29 細胞中，使用 0.3 mM JPH203 預孵育 120 分鐘可將 LAT1 活性降低至對照組的 30%【1-s2.0-S1347861320300633-main】。
>     *   在 HT-29 細胞中，使用  …（完整內容見 JSON）

---

### ⚠️ Q10 · contradiction_check

**問題**：Do the papers agree on the hydrolytic stability of boroxine structures? Describe any agreements or differences between them.

- detected_paper：`None`
- 選出論文：['Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation', '41467_2024_Article_45464', 's41467-024-45464-z (1)', 'a-method-for-the-deprotection-of-alkylpinacolyl-boronate-esters']
- gold_papers：['41467_2024_Article_45464', 'Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.654
- 延遲：1303.0s　問題標記：{'conflicts': 0, 'unsupported': 10}

**答案預覽**：

> ## 【論文直接依據】
> 
> **關於硼氧環（boroxines）的一般穩定性與傳統策略：**
> * 傳統認知認為，硼氧環在水環境中具有水解不穩定性 (41467_2024_Article_45464, s41467-024-45464-z (1))。
> * 與水接觸通常會導致硼氧環迅速水解為硼酸，進而導致功能喪失 (41467_2024_Article_45464, s41467-024-45464-z (1))。
> * 提高水解穩定性的傳統策略包括引入電子捐贈基團（electron-donating groups）(41467_2024_Article_45464, s41467-024-45464-z (1), Chemistry A European J - 2023 - Ono)。
> * 提高水解穩定性的傳統策略包括添加大尺寸（bulky）基團 (41467_2024_Article_45464, s41467-024-45464-z (1), Chemistry A European J - 2023 - Ono)。
> * 提高水解穩定性的傳統策略包括與 N-供體配體（N-donor ligands）形成加合物 (41467_2024_Article_45464, s41467-024-45464-z (1), Chemistry A European J - 2023 - Ono)。
> * 提高水解穩定性的傳統策略包括透過將三個硼酸單元整合到一個柔性大環中來採用熵穩定化（entropy stabilization）(41467_2024_Article_45464, s41467-024-45464-z (1), Chemistry A European J - 2023 - Ono)。
> * 這些傳統的穩定策略尚未從根本上解決硼氧環水解不穩定的問題 (41467_2024_A …（完整內容見 JSON）

---

### ✅ Q11 · out_of_scope

**問題**：What are the phase III clinical trial overall survival results for BNCT in glioblastoma patients?

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', '1-s2.0-S0378517325007926-main', 'cas0106-0279']
- gold_papers：[]
- 論文選擇命中率：N/A　檢索覆蓋率：N/A　grounding：—
- 延遲：353.5s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ⚠️ **資料來源說明**：本地學術文獻資料庫中未找到與此問題直接相關的內容。以下回答來自模型自身知識，非論文原文，請謹慎參考並自行查證。
> 
> **此為模型推測，建議查閱原始文獻確認。**
> 
> 針對您詢問關於硼中子捕獲治療（Boron Neutron Capture Therapy, BNCT）在膠質母細胞瘤（Glioblastoma）患者中的第三期臨床試驗整體生存率（Overall Survival, OS）結果，說明如下：
> 
> **首先，必須指出的是，目前在主流學術文獻中，尚未見到針對膠質母細胞瘤且具有標準化、大規模隨機對照之「第三期臨床試驗」所公布的確定性整體生存率數據。**
> 
> 以下為定性分析其原因與現狀：
> 
> 1.  **研究階段限制**：目前 BNCT 在膠質母細胞瘤治療上的研究大多處於第一期或第二期臨床試驗，或是基於回顧性分析（Retrospective studies）的小規模隊列研究。雖然部分研究顯示出潛在的療效，但尚未進入到能定義標準生存率基準的大規模第三期試驗階段。
> 2.  **技術與設施限制**：BNCT 的實施高度依賴中子源（如核反應爐或加速器），由於全球可用於臨床的設備數量極少，難以在短時間內招募足夠數量的受試者以滿足第三期臨床試驗對樣本數（Sample size）的統計要求。
> 3.  **缺乏統一標準**：不同研究中心使用的硼載體（Boron delivery agents）以及中子照射劑量有所差異，這使得在進行大規模橫向比較（如第三期試驗所需）時面臨較大的標準化挑戰。
> 
> **總結：**
> 由於目前缺乏正式的第三期臨床試驗數據，因此無法提供具體的整體生存率數值。目前的文獻證據主要集中於可行性、安全性以及初步的療效探索。若需獲取最新的臨床進展，建議直接查閱最新的臨床試驗註冊平台（如 ClinicalTrials.gov）或相關專業學會的最新會議論文集。
> 
> * …（完整內容見 JSON）

---

### ✅ Q12 · false_premise

**問題**：Since BPA is administered orally in BNCT, what oral bioavailability values are reported in these papers?

- detected_paper：`None`
- 選出論文：['cas0106-0279', 'Chem Biol Drug Des - 2013 - Achilli - Folic Acid‐Conjugated 4‐Amino‐Phenylboronate  a Boron‐Containing Compound Designed']
- gold_papers：[]
- 論文選擇命中率：N/A　檢索覆蓋率：N/A　grounding：—
- 延遲：514.8s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ⚠️ **誠實棄答**：檢索到的內容與此問題相關，但並不包含可直接回答的資訊。為避免編造不存在的數據，本系統選擇不作答。建議換個問法，或確認該主題是否涵蓋於文獻庫中。

---
