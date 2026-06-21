# Eval Report — `tier1_transctx`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-06-21 16:44
- 題數：12

## 彙總

| 指標 | 值 |
|------|-----|
| 平均 grounding 分數 | 0.706 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 97.5% |
| 平均總延遲 | 2179.5s |
| 平均 planning 延遲 | 7.1s |
| 平均 retrieval 延遲 | 369.3s |
| 平均 grounding 延遲 | 1082.3s |
| └ 其中 NLI | 44.7s |
| └ 其中 gemma4 | 1018.6s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ✅ | Q01 | single_paper | 100.0% | 100.0% | 0.857 | 1361.9s | C0/U2 |
| ⚠️ | Q02 | single_paper | 100.0% | 100.0% | 0.767 | 3854.8s | C0/U18 |
| ✅ | Q03 | figure_dependent | 100.0% | 100.0% | 1.0 | 1010.7s | C0/U0 |
| ⚠️ | Q04 | table_lookup | 100.0% | 100.0% | 0.633 | 2720.4s | C0/U14 |
| ⚠️ | Q05 | single_paper | 100.0% | 100.0% | 0.75 | 3447.9s | C0/U8 |
| ✅ | Q06 | multi_chunk | 100.0% | 75.0% | 0.846 | 1967.5s | C0/U4 |
| ⚠️ | Q07 | figure_dependent | 100.0% | 100.0% | 0.609 | 2405.1s | C0/U12 |
| ⚠️ | Q08 | cross_paper | 100.0% | 100.0% | 0.765 | 2127.4s | C0/U8 |
| ✅ | Q09 | cross_paper | 100.0% | 100.0% | 1.0 | 2431.8s | C0/U0 |
| ⚠️ | Q10 | contradiction_check | 100.0% | 100.0% | 0.538 | 3300.2s | C0/U16 |
| ✅ | Q11 | out_of_scope | N/A | N/A | — | 340.6s | C0/U0 |
| ❌ | Q12 | false_premise | N/A | N/A | 0.0 | 1185.8s | C0/U2 |

## 逐題細節

### ✅ Q01 · single_paper

**問題**：In the study of JPH203's preincubation inhibitory effect on LAT1, how does preincubation change JPH203's inhibitory potency? Give the reported values.

- detected_paper：`None`
- 選出論文：['1-s2.0-S1347861320300633-main', '1-s2.0-S0021979710012403-main', '1-s2.0-S1936523321002047-main', 's41421-024-00697-6']
- gold_papers：['1-s2.0-S1347861320300633-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.857
- 延遲：1361.9s　問題標記：{'conflicts': 0, 'unsupported': 2}

**答案預覽**：

> ## 【論文直接依據】
> * JPH203 的預孵育（preincubation）與共同孵育（co-incubation）具有協同作用，能顯著增強其對 LAT1 的抑制效力 [Fact 2]。
> * 在單獨共同孵育時，JPH203 的 $\text{IC}_{50}$ 值為 $99.2 \pm 11.0\text{ nM}$ [Fact 3]。
> * 當預孵育與共同孵育結合時，JPH203 的 $\text{IC}_{50}$ 值為 $34.2 \pm 3.6\text{ nM}$ [Fact 3]。
> * 在單獨共同孵育時，JPH203 的 $\text{Ki}$ 值為 $0.46\text{ mM}$ [Fact 4]。
> * 當預孵育與共同孵育結合時，JPH203 的 $\text{Ki}$ 值為 $0.37\text{ mM}$ [Fact 4]。
> * 預孵育與共同孵育之間的協同作用導致 $\text{IC}_{50}$ 值降低三倍 [Fact 5]。
> * 預孵育與共同孵育之間的協同作用導致 $\text{Ki}$ 值降低 20% [Fact 5]。
> 
> ## 【跨文獻推論】
> * 跨論文推論（基於 [Fact 3]、[Fact 4] 及 [Fact 5]）：預孵育透過提升其功能效力 ($\text{IC}_{50}$) 以及結合親和力/抑制常數 ($\text{Ki}$)，增加了 JPH203 的抑制效力，這可由與單獨共同孵育相比，兩者數值均顯著下降來證明。
> 
> ## 【知識延伸與推測】
> * 模型推測（文獻依據不足）：觀察到預孵育能增強效力 ([Fact 2, 3])，但在藥物清除後可迅速逆轉 ([Fact 8])，這表明 JPH203 與 LAT1 並未形成共價鍵。相反地，這種行為是「慢結合」（slow-binding）動力學或轉運蛋白中時間依賴性構象變化（誘導契合，induc …（完整內容見 JSON）

---

### ⚠️ Q02 · single_paper

**問題**：What is the hybrid process used for the asymmetric synthesis of optically pure L-p-boronophenylalanine, and what are its key steps?

- detected_paper：`None`
- 選出論文：['bbb0683', 'CMDC-20-e202500059']
- gold_papers：['bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.767
- 延遲：3854.8s　問題標記：{'conflicts': 0, 'unsupported': 18}

**答案預覽**：

> 基於提供的學術事實，光學純 L-p-硼苯丙氨酸 (L-BPA) 不對稱合成的混合製程詳述如下。
> 
> ## 【論文直接依據】
> *   該混合製程將「對映選擇性烷基化」與隨後的「酶水解」相結合【bbb0683】。
> *   L-BPA 是一種腫瘤靶向氨基酸【bbb0683】。
> *   L-BPA 是硼中子捕獲治療 (BNCT) 最有效的硼載體之一【bbb0683】。
> *   第一步是保護反應【bbb0683】。
> *   保護反應使用 4-溴甲基苯硼酸 (1) 和 2,2-二甲基-1,3-丙二醇【bbb0683】。
> *   保護反應的溶劑為乾燥 THF【bbb0683】。
> *   保護反應在室溫下進行【bbb0683】。
> *   保護反應的攪拌時間為 10 minutes【bbb0683】。
> *   保護反應的產率為 79%【bbb0683】。
> *   保護反應的產物為環狀硼酸酯 (2)【bbb0683】。
> *   保護反應中使用的原料量為 5.99 g (27.9 mmol)【bbb0683】。
> *   使用的 2,2-二甲基-1,3-丙二醇量為 2.91 g (27.9 mmol)【bbb0683】。
> *   保護反應中使用的乾燥 THF 體積為 30 ml【bbb0683】。
> *   第二步是對映選擇性烷基化【bbb0683】。
> *   對映選擇性烷基化在 -78°C 下進行【bbb0683】。
> *   對映選擇性烷基化使用乾燥 THF 作為溶劑【bbb0683】。
> *   對映選擇性烷基化在氬氣 (Argon) 保護下進行【bbb0683】。
> *   使用 n-BuLi (1.68M in hexane) 對手性輔助劑進行鋰化【bbb0683】。
> *   使用的手性輔助劑為 (2R)-(-)-2,5-dihydro-2-isopropyl-3,6-dimethoxypyraz …（完整內容見 JSON）

---

### ✅ Q03 · figure_dependent

**問題**：According to the cryo-EM structure, how does JPH203 bind to and inhibit the LAT1-4F2hc complex?

- detected_paper：`None`
- 選出論文：['s41421-024-00697-6', '1-s2.0-S1347861320300633-main']
- gold_papers：['s41421-024-00697-6']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：1.0
- 延遲：1010.7s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
> * JPH203 結合於 LAT1 的傳統底物結合口袋中 【s41421-024-00697-6】。
> * JPH203 的分子結構包含一個具有苯丙氨酸（phenylalanine）骨架的親水性頭部 【s41421-024-00697-6】。
> * JPH203 的分子結構包含一個具有 5-amino-2-(3-aminophenyl) benzoxazole 骨架的疏水性尾部 【s41421-024-00697-6】。
> * 結合了 JPH203 的 LAT1-4F2hc 複合物呈現出面向外部（outward-facing）的結構 【s41421-024-00697-6】。
> * JPH203 結合複合物的整體解析度為 3.30 Å 【s41421-024-00697-6】。
> * 針對 JPH203 的局部精修（focused refinement）解析度為 3.25 Å 【s41421-024-00697-6】。
> * JPH203 親水性頭部的 $\alpha$-氨基與 TM1 和 TM6 去螺旋區域（uncoiled regions）的主鏈原子形成氫鍵網絡 【s41421-024-00697-6】。
> * JPH203 親水性頭部的 $\alpha$-羧基與 TM1 和 TM6 去螺旋區域的主鏈原子形成氫鍵網絡 【s41421-024-00697-6】。
> * 親水性頭部的二鹵酪胺酸（dihalo-tyrosine）核心上的氯原子與 Tyr259 形成鹵鍵 【s41421-024-00697-6】。
> * JPH203 的疏水性尾部嵌入在一個疏水性口袋中 【s41421-024-00697-6】。
> * 此疏水性口袋包括來自 TM1 的 Ile63、Ile64 和 Ile68 【s41421-024-00697-6】。
> * 此疏水性口袋包括來自 TM3 的 I …（完整內容見 JSON）

---

### ⚠️ Q04 · table_lookup

**問題**：What are the main degradation products and impurities of BPA identified by HPLC, and under which storage conditions do they form?

- detected_paper：`None`
- 選出論文：['1-s2.0-S0731708511003451-main', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water']
- gold_papers：['1-s2.0-S0731708511003451-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.633
- 延遲：2720.4s　問題標記：{'conflicts': 0, 'unsupported': 14}

**答案預覽**：

> ## 【論文直接依據】
> 
> **降解產物與條件**
> * 以 HPLC 鑑定之 BPA 降解產物包括酪胺酸 (tyrosine) [Chunk 1]。
> * 以 HPLC 鑑定之 BPA 降解產物包括苯丙胺酸 (phenylalanine) [Chunk 1]。
> * 酪胺酸的 HPLC 保留時間為 4.5 分鐘 [Chunk 1]。
> * 酪胺酸在 230 nm 處進行定量 [Chunk 1]。
> * 酪胺酸在鹼性條件下生成 [Chunk 1, Chunk 2]。
> * 酪胺酸在氧化條件下生成 [Chunk 1, Chunk 2]。
> * 使用 $6\text{ mM H}_2\text{O}_2$ 溶液時，酪胺酸的生成速度極快 [Chunk 1, Chunk 2]。
> * 當 BPA 溶解於 $100\text{ mM NaOH}$ 並在 $55^\circ\text{C}$ 下儲存 24 小時時，可觀察到酪胺酸 [Chunk 2]。
> * 苯丙胺酸的 HPLC 保留時間為 11.0 分鐘 [Chunk 1]。
> * 苯丙胺酸在 256 nm 處進行定量 [Chunk 1]。
> * 苯丙胺酸出現在冷凍乾燥的 BPA/甘露醇 (mannitol) 藥物樣本中 [Chunk 1, Chunk 2]。
> * 苯丙胺酸的降解速度緩慢且具有溫度依賴性 [Chunk 1, Chunk 2]。
> * 在 $40^\circ\text{C}$ 下儲存 6 個月後，會產生約 1% 的苯丙胺酸（相對於 BPA 質量）[Chunk 1, Chunk 2]。
> * 產生苯丙胺酸的樣本是透過將 $\text{BPA } (100\text{ mg/ml})$ 與 $\text{mannitol } (110\text{ mg/ml})$ 在 WFI 中混合製備而成 [Chunk 2]。
> * 這些樣本在冷凍乾燥前將 pH 值調 …（完整內容見 JSON）

---

### ⚠️ Q05 · single_paper

**問題**：What is the solvent-free green protocol for N-Boc protection of amines using picric acid as a catalyst, and what reaction conditions are used?

- detected_paper：`None`
- 選出論文：['1-s2.0-S2773223124000268-main']
- gold_papers：['1-s2.0-S2773223124000268-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.75
- 延遲：3447.9s　問題標記：{'conflicts': 0, 'unsupported': 8}

**答案預覽**：

> ## 【論文直接依據】
> * 苦味酸被用作胺類 N-Boc 保護的 Brønsted 酸催化劑 【1-s2.0-S2773223124000268-main】。
> * 催化劑用量可為 2 mol% 【1-s2.0-S2773223124000268-main】。
> * 催化劑用量可為 0.2 equivalents 【1-s2.0-S2773223124000268-main】。
> * 0.2 equivalents 的催化劑用量為最佳條件 【1-s2.0-S2773223124000268-main】。
> * 二叔丁基二碳酸酯 ((Boc)₂O / Boc anhydride) 被用作保護劑 【1-s2.0-S2773223124000268-main】。
> * 反應在無溶劑 (neat) 條件下進行 【1-s2.0-S2773223124000268-main】。
> * 反應在室溫下進行 【1-s2.0-S2773223124000268-main】。
> * 最佳反應時間為 60 minutes 【1-s2.0-S2773223124000268-main】。
> * 此方法適用於芳香族胺 【1-s2.0-S2773223124000268-main】。
> * 此方法適用於雜環芳香族胺 【1-s2.0-S2773223124000268-main】。
> * 此方法適用於開鏈脂肪族胺 【1-s2.0-S2773223124000268-main】。
> * 此方法適用於環狀脂肪族胺 【1-s2.0-S2773223124000268-main】。
> * 反應的最大產率可達 98% 【1-s2.0-S2773223124000268-main】。
> * 在使用 0.2 equivalents 的催化劑、無溶劑條件、室溫且反應時間為 60 minutes 時，一種特定的芳香族一級胺達到了 95% 的產率 【 …（完整內容見 JSON）

---

### ✅ Q06 · multi_chunk

**問題**：How does poly(vinyl alcohol) enhance the therapeutic effect of 4-L-boronophenylalanine in neutron capture therapy of thoracic tumors? Explain the proposed mechanism with supporting data.

- detected_paper：`None`
- 選出論文：['1-s2.0-S0378517325007926-main', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'CMDC-20-e202500059']
- gold_papers：['1-s2.0-S0378517325007926-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：75.0%　grounding：0.846
- 延遲：1967.5s　問題標記：{'conflicts': 0, 'unsupported': 4}

**答案預覽**：

> ## 【論文直接依據】
> * 聚乙烯醇 (PVA) 透過硼酸酯與多個 4-L-硼苯丙氨酸 (BPA) 分子形成稱為 PVA-BPA 的複合物 [1-s2.0-S0378517325007926-main]。
> * PVA-BPA 複合物保留了類苯丙氨酸結構，可針對 LAT1 (L-type amino acid transporter 1) [1-s2.0-S0378517325007926-main]。
> * PVA-BPA 透過「LAT1 介導的內吞作用」被癌細胞吸收 [1-s2.0-S0378517325007926-main]。
> * 一旦進入細胞，PVA-BPA 會定位於內體/溶酶體 (endo-/lysosomes) 中 [1-s2.0-S0378517325007926-main]。
> * PVA-BPA 的吸收機制增加了細胞對 BPA 的吸收量 [1-s2.0-S0378517325007926-main]。
> * PVA-BPA 的吸收機制延緩了 BPA 從細胞質中的外排 [1-s2.0-S0378517325007926-main]。
> * 延緩的外排延長了 BPA 在腫瘤內的滯留時間 [1-s2.0-S0378517325007926-main]。
> * PVA-BPA 在 1 hour 內透過主動 LAT1 定向，在腫瘤中實現高效累積與長期滯留 [1-s2.0-S0378517325007926-main]。
> * PVA-BPA 限制了其在正常器官中的累積 [1-s2.0-S0378517325007926-main]。
> * 使用山梨醇 (sorbitol) 作為增溶劑，開發了一種名為「PVA-sorbitol-BPA」的製劑，以優化其製備與組成 [1-s2.0-S0378517325007926-main]。
> * 與原型的 PVA-BPA 複合物相比，PVA-so …（完整內容見 JSON）

---

### ⚠️ Q07 · figure_dependent

**問題**：What is the water-stable boroxine structure reported, and what role do the dynamic covalent bonds play in its fluoride binding and hydrogel formation?

- detected_paper：`None`
- 選出論文：['41467_2024_Article_45464', 's41467-024-45464-z (1)']
- gold_papers：['41467_2024_Article_45464']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.609
- 延遲：2405.1s　問題標記：{'conflicts': 0, 'unsupported': 12}

**答案預覽**：

> ## 【論文直接依據】
> 
> **結構與穩定性**
> * 一種水穩定的硼氧環（boroxine）結構源自於 2-羥基苯硼酸 (HO-PBA) 【s41467-024-45464-z】。
> * HO-PBA 三聚體的 pH 穩定範圍為 $2.4 < \text{pH} < 9$ 【s41467-024-45464-z】。
> * $\text{CH}_3\text{-HO-PBA}$ 三聚體的 pH 穩定範圍為 $2.4 < \text{pH} < 10$ 【s41467-024-45464-z】。
> * HO-PBA 在室溫下會自發脫水形成二聚體 【s41467-024-45464-z】。
> * 計算得出的 HO-PBA 脫水自由能為 $-21.4\text{ kJ mol}^{-1}$ 【s41467-024-45464-z】。
> * 在室溫下接觸水後，HO-PBA 二聚體會迅速轉化為水穩定的硼氧環三聚體 【s41467-024-45464-z】。
> 
> **動態共價鍵 (DCBs)**
> * 所報導的結構包含 B–O 動態共價鍵 (DCBs) 【s41467-024-45464-z】。
> * 這些 DCB 在室溫的水相介質中可以快速交換 【s41467-024-45464-z】。
> * 傳統的硼氧環交換反應需要在甲苯溶液中加熱至 $60\text{ °C}$ 並持續 8 小時 【s41467-024-45464-z】。
> 
> **氟化物結合**
> * HO-PBA 三聚體能選擇性地識別水溶液中的氟離子 ($\text{F}^-$) 【s41467-024-45464-z】。
> * HO-PBA 三聚體對 $\text{F}^-$ 的結合親和力顯著高於苯硼酸 (PBA) 受體 【s41467-024-45464-z】。
> * HO-PBA 三聚體對氟化物的識別通過 $^1\text{H NMR}$ 光譜確認， …（完整內容見 JSON）

---

### ⚠️ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.765
- 延遲：2127.4s　問題標記：{'conflicts': 0, 'unsupported': 8}

**答案預覽**：

> ## 【論文直接依據】
> 
> * 目前尚無合成 L-BPA 的共識方法 【CMDC-20-e202500059】。
> * 在生產高純度 L-BPA 材料方面存在挑戰 【CMDC-20-e202500059】。
> * 在生產同位素富集 ($^{10}\text{B}$) 的 L-BPA 材料方面存在挑戰 【CMDC-20-e202500059】。
> * 評估 L-BPA 合成路徑的核心目標包括可擴展性 【CMDC-20-e202500059】。
> * 評估 L-BPA 合成路徑的核心目標包括成本效益 【CMDC-20-e202500059】。
> * 評估 L-BPA 合成路徑的核心目標包括安全性 【CMDC-20-e202500059】。
> * 在評估合成路徑時，必須考慮 $^{10}\text{B}$ 同位素富集的高成本 【CMDC-20-e202500059】。
> * 一種製備光學純 L-BPA 的混合製程使用 (2R)-(-)-2,5-dihydro-2-isopropyl-3,6-dimethoxypyrazine 作為手性輔助劑 【bbb0683】。
> * 此混合製程利用了受保護的 4-bromomethylphenylboronic acid 衍生物 【bbb0683】。
> * 此混合製程中的對映選擇性烷基化在 -78°C 下進行 【bbb0683】。
> * 此混合製程中的對映選擇性烷基化在 THF 中進行 【bbb0683】。
> * 混合製程中使用酸處理以產生 L-BPA 甲酯 【bbb0683】。
> * 混合製程中使用胰蛋白酶 (chymotrypsin) 在 pH 5.0 下進行酶水解以獲得 L-BPA 【bbb0683】。
> * 該混合製程產出的 L-BPA 具有 100% ee 【bbb0683】。
> * 與先前報導的方法相比，該混合製程涉及的反應步驟較少 【bbb0683】。
> * 該 …（完整內容見 JSON）

---

### ✅ Q09 · cross_paper

**問題**：Several papers target LAT1 for cancer therapy. What different therapeutic strategies targeting LAT1 are described across the literature, and how do they differ in mechanism?

- detected_paper：`None`
- 選出論文：['LAT1 ChemComm 2026', '1-s2.0-S0378517325007926-main', '1-s2.0-S1347861320300633-main', 's41421-024-00697-6', '1-s2.0-S1936523321002047-main']
- gold_papers：['1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 'LAT1 ChemComm 2026', 's41421-024-00697-6']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：1.0
- 延遲：2431.8s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
> 
> **硼傳遞策略（用於 BNCT）**
> * 針對 LAT1 的治療策略包括使用 p-硼苯丙氨酸 (BPA) 【1-s2.0-S0378517325007926-main】。
> * 針對 LAT1 的治療策略包括使用聚乙烯醇-BPA (PVA-BPA) 【1-s2.0-S0378517325007926-main】。
> * 針對 LAT1 的治療策略包括使用優化的 PVA-山梨醇-BPA 【1-s2.0-S0378517325007926-main】。
> * BPA 通過 LAT1 從細胞外空間傳遞至細胞質 【1-s2.0-S0378517325007926-main】。
> * 在其攝取過程中，BPA 導致細胞內氨基酸（如麩醯胺酸）被排出 【1-s2.0-S0378517325007926-main】。
> * 當細胞外濃度較低時，BPA 與細胞外酪胺酸進行交換 【1-s2.0-S0378517325007926-main】。
> * STEBORONINE® 是一種由山梨醇和 BPA 組成的製劑，在日本已獲准用於治療不可切除的局部晚期或復發性頭頸癌 【1-s2.0-S0378517325007926-main】。
> * STEBORONINE® 對於此前接受過放射治療的下咽/喉癌患者顯示出潛力 【1-s2.0-S0378517325007926-main】。
> * BPA 在腫瘤中的滯留時間較短 【1-s2.0-S0378517325007926-main】。
> * BPA 通常需要在熱中子照射前或照射期間持續輸注 30–60 分鐘 【1-s2.0-S0378517325007926-main】。
> * 對於軀幹腫瘤，由於照射時間較長（約 100 分鐘），維持穩定的 BPA 輸注在技術上較困難 【1-s2.0-S0378517325007926-main】。
> * PVA-B …（完整內容見 JSON）

---

### ⚠️ Q10 · contradiction_check

**問題**：Do the papers agree on the hydrolytic stability of boroxine structures? Describe any agreements or differences between them.

- detected_paper：`None`
- 選出論文：['Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation', '41467_2024_Article_45464', 's41467-024-45464-z (1)', 'a-method-for-the-deprotection-of-alkylpinacolyl-boronate-esters']
- gold_papers：['41467_2024_Article_45464', 'Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.538
- 延遲：3300.2s　問題標記：{'conflicts': 0, 'unsupported': 16}

**答案預覽**：

> ## 【論文直接依據】
> 
> **關於硼氧烷（boroxine）的一般穩定性與傳統策略：**
> * 一般共識認為，硼氧烷在水環境中具有水解不穩定性 【41467_2024_Article_45464, s41467-024-45464-z (1)】。
> * 硼氧烷在接觸水後會迅速水解為硼酸，導致功能喪失 【41467_2024_Article_45464, s41467-024-45464-z (1)】。
> * 提高穩定性的傳統策略包括引入給電子基團（electron-donating groups） 【41467_2024_Article_45464, s41467-024-45464-z (1)】。
> * 提高穩定性的傳統策略包括引入龐大基團（bulky groups） 【41467_2024_Article_45464, s41467-024-45464-z (1)】。
> * 提高穩定性的傳統策略包括與 N-供體配體形成加合物 【41467_2024_Article_45464, s41467-024-45464-z (1)】。
> * 提高穩定性的傳統策略包括將三個硼酸單元整合至柔性大環中以實現熵穩定化（entropic stabilization） 【41467_2024_Article_45464, s41467-024-45464-z (1)】。
> * 這些傳統方法未能從根本上解決水解不穩定性的問題 【41467_2024_Article_45464, s41467-024-45464-z (1)】。
> 
> **關於 2-羥基苯硼酸（HO-PBA）衍生物：**
> * 源自 HO-PBA 的硼氧烷結構表現出水穩定性 【41467_2024_Article_45464, s41467-024-45464-z (1)】。
> * HO-PBA 在室溫下會自發脫水形成二聚體 【41467_2024 …（完整內容見 JSON）

---

### ✅ Q11 · out_of_scope

**問題**：What are the phase III clinical trial overall survival results for BNCT in glioblastoma patients?

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', '1-s2.0-S0378517325007926-main', 'cas0106-0279']
- gold_papers：[]
- 論文選擇命中率：N/A　檢索覆蓋率：N/A　grounding：—
- 延遲：340.6s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ⚠️ **資料來源說明**：本地學術文獻資料庫中未找到與此問題直接相關的內容。以下回答來自模型自身知識，非論文原文，請謹慎參考並自行查證。
> 
> 針對您提出的關於「硼中子捕獲治療（Boron Neutron Capture Therapy, BNCT）在膠質母細胞瘤（Glioblastoma, GBM）患者中的第三期臨床試驗總生存期（Overall Survival, OS）結果」之問題，以下是基於目前學術知識的詳細分析。
> 
> ### 核心結論：關於第三期臨床試驗的現狀
> 
> 首先必須明確指出：**截至目前的公開學術紀錄中，BNCT 在治療膠質母細胞瘤方面，尚未有大規模、多中心且完成的「第三期隨機對照臨床試驗（Phase III Randomized Controlled Trial）」所公布的標準化總生存期（OS）數據。**
> 
> 大多數關於 BNCT 治療 GBM 的研究仍處於第一期（安全性評估）或第二期（初步療效評估）階段。因此，目前無法提供一個如同「標準治療方案（Stupp Protocol）」那樣具有高度共識的第三期 OS 數值。
> 
> ---
> 
> ### 學術分析與現有數據討論
> 
> 雖然缺乏第三期試驗結果，但我們可以從第二期及初步臨床研究中觀察到 BNCT 的潛在療效趨勢：
> 
> #### 1. 總生存期（OS）的觀察趨勢
> 在針對**復發性膠質母細胞瘤（Recurrent GBM）**的第二期試驗或病例系列研究中，BNCT 顯示出一定的生存期延長效果。部分研究指出，接受 BNCT 的患者其 OS 可能達到 12 至 20 個月不等，而歷史對照組（僅接受姑息治療或化療）的生存期通常顯著較短。
> 
> #### 2. 新診斷 GBM 與復發性 GBM 的差異
> *   **新診斷 GBM：** 目前的研究方向傾向於將 BNCT 作為手術後的輔助治療，旨在提高局部控制率。由於缺乏與標準放化療（TMZ  …（完整內容見 JSON）

---

### ❌ Q12 · false_premise

**問題**：Since BPA is administered orally in BNCT, what oral bioavailability values are reported in these papers?

- detected_paper：`None`
- 選出論文：['cas0106-0279', 'Chem Biol Drug Des - 2013 - Achilli - Folic Acid‐Conjugated 4‐Amino‐Phenylboronate  a Boron‐Containing Compound Designed']
- gold_papers：[]
- 論文選擇命中率：N/A　檢索覆蓋率：N/A　grounding：0.0
- 延遲：1185.8s　問題標記：{'conflicts': 0, 'unsupported': 2}

**答案預覽**：

> ## 【論文直接依據】
> 
> * 提供的文本中未報告 BPA 的口服生物利用度數值。
> 
> ## 【跨文獻推論】
> 
> * 文獻依據不足：雖然提供的論文討論了負責 BPA 吸收的轉運蛋白之鑑定與動力學 ([Fact 13], [Fact 14])，但並未提供關於口服給藥後進入全身循環的藥物百分比（口服生物利用度）的定量數據。
> 
> ## 【知識延伸與推測】
> 
> * 模型推測（文獻依據不足）：針對 *Xenopus* 卵母細胞中芳香族氨基酸轉運蛋白的研究 ([Fact 13], [Fact 14]) 表明，BPA 的吸收可能是由特定的蛋白質驅動轉運機制介導，而非簡單擴散。在臨床背景下，這些轉運蛋白在人類胃腸道中的效率將是其口服生物利用度的主要決定因素，但提供的文本並未對此效應進行定量分析。
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
> ⚠️  **以下 2 個陳述未找到明確論文依據，請謹慎參考：**
> 
> - * No oral bioavailability values for BPA are reported in the provided text.（信心度：32.9%，最近似來源：cas0106-0279-d7164215）
> - * Insufficient literature basis: While the provided papers discuss the identification and kinetics of transporters responsible for B …（完整內容見 JSON）

---
