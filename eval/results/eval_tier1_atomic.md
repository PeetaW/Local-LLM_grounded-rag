# Eval Report — `tier1_atomic`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-06-19 10:37
- 題數：12

## 彙總

| 指標 | 值 |
|------|-----|
| 平均 grounding 分數 | 0.498 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 97.5% |
| 平均總延遲 | 3809.0s |
| 平均 planning 延遲 | 7.2s |
| 平均 retrieval 延遲 | 333.9s |
| 平均 grounding 延遲 | 1911.1s |
| └ 其中 NLI | 32.9s |
| └ 其中 gemma4 | 1860.1s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ⚠️ | Q01 | single_paper | 100.0% | 100.0% | 0.6 | 2525.2s | C0/U7 |
| ⚠️ | Q02 | single_paper | 100.0% | 100.0% | 0.529 | 6564.1s | C0/U29 |
| ⚠️ | Q03 | figure_dependent | 100.0% | 100.0% | 0.577 | 4733.1s | C0/U14 |
| ⚠️ | Q04 | table_lookup | 100.0% | 100.0% | 0.526 | 3027.3s | C0/U10 |
| ⚠️ | Q05 | single_paper | 100.0% | 100.0% | 0.5 | 2595.8s | C0/U11 |
| ❌ | Q06 | multi_chunk | 100.0% | 75.0% | 0.278 | 3708.8s | C0/U17 |
| ⚠️ | Q07 | figure_dependent | 100.0% | 100.0% | 0.542 | 3769.4s | C0/U16 |
| ⚠️ | Q08 | cross_paper | 100.0% | 100.0% | 0.611 | 6146.1s | C0/U19 |
| ⚠️ | Q09 | cross_paper | 100.0% | 100.0% | 0.773 | 6588.7s | C0/U20 |
| ⚠️ | Q10 | contradiction_check | 100.0% | 100.0% | 0.542 | 4444.2s | C0/U16 |
| ✅ | Q11 | out_of_scope | N/A | N/A | — | 303.1s | C0/U0 |
| ❌ | Q12 | false_premise | N/A | N/A | 0.0 | 1302.6s | C0/U2 |

## 逐題細節

### ⚠️ Q01 · single_paper

**問題**：In the study of JPH203's preincubation inhibitory effect on LAT1, how does preincubation change JPH203's inhibitory potency? Give the reported values.

- detected_paper：`None`
- 選出論文：['1-s2.0-S1347861320300633-main', '1-s2.0-S0021979710012403-main', '1-s2.0-S1936523321002047-main', 's41421-024-00697-6']
- gold_papers：['1-s2.0-S1347861320300633-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.6
- 延遲：2525.2s　問題標記：{'conflicts': 0, 'unsupported': 7}

**答案預覽**：

> ## 【論文直接依據】
> 
> * 在 HT-29 細胞中，JPH203 對 LAT1 的預孵育（preincubation）表現出濃度依賴性的抑制效果 [1-s2.0-S1347861320300633-main]。
> * 在 HT-29 細胞中，JPH203 對 LAT1 的預孵育表現出時間依賴性的抑制效果 [1-s2.0-S1347861320300633-main]。
> * 僅就預孵育而言，計算出的 $\text{IC}_{50}$ 值為 $193 \pm 50\text{ nM}$（基於 120 分鐘的暴露時間）[1-s2.0-S1347861320300633-main]。
> * 當預孵育與共同孵育（co-incubation）結合時，能顯著增強 JPH203 的抑制效力，產生協同效應 [1-s2.0-S1347861320300633-main]。
> * 僅就共同孵育而言，$\text{IC}_{50}$ 值為 $99.2 \pm 11.0\text{ nM}$ [1-s2.0-S1347861320300633-main]。
> * 預孵育與共同孵育結合後的 $\text{IC}_{50}$ 值為 $34.2 \pm 3.6\text{ nM}$ [1-s2.0-S1347861320300633-main]。
> * 僅就共同孵育而言，$\text{Ki}$ 值為 $0.46\text{ mM}$ [1-s2.0-S1347861320300633-main]。
> * 預孵育與共同孵育結合後的 $\text{Ki}$ 值為 $0.37\text{ mM}$ [1-s2.0-S1347861320300633-main]。
> * 預孵育與共同孵育之間的協同作用導致 $\text{IC}_{50}$ 值降低三倍 [1-s2.0-S1347861320300633-main]。
> * 預孵 …（完整內容見 JSON）

---

### ⚠️ Q02 · single_paper

**問題**：What is the hybrid process used for the asymmetric synthesis of optically pure L-p-boronophenylalanine, and what are its key steps?

- detected_paper：`None`
- 選出論文：['bbb0683', 'CMDC-20-e202500059']
- gold_papers：['bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.529
- 延遲：6564.1s　問題標記：{'conflicts': 0, 'unsupported': 29}

**答案預覽**：

> ## 【論文直接依據】
> 
> **一般流程與目標**
> * 研究目標是探索一種更高效的對光學純 L-p-硼苯丙氨酸 (L-BPA) 的合成方法 【bbb0683】。
> * L-BPA 作為一種腫瘤靶向氨基酸 【bbb0683】。
> * L-BPA 是硼中子捕獲治療 (BNCT) 中最有效的硼載體之一 【bbb0683】。
> * 合成光學純 L-BPA 的混合流程結合了「不對稱烷基化」以及隨後的「酶水解」 【bbb0683】。
> * 與使用 $\alpha$-苯甲酰胺肉桂酸衍生物作為底物的方法相比，該混合流程涉及的反應步驟較少 【bbb0683】。
> * 與需要額外分離和純化的方法相比，該混合流程在操作和後處理方面更簡單 【bbb0683】。
> 
> **步驟 1：保護反應**
> * 起始原料為 4-溴甲基苯硼酸 (1) 【bbb0683】。
> * 使用 2,2-二甲基-1,3-丙二醇將物質 (1) 的二羥基硼酸基團保護為環狀硼酸酯 (2) 【bbb0683】。
> * 該保護反應在乾燥的 THF 中進行 【bbb0683】。
> * 反應在室溫下攪拌 10 分鐘 【bbb0683】。
> * 保護反應的產率為 79% 【bbb0683】。
> 
> **步驟 2：對映選擇性烷基化**
> * 使用 n-BuLi（正己烷中 1.68M）對 $(2R)-(-)$-2,5-二氫-2-異丙基-3,6-二甲氧基吡嗪 (3) 進行鋰化 【bbb0683】。
> * 鋰化的物質 (3) 與受保護的 4-溴甲基苯硼酸衍生物 (2) 反應 【bbb0683】。
> * 反應在氬氣環境下進行 【bbb0683】。
> * 使用乾燥的 THF 作為該反應的溶劑 【bbb0683】。
> * 反應溫度維持在 -78°C 【bbb0683】。
> * n-BuLi 在 -78°C 下分 30 分鐘加入 【bbb0683】。
> * 加入 n-BuLi 後，混合物靜置 1  …（完整內容見 JSON）

---

### ⚠️ Q03 · figure_dependent

**問題**：According to the cryo-EM structure, how does JPH203 bind to and inhibit the LAT1-4F2hc complex?

- detected_paper：`None`
- 選出論文：['s41421-024-00697-6', '1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 's41467-024-45464-z (1)', 'LAT1 ChemComm 2026']
- gold_papers：['s41421-024-00697-6']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.577
- 延遲：4733.1s　問題標記：{'conflicts': 0, 'unsupported': 14}

**答案預覽**：

> ## 【論文直接依據】
> * JPH203 結合於 LAT1 的傳統底物結合口袋中 【s41421-024-00697-6】。
> * JPH203 的分子結構由具有苯丙氨酸（phenylalanine）骨架的親水性頭部組成 【s41421-024-00697-6】。
> * JPH203 的分子結構由具有 5-amino-2-(3-aminophenyl) benzoxazole 骨架的疏水性尾部組成 【s41421-024-00697-6】。
> * JPH203 親水性頭部的 $\alpha$-氨基與 LAT1 的 TM1 和 TM6 未折疊區域的主鏈原子形成氫鍵網絡 【s41421-024-00697-6】。
> * JPH203 親水性頭部的 $\alpha$-羧基與 LAT1 的 TM1 和 TM6 未折疊區域的主鏈原子形成氫鍵網絡 【s41421-024-00697-6】。
> * JPH203 二鹵酪胺酸（dihalo-tyrosine）核心上的一個氯原子與 Tyr259 形成鹵鍵 【s41421-024-00697-6】。
> * JPH203 的疏水性尾部嵌入由 TM1 (Ile63, Ile64, Ile68)、TM3 (Ile140, Ile147, Ile148)、TM6 (Phe252, Gly255) 和 TM10 (Ile397, Phe400, Phe403) 組成的疏水性口袋中 【s41421-024-00697-6】。
> * JPH203 尾部的氨基與 TM10 上 Phe400 的主鏈形成氫鍵 【s41421-024-00697-6】。
> * JPH203 尾部的苯環與 Phe252 的側鏈發生 T 型 $\pi$-$\pi$ 交互作用 【s41421-024-00697-6】。
> * JPH203 與 LAT1-4F2hc 複合物的冷凍電子顯微鏡（cryo-EM） …（完整內容見 JSON）

---

### ⚠️ Q04 · table_lookup

**問題**：What are the main degradation products and impurities of BPA identified by HPLC, and under which storage conditions do they form?

- detected_paper：`None`
- 選出論文：['1-s2.0-S0731708511003451-main', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water']
- gold_papers：['1-s2.0-S0731708511003451-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.526
- 延遲：3027.3s　問題標記：{'conflicts': 0, 'unsupported': 10}

**答案預覽**：

> ## 【論文直接依據】
> 
> **分解產物與條件**
> * BPA 在鹼性條件下分解為酪胺酸 (Tyrosine) [Chunk 1, Chunk 2]。
> * BPA 在氧化條件下分解為酪胺酸 (Tyrosine) [Chunk 1, Chunk 2]。
> * 在鹼性水溶液中，將 BPA 溶解於 $100\text{ mM NaOH}$ 並在 $55^\circ\text{C}$ 下培養 24 小時會導致其分解為酪胺酸 (Tyrosine) [Chunk 2]。
> * 在水相中使用 $6\text{ mM H}_2\text{O}_2$ 的氧化條件下，BPA 會迅速分解為酪胺酸 (Tyrosine) [Chunk 1, Chunk 2]。
> * 凍乾的 BPA 與甘露醇 (mannitol) 結合後，在 $40^\circ\text{C}$ 下 6 個月內緩慢分解 [Chunk 1, Chunk 2]。
> * 此 BPA-甘露醇凍乾產品的分解會產生苯丙胺酸 (Phenylalanine) [Chunk 1, Chunk 2]。
> * 在 BPA-甘露醇混合物中產生的苯丙胺酸 (Phenylalanine) 數量約為 BPA 質量的 1% [Chunk 1, Chunk 2]。
> * 在鹼性水溶液條件下，會發生原脫硼反應 (protodeborylation)，最常見的結果是形成母體芳香族<0xE7><0x83><0xB4> [Chunk 2]。
> 
> **HPLC 鑑定特徵**
> * 酪胺酸 (Tyrosine) 在 $230\text{ nm}$ 下定量，保留時間為 $4.5\text{ minutes}$ [Chunk 1]。
> * 苯丙胺酸 (Phenylalanine) 在 $256\text{ nm}$ 下定量，保留時間為 $11.0\text{ minutes}$ [Chunk 1]。
>  …（完整內容見 JSON）

---

### ⚠️ Q05 · single_paper

**問題**：What is the solvent-free green protocol for N-Boc protection of amines using picric acid as a catalyst, and what reaction conditions are used?

- detected_paper：`None`
- 選出論文：['1-s2.0-S2773223124000268-main']
- gold_papers：['1-s2.0-S2773223124000268-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.5
- 延遲：2595.8s　問題標記：{'conflicts': 0, 'unsupported': 11}

**答案預覽**：

> ## 【論文直接依據】
> * 苦味酸被用作胺類 N-Boc 保護的 Brønsted 酸催化劑 [1-s2.0-S2773223124000268-main]。
> * 催化劑用量可為 2 mol% 或 0.2 當量 [1-s2.0-S2773223124000268-main]。
> * 二叔丁基二碳酸酯 ((Boc)₂O / Boc anhydride) 被用作保護劑 [1-s2.0-S2773223124000268-main]。
> * 反應在室溫且無溶劑的條件下進行 [1-s2.0-S2773223124000268-main]。
> * 優化後的反應時間為 60 分鐘 [1-s2.0-S2773223124000268-main]。
> * 此方法的最高產率可達 98% [1-s2.0-S2773223124000268-main]。
> * 在一項特定的篩選實驗中（表 1，Entry 13），在室溫下使用 0.2 當量的催化劑反應 60 分鐘，產率為 95% [1-s2.0-S2773223124000268-main]。
> * 適用底物包括芳香胺、雜環芳香胺、開鏈脂肪胺和環狀脂肪胺 [1-s2.0-S2773223124000268-main]。
> * 反應機制涉及苦味酸活化 Boc anhydride 的羰基氧，隨後由胺的氮原子攻擊羰基碳 [1-s2.0-S2773223124000268-main]。
> * 反應以質子取代以及消除 $\text{CO}_2$ 和 t-BuOH 作為副產物而結束，從而形成 N-Boc 保護化合物 [1-s2.0-S2773223124000268-main]。
> * 論文未討論濃度或攪拌速度 [1-s2.0-S2773223124000268-main]。
> 
> ## 【跨文獻推論】
> 文獻依據不足；僅提供了一篇論文，無法進行跨論文綜合分析。
> 
> ## 【知識延伸 …（完整內容見 JSON）

---

### ❌ Q06 · multi_chunk

**問題**：How does poly(vinyl alcohol) enhance the therapeutic effect of 4-L-boronophenylalanine in neutron capture therapy of thoracic tumors? Explain the proposed mechanism with supporting data.

- detected_paper：`None`
- 選出論文：['1-s2.0-S0378517325007926-main', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'CMDC-20-e202500059']
- gold_papers：['1-s2.0-S0378517325007926-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：75.0%　grounding：0.278
- 延遲：3708.8s　問題標記：{'conflicts': 0, 'unsupported': 17}

**答案預覽**：

> ## 【論文直接依據】
> * 聚乙烯醇 (PVA) 通過硼酸酯與多個 BPA 分子形成複合物 [1-s2.0-S0378517325007926-main]。
> * PVA-BPA 複合物的形成已通過 NMR 確認 [1-s2.0-S0378517325007926-main]。
> * PVA-BPA 複合物保留了類苯丙氨酸結構 [1-s2.0-S0378517325007926-main]。
> * 此類苯丙氨酸結構對於與大型中性氨基酸轉運蛋白 1 (LAT1) 的相互作用至關重要 [1-s2.0-S0378517325007926-main]。
> * PVA-BPA 與 LAT1 發生相互作用 [1-s2.0-S0378517325007926-main]。
> * PVA-BPA 通過 LAT1 介導的胞吞作用被攝入癌細胞中 [1-s2.0-S0378517325007926-main]。
> * 一旦進入細胞，PVA-BPA 會定位於內體和溶酶體中 [1-s2.0-S0378517325007926-main]。
> * 這種攝取機制增加了細胞攝取的 BPA 總量 [1-s2.0-S0378517325007926-main]。
> * 該機制延緩了 BPA 從細胞質中的外排 [1-s2.0-S0378517325007926-main]。
> * 這些因素共同延長了 BPA 在腫瘤內的滯留時間 [1-s2.0-S0378517325007926-main]。
> * PVA-BPA 通過對 LAT1 的主動靶向，在 1 hour 內實現了在腫瘤中的高效累積與滯留 [1-s2.0-S0378517325007926-main]。
> * PVA-BPA 在正常器官中的累積有限 [1-s2.0-S0378517325007926-main]。
> * 胸部腫瘤需要約 100 minutes 的熱中子照射才能提供 …（完整內容見 JSON）

---

### ⚠️ Q07 · figure_dependent

**問題**：What is the water-stable boroxine structure reported, and what role do the dynamic covalent bonds play in its fluoride binding and hydrogel formation?

- detected_paper：`None`
- 選出論文：['41467_2024_Article_45464', 's41467-024-45464-z (1)']
- gold_papers：['41467_2024_Article_45464']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.542
- 延遲：3769.4s　問題標記：{'conflicts': 0, 'unsupported': 16}

**答案預覽**：

> ## 【論文直接依據】
> * 所報導的水穩定硼氧環結構源自 2-羥基苯硼酸 (HO-PBA) 【s41467-024-45464-z】。
> * 該結構的開發旨在解決硼氧環在水環境中易水解且不穩定的挑戰 【s41467-024-45464-z】。
> * 在室溫下，HO-PBA 會自發脫水形成二聚體 【s41467-024-45464-z】。
> * 這些 HO-PBA 二聚體具有動態共價鍵 (DCBs) 【s41467-024-45464-z】。
> * 這些 HO-PBA 二聚體表現出聚集誘導發光增強 (AIEE) 活性 【s41467-024-45464-z】。
> * 在室溫下接觸水後，HO-PBA 二聚體會迅速轉化為被描述為三聚體-$\text{H}_2\text{O}$ 複合物的硼氧環結構 【s41467-024-45464-z】。
> * HO-PBA 三聚體在 pH 範圍 2.4 至 9 之間保持穩定 【s41467-024-45464-z】。
> * $\text{CH}_3\text{-HO-PBA}$ 三聚體在 pH 範圍 2.4 至 10 之間保持穩定 【s41467-024-45464-z】。
> * 此硼氧環結構中的 $\text{B–O}$ 鍵在室溫水溶液中可進行快速交換 【s41467-024-45464-z】。
> * 傳統的硼氧環結構需要將甲苯溶液加熱至 $60\text{ °C}$ 並維持 8 小時才能實現交換反應 【s41467-024-45464-z】。
> * HO-PBA 三聚體在水溶液中能選擇性地識別氟離子 ($\text{F}^-$) 【s41467-024-45464-z】。
> * HO-PBA 三聚體對 $\text{F}^-$ 的結合親和力顯著強於苯硼酸 (PBA) 【s41467-024-45464-z】。
> * $^1\text{H NMR}$ 分析顯示 HO …（完整內容見 JSON）

---

### ⚠️ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.611
- 延遲：6146.1s　問題標記：{'conflicts': 0, 'unsupported': 19}

**答案預覽**：

> ## 【論文直接依據】
> 
> **同位素富集與成本因素**
> * $^{10}\text{B}$ 透過化學交換蒸餾法從 $^{11}\text{B}$ 中分離 【CMDC-20-e202500059】。
> * 分離出的 $^{10}\text{B}$ 被轉化為硼酸酯或水解為 $^{10}\text{B}$ 硼酸 【CMDC-20-e202500059】。
> * $^{10}\text{B}$ 的價格比普通硼酸高出 1000 倍以上 【CMDC-20-e202500059】。
> * $^{10}\text{B}$ 的成本主導了 L-BPA 合成的經濟效率 【CMDC-20-e202500059】。
> * 依賴 $\text{B}_2\text{pin}_2$ 的合成路徑被認為並不理想，因為與 $^{10}\text{B}$ 原子核浪費相關的成本較高 【CMDC-20-e202500059】。
> 
> **Pd 催化與雙鹵代路徑**
> * 由於步驟過多，雙鹵代苯合成路徑在很大程度上已被直接氨基酸官能化所取代 【CMDC-20-e202500059】。
> * 在 Pd 催化硼化反應中，$\text{B}_2\text{pin}_2$ 結合 DMSO 和 KOAc 的效果優於 HBpin 結合 1,4-dioxane 和 $\text{NEt}_3$ 【CMDC-20-e202500059】。
> * Pd 催化路徑中交叉耦合與脫保護的總產率未超過 48% 【CMDC-20-e202500059】。
> * 森田製藥公司（Morita Pharmaceutical Company）的專利方法步驟較少且產率極高 【CMDC-20-e202500059】。
> * 森田製藥公司方法的倒數第二步需要 Pd 催化劑 【CMDC-20-e202500059】。
> 
> **非 Pd 催化路徑**
> * n-BuLi 方法在 THF 中 …（完整內容見 JSON）

---

### ⚠️ Q09 · cross_paper

**問題**：Several papers target LAT1 for cancer therapy. What different therapeutic strategies targeting LAT1 are described across the literature, and how do they differ in mechanism?

- detected_paper：`None`
- 選出論文：['LAT1 ChemComm 2026', '1-s2.0-S0378517325007926-main', '1-s2.0-S1347861320300633-main', 's41421-024-00697-6', '1-s2.0-S1936523321002047-main']
- gold_papers：['1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 'LAT1 ChemComm 2026', 's41421-024-00697-6']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.773
- 延遲：6588.7s　問題標記：{'conflicts': 0, 'unsupported': 20}

**答案預覽**：

> ## 【論文直接依據】
> 
> **通用策略**
> * 針對 LAT1 的治療策略包括硼中子捕獲治療 (BNCT) 【1-s2.0-S0378517325007926-main, 1-s2.0-S1347861320300633-main, 1-s2.0-S1936523321002047-main, LAT1 ChemComm 2026, s41421-024-00697-6】。
> * 針對 LAT1 的治療策略包括藥理學抑制 【1-s2.0-S0378517325007926-main, 1-s2.0-S1347861320300633-main, 1-s2.0-S1936523321002047-main, LAT1 ChemComm 2026, s41421-024-00697-6】。
> * 針對 LAT1 的治療策略包括基因操縱 【1-s2.0-S0378517325007926-main, 1-s2.0-S1347861320300633-main, 1-s2.0-S1936523321002047-main, LAT1 ChemComm 2026, s41421-024-00697-6】。
> * 針對 LAT1 的治療策略包括使用自組裝肽 【1-s2.0-S0378517325007926-main, 1-s2.0-S1347861320300633-main, 1-s2.0-S1936523321002047-main, LAT1 ChemComm 2026, s41421-024-00697-6】。
> 
> **硼中子捕獲治療 (BNCT)**
> * 標準的 BPA 策略（例如 STEBORONINE®）利用 LAT1 將細胞外的 BPA 運輸至細胞質中 【1-s2.0-S0378517325007926-main】。
> * 標準的 BPA 運輸伴隨著細胞內氨基酸（如谷氨酰胺）的 …（完整內容見 JSON）

---

### ⚠️ Q10 · contradiction_check

**問題**：Do the papers agree on the hydrolytic stability of boroxine structures? Describe any agreements or differences between them.

- detected_paper：`None`
- 選出論文：['Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation', '41467_2024_Article_45464', 's41467-024-45464-z (1)', 'a-method-for-the-deprotection-of-alkylpinacolyl-boronate-esters']
- gold_papers：['41467_2024_Article_45464', 'Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.542
- 延遲：4444.2s　問題標記：{'conflicts': 0, 'unsupported': 16}

**答案預覽**：

> ## 【論文直接依據】
> 
> **來自【s41467-024-45464-z】：**
> * 傳統認知認為硼氧環（boroxines）在水環境中具有水解不穩定性 [s41467-024-45464-z]。
> * 硼氧環在接觸水後通常會迅速水解為硼酸，導致功能喪失 [s41467-024-45464-z]。
> * 傳統的穩定性提升策略包括透過引入電子捐贈基團來降低路易斯酸性（Lewis acidic）硼位點的親電性 [s41467-024-45464-z]。
> * 傳統的穩定性提升策略包括引入大體積基團 [s41467-024-45464-z]。
> * 傳統的穩定性提升策略包括與 N-供體配體形成加合物 [s41467-024-45464-z]。
> * 傳統的穩定性提升策略包括將三個硼酸單元整合到柔性大環中，以採用熵穩定（entropy stabilization）[s41467-024-45464-z]。
> * 涉及電子捐贈基團、大體積基團或 N-供體配體加合物的策略僅能使硼氧環更強韌，但未能從根本上解決水解不穩定性 [s41467-024-45464-z]。
> * 2-羥基苯硼酸（HO-PBA）可以形成在水中穩定的硼氧環結構 [s41467-024-45464-z]。
> * HO-PBA 在室溫下會自發脫水形成二聚體 [s41467-024-45464-z]。
> * 接觸水後，HO-PBA 二聚體會迅速轉化為 HO-PBA 三聚體–$\text{H}_2\text{O}$ 複合物 [s41467-024-45464-z]。
> * 這些 HO-PBA 三聚體在室溫下可保持穩定長達 7 天 [s41467-024-45464-z]。
> * HO-PBA 三聚體的穩定性已透過 $^1\text{H NMR}$、ESI-Q-TOF MS 和 UV-Raman 測量得到證實 [s41467-024-45464- …（完整內容見 JSON）

---

### ✅ Q11 · out_of_scope

**問題**：What are the phase III clinical trial overall survival results for BNCT in glioblastoma patients?

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', '1-s2.0-S0378517325007926-main', 'cas0106-0279']
- gold_papers：[]
- 論文選擇命中率：N/A　檢索覆蓋率：N/A　grounding：—
- 延遲：303.1s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ⚠️ **資料來源說明**：本地學術文獻資料庫中未找到與此問題直接相關的內容。以下回答來自模型自身知識，非論文原文，請謹慎參考並自行查證。
> 
> 針對您提出的關於「硼中子捕獲治療（Boron Neutron Capture Therapy, BNCT）在膠質母細胞瘤（Glioblastoma, GBM）患者中的第三期臨床試驗整體生存期（Overall Survival, OS）結果」之問題，以下是基於目前學術知識的詳細分析。
> 
> ### 核心結論：關於第三期臨床試驗的現狀
> 
> 首先必須明確指出：**目前在國際主流醫學文獻中，尚未有針對膠質母細胞瘤（GBM）且具有大規模、隨機對照（Randomized Controlled Trial, RCT）性質的「標準第三期臨床試驗」所公布的統一 OS 數據。**
> 
> 大多數關於 BNCT 治療 GBM 的研究目前仍處於第一期（安全性評估）或第二期（初步療效評估）階段。BNCT 的特殊性在於其對中子源（Neutron Source）的高度依賴，過去主要依賴核反應爐，而近年才逐漸轉向加速器（Accelerator-based BNCT, AB-BNCT），這導致臨床試驗的規模較難迅速擴大至第三期所需的樣本數。
> 
> ### 現有研究之療效分析（以第二期及回顧性研究為主）
> 
> 雖然缺乏標準的第三期數據，但根據日本與芬蘭等前沿研究中心的第二期或小規模臨床數據，BNCT 在 GBM 患者中展現出一定的潛力：
> 
> 1.  **生存期的改善趨勢：**
>     在部分針對復發性 GBM（Recurrent GBM）的小規模研究中，BNCT 顯示出能延長患者生存期的傾向。部分報告指出，接受 BNCT 的患者其中位數整體生存期（Median OS）較歷史對照組有所提升。
> 2.  **亞群差異：**
>     研究發現，OS 的結果高度依賴於腫瘤的生物學特性（如 MGMT 啟 …（完整內容見 JSON）

---

### ❌ Q12 · false_premise

**問題**：Since BPA is administered orally in BNCT, what oral bioavailability values are reported in these papers?

- detected_paper：`None`
- 選出論文：['cas0106-0279', 'Chem Biol Drug Des - 2013 - Achilli - Folic Acid‐Conjugated 4‐Amino‐Phenylboronate  a Boron‐Containing Compound Designed']
- gold_papers：[]
- 論文選擇命中率：N/A　檢索覆蓋率：N/A　grounding：0.0
- 延遲：1302.6s　問題標記：{'conflicts': 0, 'unsupported': 2}

**答案預覽**：

> ## 【論文直接依據】
> * 提供的文本中未報告 BPA (p-boronophenylalanine) 的口服生物利用度數值。【Chem Biol Drug Des - 2013 - Achilli; cas0106-0279】
> 
> ## 【知識延伸與推測】
> * 模型推測（文獻依據不足）：雖然原問題假設 BPA 為口服給藥，但提供的文本並未具體提及 BPA 的給藥途徑。文中僅提到 BNCT 載體通常需要靜脈注射【Chem Biol Drug Des - 2013 - Achilli】。這些特定摘錄中缺乏生物利用度數據，表明所引用的論文側重於細胞轉運機制（例如，使用 *Xenopus oocytes* 來鑑定轉運體【cas0106-0279】）和生物相容性【Chem Biol Drug Des - 2013 - Achilli】，而非藥代動力學分析或口服劑量研究。
> 
> ---
> 📋 **答案品質報告**
> 
> 📊 **分段論文依據率：**
> 
>   ❌ 【論文直接依據】：0.0%（0/1 句）
>   ❌ 【知識延伸推測】：0.0%（0/1 句）  ← 知識延伸推測，低分為預期範圍
> 
> ❌ **直引依據率**：0.0%　低（建議縮小問題範圍）
> 
> ⚠️  **以下 2 個陳述未找到明確論文依據，請謹慎參考：**
> 
> - * No oral bioavailability values for BPA (p-boronophenylalanine) are reported in the provided text. [Chem Biol Drug Des - 2013 - Achilli; cas0106-0279]（信心度：8.4%，最近似來源：cas0106-0279-b2eec8a9）
> - * Model speculation (insufficient literature basis): …（完整內容見 JSON）

---
