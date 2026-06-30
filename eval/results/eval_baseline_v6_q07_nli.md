# Eval Report — `baseline_v6_q07_nli`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-06-30 22:30
- 題數：12

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.833 |
| 平均 grounding 分數 | 0.702 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 97.5% |
| 平均總延遲 | 1011.9s |
| 平均 planning 延遲 | 7.4s |
| 平均 retrieval 延遲 | 428.6s |
| 平均 grounding 延遲 | 42.3s |
| └ 其中 NLI | 26.1s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ✅ | Q01 | single_paper | 100.0% | 100.0% | 1.0 | 704.4s | C0/U0 |
| ⚠️ | Q02 | single_paper | 100.0% | 100.0% | 0.737 | 978.1s | C0/U12 |
| ✅ | Q03 | figure_dependent | 100.0% | 100.0% | 1.0 | 932.5s | C0/U0 |
| ⚠️ | Q04 | table_lookup | 100.0% | 100.0% | 0.731 | 752.9s | C0/U8 |
| ⚠️ | Q05 | single_paper | 100.0% | 100.0% | 0.438 | 627.6s | C0/U12 |
| ❌ | Q06 | multi_chunk | 100.0% | 75.0% | 0.261 | 1154.1s | C0/U21 |
| ⚠️ | Q07 | figure_dependent | 100.0% | 100.0% | 0.606 | 978.4s | C0/U15 |
| ⚠️ | Q08 | cross_paper | 100.0% | 100.0% | 0.667 | 1738.7s | C0/U11 |
| ✅ | Q09 | cross_paper | 100.0% | 100.0% | 0.983 | 1867.1s | C0/U1 |
| ⚠️ | Q10 | contradiction_check | 100.0% | 100.0% | 0.6 | 1467.8s | C0/U15 |
| ✅ | Q11 | out_of_scope | N/A | N/A | — | 367.7s | C0/U0 |
| ✅ | Q12 | false_premise | N/A | N/A | — | 573.9s | C0/U0 |

## 逐題細節

### ✅ Q01 · single_paper

**問題**：In the study of JPH203's preincubation inhibitory effect on LAT1, how does preincubation change JPH203's inhibitory potency? Give the reported values.

- detected_paper：`None`
- 選出論文：['1-s2.0-S1347861320300633-main', '1-s2.0-S0021979710012403-main', '1-s2.0-S1936523321002047-main', 's41421-024-00697-6']
- gold_papers：['1-s2.0-S1347861320300633-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：1.0
- 延遲：704.4s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
> * 預孵育（preincubation）結合共同孵育（co-incubation）會產生協同效應，從而增強 JPH203 的抑制效能 (1-s2.0-S1347861320300633-main)。
> * 當僅進行共同孵育時，JPH203 的 $\text{IC}_{50}$ 值為 $99.2 \pm 11.0\text{ nM}$ (1-s2.0-S1347861320300633-main)。
> * 當加入預孵育時，JPH203 的 $\text{IC}_{50}$ 值降低至 $34.2 \pm 3.6\text{ nM}$ (1-s2.0-S1347861320300633-main)。
> * 在簡單共同孵育期間，JPH203 的 $\text{K}_i$ 值為 $0.46\text{ mM}$ (1-s2.0-S1347861320300633-main)。
> * 預孵育與共同孵育結合後的 $\text{K}_i$ 值為 $0.37\text{ mM}$ (1-s2.0-S1347861320300633-main)。
> * JPH203 的協同效應特徵為 $\text{IC}_{50}$ 值降低三倍 (1-s2.0-S1347861320300633-main)。
> * JPH203 的協同效應特徵為 $\text{K}_i$ 值降低 20% (1-s2.0-S1347861320300633-main)。
> * 此協同作用被定義為輕微至中度水平 (1-s2.0-S1347861320300633-main)。
> 
> ## 【跨文獻推論】
> * 跨論文推論（基於 1-s2.0-S1347861320300633-main）：預孵育導致的抑制效能增加與抑制機制的改變有關；雖然單獨共同孵育表現出競爭性抑制（$\text{V}_{\max}$ 持平，$\text{K …（完整內容見 JSON）

---

### ⚠️ Q02 · single_paper

**問題**：What is the hybrid process used for the asymmetric synthesis of optically pure L-p-boronophenylalanine, and what are its key steps?

- detected_paper：`None`
- 選出論文：['bbb0683', 'CMDC-20-e202500059']
- gold_papers：['bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.737
- 延遲：978.1s　問題標記：{'conflicts': 0, 'unsupported': 12}

**答案預覽**：

> 光學純 L-p-硼苯丙氨酸 (L-BPA) 的不對稱合成雜化製程詳細說明如下：
> 
> ## 【論文直接依據】
> * 合成光學純 L-BPA 的雜化製程結合了「不對稱烷基化」以及隨後的「酶水解」(bbb0683)。
> * 與先前報導的方法相比，此合成路徑的反應步驟較少 (bbb0683)。
> * 此路徑的操作和後處理過程比以往的方法更簡單 (bbb0683)。
> * 第一步是保護反應 (bbb0683)。
> * 在第一步中，使用 4-溴甲基苯硼酸 (1) 或 19 (5.99 g, 27.9 mmol) 作為起始原料 (bbb0683)。
> * 在第一步中，使用 2,2-二甲基-1,3-丙二醇 (2.91 g, 27.9 mmol) 作為試劑 (bbb0683)。
> * 第一步在乾燥的 THF 中進行 (bbb0683)。
> * 第一步在環境溫度下進行 (bbb0683)。
> * 第一步涉及攪拌 10 分鐘 (bbb0683)。
> * 第一步產生環狀硼酸酯 (2)，產率為 79% (bbb0683)。
> * 第二步是不對稱/對映選擇性烷基化 (bbb0683)。
> * 在第二步中，使用 n-BuLi (1.68M in hexane) 對手性輔助劑 (2R)-(-)-2,5-二氫-2-異丙基-3,6-二甲氧基吡嗪 (bislactimether, 3) 進行鋰化 (bbb0683)。
> * 鋰化後的輔助劑隨後與受保護的 4-溴甲基苯硼酸衍生物 (2) 反應，產生加成產物 4 (bbb0683)。
> * 第二步在氬氣環境下的乾燥 THF 中進行 (bbb0683)。
> * 第二步在 -78°C 下進行 (bbb0683)。
> * 在第二步中，n-BuLi 在 30 分鐘內加入 (bbb0683)。
> * 加入 n-BuLi 後，混合物靜置 1 小時 (bbb0683)。
> * 然後在 30 分鐘內加入物質 2 的溶液 …（完整內容見 JSON）

---

### ✅ Q03 · figure_dependent

**問題**：According to the cryo-EM structure, how does JPH203 bind to and inhibit the LAT1-4F2hc complex?

- detected_paper：`None`
- 選出論文：['s41421-024-00697-6', '1-s2.0-S1347861320300633-main']
- gold_papers：['s41421-024-00697-6']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：1.0
- 延遲：932.5s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
> 
> **結合位點與結構**
> * JPH203 結合於 LAT1 的傳統底物結合口袋中 (s41421-024-00697-6)。
> * JPH203 的分子結構包含一個具有苯丙氨酸（phenylalanine）骨架的親水性頭部基團 (s41421-024-00697-6)。
> * JPH203 的分子結構包含一個具有 5-amino-2-(3-aminophenyl) benzoxazole 骨架的疏水性尾部基團 (s41421-024-00697-6)。
> 
> **分子交互作用**
> * JPH203 親水性頭部的 $\alpha$-氨基與 TM1 和 TM6 未捲曲（或展開）區域的主鏈原子形成氫鍵網絡，以穩定該分子 (s41421-024-00697-6)。
> * JPH203 親水性頭部的 $\alpha$-羧基與 TM1 和 TM6 未捲曲（或展開）區域的主鏈原子形成氫鍵網絡，以穩定該分子 (s41421-024-00697-6)。
> * JPH203 雙鹵代酪氨酸（tyrosine）核心上的氯原子與 Tyr259 形成鹵鍵 (s41421-024-00697-6)。
> * JPH203 的疏水性尾部嵌入由 TM1 (Ile63, Ile64, Ile68)、TM3 (Ile140, Ile147, Ile148)、TM6 (Phe252, Gly255) 和 TM10 (Ile397, Phe400, Phe403) 組成的疏水性口袋中 (s41421-024-00697-6)。
> * JPH203 尾部的氨基與 TM10 上 Phe400 的主鏈形成氫鍵 (s41421-024-00697-6)。
> * JPH203 尾部的苯環與 Phe252 的側鏈產生 T 型 $\pi$-$\pi$ 交互作用 (s41421-024-00697-6)。
> 
> **結構與抑制機 …（完整內容見 JSON）

---

### ⚠️ Q04 · table_lookup

**問題**：What are the main degradation products and impurities of BPA identified by HPLC, and under which storage conditions do they form?

- detected_paper：`None`
- 選出論文：['1-s2.0-S0731708511003451-main', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water']
- gold_papers：['1-s2.0-S0731708511003451-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.731
- 延遲：752.9s　問題標記：{'conflicts': 0, 'unsupported': 8}

**答案預覽**：

> ## 【論文直接依據】
> 
> **降解產物與條件**
> * BPA 在鹼性和氧化條件下會降解為酪胺酸 (Tyrosine) (1-s2.0-S0731708511003451-main)。
> * 當 BPA 溶解於 $100\text{ mM NaOH}$ 並在 $55^\circ\text{C}$ 下孵育 24 小時時，會產生酪胺酸 (Tyrosine) (1-s2.0-S0731708511003451-main)。
> * 在使用 $6\text{ mM H}_2\text{O}_2$ 的氧化條件下會產生酪胺酸 (Tyrosine) (1-s2.0-S0731708511003451-main)。
> * 在 $6\text{ mM H}_2\text{O}_2$ 的氧化條件下，BPA 降解為酪胺酸 (Tyrosine) 的速度極快 (1-s2.0-S0731708511003451-main)。
> * 酪胺酸 (Tyrosine) 的 HPLC 保留時間為 4.5 分鐘 (1-s2.0-S0731708511003451-main)。
> * 酪胺酸 (Tyrosine) 在波長 230 nm 處進行定量分析 (1-s2.0-S0731708511003451-main)。
> * 在冷凍乾燥的 BPA/甘露醇 (mannitol) 藥品中會產生苯丙胺酸 (Phenylalanine) (1-s2.0-S0731708511003451-main)。
> * 該冷凍乾燥藥品配方包含 $100\text{ mg/ml}$ BPA 和 $110\text{ mg/ml}$ 甘露醇 (mannitol) (1-s2.0-S0731708511003451-main)。
> * 冷凍乾燥藥品配方的 pH 值被調整至 $8.0 \pm 0.1$ (1-s2.0-S0731708511003451-main)。 …（完整內容見 JSON）

---

### ⚠️ Q05 · single_paper

**問題**：What is the solvent-free green protocol for N-Boc protection of amines using picric acid as a catalyst, and what reaction conditions are used?

- detected_paper：`None`
- 選出論文：['1-s2.0-S2773223124000268-main']
- gold_papers：['1-s2.0-S2773223124000268-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.438
- 延遲：627.6s　問題標記：{'conflicts': 0, 'unsupported': 12}

**答案預覽**：

> ## 【論文直接依據】
> * 苦味酸被用作 Brønsted 酸或有機催化劑 (1-s2.0-S2773223124000268-main)。
> * 使用的苦味酸量為 0.2 當量 (1-s2.0-S2773223124000268-main)。
> * 使用的苦味酸量為 2 mol% (1-s2.0-S2773223124000268-main)。
> * 二叔丁基二碳酸酯 ((Boc)₂O / Boc anhydride) 被用作保護試劑 (1-s2.0-S2773223124000268-main)。
> * 反應在無溶劑（neat）條件下進行 (1-s2.0-S2773223124000268-main)。
> * 反應在室溫 (rt) 下進行 (1-s2.0-S2773223124000268-main)。
> * 最佳反應時間為 60 分鐘 (1-s2.0-S2773223124000268-main)。
> * 此方法達到的最高產率為 98% (1-s2.0-S2773223124000268-main)。
> * 在優化條件下，芳香族一級胺的產率為 95% (1-s2.0-S2773223124000268-main)。
> * 此方法適用於芳香族胺 (1-s2.0-S2773223124000268-main)。
> * 此方法適用於雜環芳香族胺 (1-s2.0-S2773223124000268-main)。
> * 此方法適用於開鏈脂肪族胺 (1-s2.0-S2773223124000268-main)。
> * 此方法適用於環狀脂肪族胺 (1-s2.0-S2773223124000268-main)。
> * 在反應機制中，苦味酸首先活化 Boc anhydride 中羰基的氧原子 (1-s2.0-S2773223124000268-main)。
> * 隨後胺的氮原子攻擊羰基碳 (1-s2.0-S277 …（完整內容見 JSON）

---

### ❌ Q06 · multi_chunk

**問題**：How does poly(vinyl alcohol) enhance the therapeutic effect of 4-L-boronophenylalanine in neutron capture therapy of thoracic tumors? Explain the proposed mechanism with supporting data.

- detected_paper：`None`
- 選出論文：['1-s2.0-S0378517325007926-main', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'CMDC-20-e202500059']
- gold_papers：['1-s2.0-S0378517325007926-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：75.0%　grounding：0.261
- 延遲：1154.1s　問題標記：{'conflicts': 0, 'unsupported': 21}

**答案預覽**：

> ## 【論文直接依據】
> * 聚乙烯醇 (Poly(vinyl alcohol), PVA) 通過硼酸酯與多個 BPA 分子反應，形成稱為 PVA-BPA 的複合物 (1-s2.0-S0378517325007926-main)。
> * PVA-BPA 複合物保留了與 LAT1 交互作用所需的關鍵類苯丙氨酸 (phenylalanine-like) 結構 (1-s2.0-S0378517325007926-main)。
> * PVA-BPA 在與 LAT1 交互作用後，通過稱為「LAT1 介導的胞吞作用」(LAT1-mediated endocytosis) 的過程進入癌細胞 (1-s2.0-S0378517325007926-main)。
> * 一旦進入細胞，PVA-BPA 會定位於內體/溶酶體 (endo-/lysosomes) 中 (1-s2.0-S0378517325007926-main)。
> * PVA-BPA 的攝取路徑增加了細胞對 BPA 的攝取量 (1-s2.0-S0378517325007926-main)。
> * PVA-BPA 的攝取路徑延緩了 BPA 從細胞質中的外排 (efflux) (1-s2.0-S0378517325007926-main)。
> * 傳統的 BPA 依賴於 LAT1，通過與胞內氨基酸（如麩醯胺酸, glutamine）交換，將胞外 BPA 導入細胞質 (1-s2.0-S0378517325007926-main)。
> * 當胞外濃度降低時，傳統的 BPA 通過與胞外酪胺酸 (tyrosine) 交換而流出細胞 (1-s2.0-S0378517325007926-main)。
> * PVA-BPA 通過對 LAT1 的主動靶向，在 1 h 內實現了腫瘤的高效累積 (1-s2.0-S0378517325007926-main)。
> * PVA-BP …（完整內容見 JSON）

---

### ⚠️ Q07 · figure_dependent

**問題**：What is the water-stable boroxine structure reported, and what role do the dynamic covalent bonds play in its fluoride binding and hydrogel formation?

- detected_paper：`None`
- 選出論文：['41467_2024_Article_45464', 'Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation']
- gold_papers：['41467_2024_Article_45464']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.606
- 延遲：978.4s　問題標記：{'conflicts': 0, 'unsupported': 15}

**答案預覽**：

> ⚠️ **資料可能不足**：檢索內容僅部分涵蓋此問題，以下回答可能不完整，請謹慎參考並自行查證。
> 
> ## 【論文直接依據】
> * 發現了一種由 2-羥基苯硼酸 (HO-PBA) 衍生的水穩定硼氧環結構 (41467_2024_Article_45464)。
> * 此水穩定硼氧環結構表現出卓越的 pH 穩定性 (41467_2024_Article_45464)。
> * 該結構中的硼-氧 (B–O) 鍵是水兼容的動態共價鍵 (DCBs) (41467_2024_Article_45464)。
> * 這些 B–O DCBs 在室溫的水相介質中可進行快速交換 (41467_2024_Article_45464)。
> * HO-PBA 在環境條件下會自發脫水形成二聚體 (41467_2024_Article_45464)。
> * 在室溫下接觸水時，HO-PBA 二聚體會迅速轉化為硼氧環結構，被鑑定為 HO-PBA 三聚體–$\text{H}_2\text{O}$ 複合物 (41467_2024_Article_45464)。
> * 使用核磁共振光譜 (NMR) 確認了此水穩定硼氧環結構的存在 (41467_2024_Article_45464)。
> * 使用質譜 (MS) 確認了此水穩定硼氧環結構的存在 (41467_2024_Article_45464)。
> * 使用 UV-Raman 光譜確認了此水穩定硼氧環結構的存在 (41467_2024_Article_45464)。
> * 通過理論計算確認了此水穩定硼氧環結構的存在 (41467_2024_Article_45464)。
> * 在水相介質中，HO-PBA 三聚體能選擇性地識別氟離子 ($\text{F}^-$) (41467_2024_Article_45464)。
> * 在水相介質中向 HO-PBA 三聚體加入 $\text{Cl}^-$,  …（完整內容見 JSON）

---

### ⚠️ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.667
- 延遲：1738.7s　問題標記：{'conflicts': 0, 'unsupported': 11}

**答案預覽**：

> ## 【論文直接依據】
> 
> ### 一般考量與工業限制
> * $^{10}\text{B}$ 純同位素形式比普通硼酸貴 1000 倍以上 (CMDC-20-e202500059)。
> * $^{10}\text{B}$ 通常透過化學交換蒸餾分離，然後轉化為硼酸或各種硼酸酯 (CMDC-20-e202500059)。
> * 依賴 $\text{B}_2\text{pin}_2$ 的合成路徑較不被推薦，因為它會導致 $^{10}\text{B}$ 原子核的浪費且成本極高 (CMDC-20-e202500059)。
> * 氨基酸的直接官能基化比修改二鹵代苯更具可擴展性，後者在過去十年中因步驟過多而大多被放棄 (CMDC-20-e202500059)。
> * 在超過克級規模的反應中，使用水/丙酮混合物中的 $\text{HCl}$ 進行脫保護通常會留下殘餘的硼酸酯 (CMDC-20-e202500059)。
> * 添加 $\text{NaIO}_4$ 可以限制脫保護過程中硼酸酯的殘留，但 $\text{NaIO}_4$ 具有毒性，其小鼠腹腔注射 $\text{LD}_{50}$ 為 $58\text{ mg kg}^{-1}$，且在大規模生產中存在製程安全風險 (CMDC-20-e202500059)。
> * 在最後一步使用 Pd 催化氫化會對人類藥品（尤其是非口服藥品）帶來污染風險，因為過濾可能無法完全去除重金屬痕跡 (CMDC-20-e202500059)。
> * 評估 L-BPA 合成成本效益最相關的參數是硼引入後的總步驟數和產率 (CMDC-20-e202500059)。
> 
> ### 路徑：有機金屬官能基化（經由 4-碘苯丙氨酸）
> * 在 Boc 保護的 4-碘苯丙氨酸上使用 $\text{n-BuLi}$ 和 $\text{B(OBu)}_3$ 的路徑，其兩步總產率為 $53\text{-- …（完整內容見 JSON）

---

### ✅ Q09 · cross_paper

**問題**：Several papers target LAT1 for cancer therapy. What different therapeutic strategies targeting LAT1 are described across the literature, and how do they differ in mechanism?

- detected_paper：`None`
- 選出論文：['LAT1 ChemComm 2026', '1-s2.0-S0378517325007926-main', '1-s2.0-S1936523321002047-main', '1-s2.0-S1347861320300633-main', 's41421-024-00697-6']
- gold_papers：['1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 'LAT1 ChemComm 2026', 's41421-024-00697-6']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.983
- 延遲：1867.1s　問題標記：{'conflicts': 0, 'unsupported': 1}

**答案預覽**：

> 根據提供的文獻，針對 LAT1 的治療策略可分為三種主要方法：將 LAT1 作為硼中子捕獲治療 (BNCT) 的遞送門戶、直接藥理學抑制 LAT1 以使癌細胞飢餓或增加其對輻射的敏感性，以及對 LAT1 表達進行基因調控。
> 
> ## 【論文直接依據】
> 
> ### 1. BNCT 的硼遞送
> **BPA (p-boronophenylalanine)**
> * BPA 經由 LAT1 從細胞外空間運輸至細胞質中 [1-s2.0-S0378517325007926-main]。
> * 在進入細胞的過程中，BPA 會將細胞內的氨基酸（如麩醯胺酸）運出細胞 [1-s2.0-S0378517325007926-main]。
> * 當細胞外 BPA 濃度降低時，它會與細胞外的酪胺酸進行交換 [1-s2.0-S0378517325007926-main]。
> * 由於停留時間短，BPA 需要在熱中子照射前或照射期間持續輸注 30-60 分鐘 [1-s2.0-S0378517325007926-main]。
> 
> **PVA-BPA 與 PVA-sorbitol-BPA**
> * PVA-BPA 是透過硼酸酯與多個 BPA 分子形成的複合物 [1-s2.0-S0378517325007926-main]。
> * PVA-BPA 透過「LAT1 介導的內吞作用」被癌細胞吸收 [1-s2.0-S0378517325007926-main]。
> * PVA-BPA 定位於內體/溶酶體中，這增加了細胞吸收並延緩了其從細胞質中的流出 [1-s2.0-S0378517325007926-main]。
> * PVA-sorbitol-BPA 使用山梨醇將 pH 值調整至 7.6 [1-s2.0-S0378517325007926-main]。
> * PVA-sorbitol-BPA 的 pH 值調整解決了原先 PVA-BPA 為維 …（完整內容見 JSON）

---

### ⚠️ Q10 · contradiction_check

**問題**：Do the papers agree on the hydrolytic stability of boroxine structures? Describe any agreements or differences between them.

- detected_paper：`None`
- 選出論文：['Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation', '41467_2024_Article_45464', 'a-method-for-the-deprotection-of-alkylpinacolyl-boronate-esters']
- gold_papers：['41467_2024_Article_45464', 'Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.6
- 延遲：1467.8s　問題標記：{'conflicts': 0, 'unsupported': 15}

**答案預覽**：

> ## 【論文直接依據】
> 
> **關於傳統與分子間硼氧環 (Boroxines)：**
> * 傳統硼氧環在水環境中具有水解不穩定性 (【41467_2024_Article_45464】)。
> * 傳統硼氧環在接觸水後會迅速水解為硼酸 (【41467_2024_Article_45464】)。
> * 嘗試透過引入電子捐贈基團來增強傳統硼氧環的水解穩定性，未能從根本上解決該問題 (【41467_2024_Article_45464】)。
> * 嘗試透過引入空間位阻基團來增強傳統硼氧環的水解穩定性，未能從根本上解決該問題 (【41467_2024_Article_45464】)。
> * 嘗試透過與 N-供體配體形成加合物來增強傳統硼氧環的水解穩定性，未能從根本上解決該問題 (【41467_2024_Article_45464】)。
> * 傳統硼氧環的 B–O 鍵表現出動態特性 (【41467_2024_Article_45464】)。
> * 傳統硼氧環的交換反應需要將其甲苯溶液加熱至 60 °C (【41467_2024_Article_45464】)。
> * 傳統硼氧環的交換反應必須在 60 °C 下維持 8 hours (【41467_2024_Article_45464】)。
> * 分子間硼氧烷 4 的水解穩定性低於三環硼氧烷 2 (【Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation】)。
> * 在 $328\text{ K}$ 的 $[D_6]$丙酮中，當 $[H_2O] = 16.9\text{ mM}$ 時，硼氧烷 4 會完全水解為硼酸 3 (【Chemistry A European J - 2023 - Ono - Str …（完整內容見 JSON）

---

### ✅ Q11 · out_of_scope

**問題**：What are the phase III clinical trial overall survival results for BNCT in glioblastoma patients?

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', '1-s2.0-S0378517325007926-main', 'cas0106-0279']
- gold_papers：[]
- 論文選擇命中率：N/A　檢索覆蓋率：N/A　grounding：—
- 延遲：367.7s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ⚠️ **資料來源說明**：本地學術文獻資料庫中未找到與此問題直接相關的內容。以下回答來自模型自身知識，非論文原文，請謹慎參考並自行查證。
> 
> （此為模型推測，建議查閱原始文獻確認）
> 
> 關於您詢問的硼中子捕獲治療（Boron Neutron Capture Therapy, BNCT）在膠質母細胞瘤（Glioblastoma）患者中的第三期臨床試驗整體生存率（Overall Survival, OS）結果，其情況說明如下：
> 
> **首先，必須指出該問題的前提存在偏差：目前在學術界與臨床醫學領域中，BNCT 治療膠質母細胞瘤尚未有廣泛公認且定論的第三期臨床試驗數據。**
> 
> （此為模型推測，建議查閱原始文獻確認）
> 
> 詳細定性說明如下：
> 
> 1.  **研究階段限制**：目前 BNCT 在膠質母細胞瘤治療上的研究大多集中於第一期（Phase I）或第二期（Phase II）臨床試驗。這些研究的主要目的在於評估治療的安全性、耐受性、劑量優化以及初步的療效探索，而非旨在提供大規模隨機對照的生存率統計結論。
> 2.  **缺乏標準對照組之大規模數據**：由於 BNCT 需要特殊的中子源設備（如加速器或反應爐），其設施分佈有限，導致難以在短期內招募足夠數量的受試者以進行符合第三期臨床試驗標準的大規模、多中心隨機對照研究。因此，目前缺乏能與現行標準治療方案（例如 Stupp protocol）直接對比且具有統計學顯著性的整體生存率數據。
> 3.  **結論**：由於上述原因，目前不存在可供引用之第三期臨床試驗的具體生存數值或百分比。若您在某些文獻中看到生存率數據，該數據通常來自於小樣本的病例系列報告（Case Series）或早期階段的臨床研究，而非正式的第三期臨床試驗結果。
> 
> （此為模型推測，建議查閱原始文獻確認）

---

### ✅ Q12 · false_premise

**問題**：Since BPA is administered orally in BNCT, what oral bioavailability values are reported in these papers?

- detected_paper：`None`
- 選出論文：['cas0106-0279', 'Chem Biol Drug Des - 2013 - Achilli - Folic Acid‐Conjugated 4‐Amino‐Phenylboronate  a Boron‐Containing Compound Designed']
- gold_papers：[]
- 論文選擇命中率：N/A　檢索覆蓋率：N/A　grounding：—
- 延遲：573.9s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ⚠️ **誠實棄答**：檢索到的內容與此問題相關，但並不包含可直接回答的資訊。為避免編造不存在的數據，本系統選擇不作答。建議換個問法，或確認該主題是否涵蓋於文獻庫中。

---
