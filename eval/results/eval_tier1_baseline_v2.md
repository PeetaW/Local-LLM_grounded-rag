# Eval Report — `tier1_baseline_v2`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-06-22 04:14
- 題數：12

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.583 |
| 平均 grounding 分數 | 0.752 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 97.5% |
| 平均總延遲 | 889.4s |
| 平均 planning 延遲 | 6.6s |
| 平均 retrieval 延遲 | 326.5s |
| 平均 grounding 延遲 | 41.3s |
| └ 其中 NLI | 25.1s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ✅ | Q01 | single_paper | 100.0% | 100.0% | 0.889 | 740.9s | C0/U1 |
| ⚠️ | Q02 | single_paper | 100.0% | 100.0% | 0.717 | 960.1s | C0/U18 |
| ✅ | Q03 | figure_dependent | 100.0% | 100.0% | 1.0 | 872.1s | C0/U0 |
| ✅ | Q04 | table_lookup | 100.0% | 100.0% | 0.852 | 783.9s | C0/U6 |
| ⚠️ | Q05 | single_paper | 100.0% | 100.0% | 0.533 | 597.3s | C0/U10 |
| ⚠️ | Q06 | multi_chunk | 100.0% | 75.0% | 0.722 | 1125.7s | C0/U8 |
| ⚠️ | Q07 | figure_dependent | 100.0% | 100.0% | 0.476 | 821.0s | C0/U14 |
| ✅ | Q08 | cross_paper | 100.0% | 100.0% | 0.87 | 871.4s | C0/U4 |
| ✅ | Q09 | cross_paper | 100.0% | 100.0% | 1.0 | 1600.9s | C0/U0 |
| ⚠️ | Q10 | contradiction_check | 100.0% | 100.0% | 0.618 | 1204.1s | C0/U16 |
| ✅ | Q11 | out_of_scope | N/A | N/A | — | 375.2s | C0/U0 |
| ⚠️ | Q12 | false_premise | N/A | N/A | 0.6 | 719.7s | C0/U4 |

## 逐題細節

### ✅ Q01 · single_paper

**問題**：In the study of JPH203's preincubation inhibitory effect on LAT1, how does preincubation change JPH203's inhibitory potency? Give the reported values.

- detected_paper：`None`
- 選出論文：['1-s2.0-S1347861320300633-main', '1-s2.0-S0021979710012403-main', '1-s2.0-S1936523321002047-main', 's41421-024-00697-6']
- gold_papers：['1-s2.0-S1347861320300633-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.889
- 延遲：740.9s　問題標記：{'conflicts': 0, 'unsupported': 1}

**答案預覽**：

> ## 【論文直接依據】
> * JPH203 能夠對 LAT1 產生預孵育（preincubation）抑制效果 【1-s2.0-S1347861320300633-main】。
> * 預孵育效果能協同增強共同孵育（co-incubation）的抑制效果 【1-s2.0-S1347861320300633-main】。
> * 在 HT-29 細胞中，僅進行共同孵育時 JPH203 的 $\text{IC}_{50}$ 值為 $99.2 \pm 11.0\text{ nM}$ 【1-s2.0-S1347861320300633-main】。
> * 在 HT-29 細胞中，進行預孵育加共同孵育時 JPH203 的 $\text{IC}_{50}$ 值為 $34.2 \pm 3.6\text{ nM}$ 【1-s2.0-S1347861320300633-main】。
> * 僅進行共同孵育時 JPH203 的 $\text{K}_i$ 值為 $0.46\text{ mM}$ 【1-s2.0-S1347861320300633-main】。
> * 進行預孵育加共同孵育時 JPH203 的 $\text{K}_i$ 值為 $0.37\text{ mM}$ 【1-s2.0-S1347861320300633-main】。
> * 預孵育的協同效應表現為 $\text{IC}_{50}$ 值降低三倍 【1-s2.0-S1347861320300633-main】。
> * 預孵育的協同效應表現為 $\text{K}_i$ 值降低 20% 【1-s2.0-S1347861320300633-main】。
> * 此協同水平被定義為輕微至中度 【1-s2.0-S1347861320300633-main】。
> 
> ## 【跨文獻推論】
> * 跨論文推論（基於 【1-s2.0-S1347861320300633-main】） …（完整內容見 JSON）

---

### ⚠️ Q02 · single_paper

**問題**：What is the hybrid process used for the asymmetric synthesis of optically pure L-p-boronophenylalanine, and what are its key steps?

- detected_paper：`None`
- 選出論文：['bbb0683', 'CMDC-20-e202500059']
- gold_papers：['bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.717
- 延遲：960.1s　問題標記：{'conflicts': 0, 'unsupported': 18}

**答案預覽**：

> ## 【論文直接依據】
> 
> **通用製程**
> * L-p-硼苯丙氨酸 (L-BPA) 是一種腫瘤尋跡氨基酸 【bbb0683】。
> * L-BPA 是硼中子捕獲治療 (BNCT) 最有效的硼載體之一 【bbb0683】。
> * 合成光學純 L-BPA 的混合製程結合了對映選擇性烷基化與隨後的酶促水解 【bbb0683】。
> 
> **步驟 1：起始原料保護**
> * 起始原料為 4-溴甲基苯硼酸 (1) 【bbb0683】。
> * 化合物 1 與 2,2-二甲基-1,3-丙二醇反應 【bbb0683】。
> * 此反應在乾燥 THF 中進行 【bbb0683】。
> * 反應於環境溫度下執行 【bbb0683】。
> * 此步驟的攪拌時間為 10 分鐘 【bbb0683】。
> * 該過程將二羥基硼酸基團保護為環狀硼酸酯 (2) 【bbb0683】。
> * 化合物 2 的產率為 79% 【bbb0683】。
> 
> **步驟 2：對映選擇性烷基化**
> * 此步驟在氬氣環境下的乾燥 THF 中進行 【bbb0683】。
> * 反應溫度為 -78°C 【bbb0683】。
> * 反應物為鋰化的 $(2R)-(-)$-2,5-二氫-2-異丙基-3,6-二甲氧基吡嗪 (3) 以及受保護的 4-溴甲基苯硼酸衍生物 (2) 【bbb0683】。
> * n-BuLi（於正己烷中濃度為 1.68M）與手性輔助劑 (3) 共同使用 【bbb0683】。
> * n-BuLi 在 30 分鐘內加入 【bbb0683】。
> * n-BuLi 加入後，混合物攪拌 1 小時 【bbb0683】。
> * 隨後在 30 分鐘內加入化合物 2 的溶液 【bbb0683】。
> * 接著將混合物攪拌 15 小時 【bbb0683】。
> * 過程以使用 pH 7.0 的磷酸鹽緩衝液處理結束 【bbb0683】。
> * 此步驟產生具有 2S 構型的加成產物 4 【bbb0 …（完整內容見 JSON）

---

### ✅ Q03 · figure_dependent

**問題**：According to the cryo-EM structure, how does JPH203 bind to and inhibit the LAT1-4F2hc complex?

- detected_paper：`None`
- 選出論文：['s41421-024-00697-6', '1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 's41467-024-45464-z (1)', 'LAT1 ChemComm 2026']
- gold_papers：['s41421-024-00697-6']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：1.0
- 延遲：872.1s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
> 
> **結合位點與結構**
> * JPH203 結合於 LAT1 的傳統底物結合口袋內 [Chunk 1, Chunk 3]。
> * JPH203 的分子結構包含一個具有苯丙氨酸（phenylalanine）骨架的親水性頭部 [Chunk 1, Chunk 3]。
> * JPH203 的分子結構包含一個具有 5-amino-2-(3-aminophenyl) benzoxazole 骨架的疏水性尾部 [Chunk 1, Chunk 3]。
> * JPH203 結合於 LAT1-4F2hc 複合體的向外開放（outward-facing）結構 [Chunk 1, Chunk 2]。
> * 該複合體的低溫電子顯微鏡（cryo-EM）解析度為 3.30 Å [Chunk 1, Chunk 2]。
> * 針對 JPH203 的聚焦精修（focused refinement）解析度為 3.25 Å [Chunk 1, Chunk 2]。
> 
> **分子間交互作用（頭部與核心）**
> * JPH203 頭部的 $\alpha$-氨基與 TM1 和 TM6 未捲曲/展開區域的主鏈原子形成氫鍵網絡 [Chunk 1, Chunk 2, Chunk 3]。
> * JPH203 頭部的 $\alpha$-羧基與 TM1 和 TM6 未捲曲/展開區域的主鏈原子形成氫鍵網絡 [Chunk 1, Chunk 2, Chunk 3]。
> * JPH203 二鹵酪胺酸（dihalo-tyrosine）核心上的氯原子與 Tyr259 形成鹵鍵（halogen bond）[Chunk 1, Chunk 2, Chunk 3]。
> 
> **分子間交互作用（尾部）**
> * JPH203 的疏水性尾部嵌入一個疏水性口袋中 [Chunk 1, Chunk 2, Chunk 3]。
> * 此疏水性口袋由 TM1 (Ile6 …（完整內容見 JSON）

---

### ✅ Q04 · table_lookup

**問題**：What are the main degradation products and impurities of BPA identified by HPLC, and under which storage conditions do they form?

- detected_paper：`None`
- 選出論文：['1-s2.0-S0731708511003451-main', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water']
- gold_papers：['1-s2.0-S0731708511003451-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.852
- 延遲：783.9s　問題標記：{'conflicts': 0, 'unsupported': 6}

**答案預覽**：

> ## 【論文直接依據】
> 
> **降解產物及其形成條件**
> * Boronphenylalanine (BPA) 的降解產物包括酪胺酸 【1-s2.0-S0731708511003451-main】。
> * BPA 的降解產物包括苯丙胺酸 【1-s2.0-S0731708511003451-main】。
> * BPA 的降解產物包括母體芳環化合物 (parent arene) 【1-s2.0-S0731708511003451-main】。
> * 酪胺酸在鹼性條件下產生 【1-s2.0-S0731708511003451-main】。
> * 酪胺酸在氧化條件下產生 【1-s2.0-S0731708511003451-main】。
> * 在暴露於空氣中的鹼性水溶液中，BPA 向酪胺酸的轉化會加速 【1-s2.0-S0731708511003451-main】。
> * 在使用 6 mM $\text{H}_2\text{O}_2$ 的氧化條件下，BPA 會非常快速地降解為酪胺酸 【1-s2.0-S0731708511003451-main】。
> * 當 BPA 溶解於 100 mM NaOH 並在 55 °C 下孵育 24 小時時，可觀察到酪胺酸 【1-s2.0-S0731708511003451-main】。
> * 苯丙胺酸形成於含有 BPA 和甘露醇 (mannitol) 的冷凍乾燥藥品中 【1-s2.0-S0731708511003451-main】。
> * 苯丙胺酸的形成是一個緩慢的過程 【1-s2.0-S0731708511003451-main】。
> * 苯丙胺酸的形成隨溫度而變化 【1-s2.0-S0731708511003451-main】。
> * 當冷凍乾燥的 BPA 和甘露醇在 40 °C 下儲存 6 個月時，會產生約 1%（相對於 BPA 質量）的苯丙胺酸 【1-s2.0-S073 …（完整內容見 JSON）

---

### ⚠️ Q05 · single_paper

**問題**：What is the solvent-free green protocol for N-Boc protection of amines using picric acid as a catalyst, and what reaction conditions are used?

- detected_paper：`None`
- 選出論文：['1-s2.0-S2773223124000268-main']
- gold_papers：['1-s2.0-S2773223124000268-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.533
- 延遲：597.3s　問題標記：{'conflicts': 0, 'unsupported': 10}

**答案預覽**：

> ## 【論文直接依據】
> * 苦味酸被用作 Brønsted 酸或有機催化劑 【1-s2.0-S2773223124000268-main】。
> * 所使用的苦味酸量為 2 mol% 或 0.2 當量 【1-s2.0-S2773223124000268-main】。
> * 二叔丁基二碳酸酯 ((Boc)₂O / Boc 酐) 被用作保護試劑 【1-s2.0-S2773223124000268-main】。
> * 反應在無溶劑（neat）條件下進行 【1-s2.0-S2773223124000268-main】。
> * 反應在室溫下進行 【1-s2.0-S2773223124000268-main】。
> * 優化後的反應時間為 60 分鐘 【1-s2.0-S2773223124000268-main】。
> * 此方法達到的最高產率為 98% 【1-s2.0-S2773223124000268-main】。
> * 對於芳香族一級胺，在室溫下使用 0.2 當量的催化劑反應 60 分鐘，產率為 95% 【1-s2.0-S2773223124000268-main】。
> * 適用底物包括芳香族胺，例如 2-aminophenol 【1-s2.0-S2773223124000268-main】。
> * 適用底物包括雜環芳香族胺，例如 quinolin-4-amine、benzo[d]thiazol-2-amine 和 1H-imidazole 【1-s2.0-S2773223124000268-main】。
> * 適用底物包括開鏈脂肪族胺 【1-s2.0-S2773223124000268-main】。
> * 適用底物包括環狀脂肪族胺，例如 piperidine、morpholine 和 pyrrolidine 【1-s2.0-S2773223124000268-main】。
> * 提出的機理始於苦味酸活化 B …（完整內容見 JSON）

---

### ⚠️ Q06 · multi_chunk

**問題**：How does poly(vinyl alcohol) enhance the therapeutic effect of 4-L-boronophenylalanine in neutron capture therapy of thoracic tumors? Explain the proposed mechanism with supporting data.

- detected_paper：`None`
- 選出論文：['1-s2.0-S0378517325007926-main', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'CMDC-20-e202500059']
- gold_papers：['1-s2.0-S0378517325007926-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：75.0%　grounding：0.722
- 延遲：1125.7s　問題標記：{'conflicts': 0, 'unsupported': 8}

**答案預覽**：

> ## 【論文直接依據】
> 
> * 聚乙烯醇 (PVA) 透過硼酸酯與 4-L-硼苯丙氨酸 (BPA) 形成複合物 [1-s2.0-S0378517325007926-main]。
> * 所形成的複合物被稱為 PVA-BPA [1-s2.0-S0378517325007926-main]。
> * PVA-BPA 複合物保留了類苯丙氨酸結構，這對於與大型中性氨基酸轉運蛋白 1 (LAT1) 的交互作用至關重要 [1-s2.0-S0378517325007926-main]。
> * PVA-BPA 透過一種稱為 LAT1 介導的內吞作用被癌細胞吸收 [1-s2.0-S0378517325007926-main]。
> * 一旦進入細胞，PVA-BPA 會定位於內體與溶酶體 (endo-/lysosomes) 中 [1-s2.0-S0378517325007926-main]。
> * PVA-BPA 的吸收機制增加了細胞吸收 BPA 的總量 [1-s2.0-S0378517325007926-main]。
> * PVA-BPA 的吸收機制延緩了 BPA 從細胞質的外排 [1-s2.0-S0378517325007926-main]。
> * 這種外排的延緩延長了硼化合物在腫瘤內的滯留時間 [1-s2.0-S0378517325007926-main]。
> * 傳統的 BPA 透過 LAT1 進入細胞 [1-s2.0-S0378517325007926-main]。
> * 當細胞外 BPA 濃度降低時，傳統的 BPA 透過與細胞外酪胺酸 (tyrosine) 交換而從細胞中外排 [1-s2.0-S0378517325007926-main]。
> * PVA-BPA 透過 LAT1 的主動靶向，在 1 hour 內實現了在腫瘤中的高效累積與滯留 [1-s2.0-S0378517325007926-main]。
> * …（完整內容見 JSON）

---

### ⚠️ Q07 · figure_dependent

**問題**：What is the water-stable boroxine structure reported, and what role do the dynamic covalent bonds play in its fluoride binding and hydrogel formation?

- detected_paper：`None`
- 選出論文：['41467_2024_Article_45464', 's41467-024-45464-z (1)']
- gold_papers：['41467_2024_Article_45464']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.476
- 延遲：821.0s　問題標記：{'conflicts': 0, 'unsupported': 14}

**答案預覽**：

> ## 【論文直接依據】
> * 所報導的水穩定硼氧環（boroxine）結構源自於 2-羥基苯硼酸 (HO-PBA) 【s41467-024-45464-z】。
> * HO-PBA 三聚體的穩定範圍為 $2.4 < \text{pH} < 9$ 【s41467-024-45464-z】。
> * $\text{CH}_3\text{-HO-PBA}$ 三聚體的穩定範圍為 $2.4 < \text{pH} < 10$ 【s41467-024-45464-z】。
> * 此結構中的 B–O 鍵在室溫水溶液中會發生快速的交換反應 【s41467-024-45464-z】。
> * 傳統的硼氧環結構需要在甲苯中加熱至 $60^\circ\text{C}$ 並持續 8 小時才能進行交換 【s41467-024-45464-z】。
> * 該結構在水介質中能選擇性地識別氟離子 ($\text{F}^-$) 【s41467-024-45464-z】。
> * 此結構對 $\text{F}^-$ 的親和力顯著強於苯硼酸 (PBA) 【s41467-024-45464-z】。
> * 將一摩爾當量的 PBA 加入到 $\text{NaF}$ 懸浮液中，混合物仍保持在懸浮狀態 【s41467-024-45464-z】。
> * PBA 與 $\text{NaF}$ 混合物的 $^{19}\text{F NMR}$ 信號在 $-135.2\text{ ppm}$ 處較弱 【s41467-024-45464-z】。
> * 將一摩爾當量的 HO-PBA 三聚體加入到 $\text{NaF}$ 懸浮液中，可將懸浮液轉化為透明溶液 【s41467-024-45464-z】。
> * 在向 $\text{NaF}$ 懸浮液中加入 HO-PBA 三聚體後，$^{19}\text{F NMR}$ 中的游離 $\text{F}^-$ 信號消失 【s4 …（完整內容見 JSON）

---

### ✅ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.87
- 延遲：871.4s　問題標記：{'conflicts': 0, 'unsupported': 4}

**答案預覽**：

> ## 【論文直接依據】
> * 目前 4-硼基-L-苯丙氨酸 (L-BPA) 的合成缺乏共識方法 【CMDC-20-e202500059】。
> * 生產高純度、$^{10}\text{B}$ 同位素富集的 L-BPA 材料具有挑戰性 【CMDC-20-e202500059】。
> * 【CMDC-20-e202500059】 是一篇綜合綜述，分析了各種合成路徑的可擴展性 【CMDC-20-e202500059】。
> * 【CMDC-20-e202500059】 是一篇綜合綜述，分析了各種合成路徑的成本效益 【CMDC-20-e202500059】。
> * 【CMDC-20-e202500059】 是一篇綜合綜述，分析了各種合成路徑的安全性 【CMDC-20-e202500059】。
> * 【bbb0683】 提出了一種結合對稱烷基化和酶水解的「混合製程」以生產光學純 L-BPA 【bbb0683】。
> * 【bbb0683】 中的混合製程步驟較少 【bbb0683】。
> * 【bbb0683】 中的混合製程具有操作簡單的特點 【bbb0683】。
> * 【bbb0683】 中的混合製程具有後處理簡單的特點 【bbb0683】。
> * 【bbb0683】 路徑的第一步使 4-溴甲基苯硼酸與 2,2-二甲基-1,3-丙二醇反應，形成環狀硼酸酯 【bbb0683】。
> * 【bbb0683】 中第一步的產率為 79% 【bbb0683】。
> * 【bbb0683】 路徑的第二步在 THF 中於 -78°C 使用 n-BuLi 對 (2R)-(-)-2,5-二氫-2-異丙基-3,6-二甲氧基吡嗪進行鋰化 【bbb0683】。
> * 在 【bbb0683】 的第二步中，鋰化吡嗪與環狀硼酸酯反應 【bbb0683】。
> * 【bbb0683】 的第二步產生對映體過量值 (e.e.) 為 74% 的加成產物 【bbb0 …（完整內容見 JSON）

---

### ✅ Q09 · cross_paper

**問題**：Several papers target LAT1 for cancer therapy. What different therapeutic strategies targeting LAT1 are described across the literature, and how do they differ in mechanism?

- detected_paper：`None`
- 選出論文：['LAT1 ChemComm 2026', '1-s2.0-S0378517325007926-main', '1-s2.0-S1347861320300633-main', 's41421-024-00697-6', '1-s2.0-S1936523321002047-main']
- gold_papers：['1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 'LAT1 ChemComm 2026', 's41421-024-00697-6']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：1.0
- 延遲：1600.9s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
> 
> **LAT1 的基因與小分子抑制**
> * 降低 LAT1 mRNA 表達（mRNA 敲低）可顯著減緩前列腺癌細胞的增殖、侵襲和遷移 【1-s2.0-S1347861320300633-main】。
> * JPH203 是一種選擇性 LAT1 抑制劑 【1-s2.0-S1347861320300633-main, 1-s2.0-S1936523321002047-main】。
> * JPH203 是一種酪胺酸衍生物 【1-s2.0-S1347861320300633-main, 1-s2.0-S1936523321002047-main】。
> * JPH203 的化學名稱為 (S)-2-amino-3-(4-((5-amino-2-phenylbenzo [d] oxazol-7-yl) methoxy)-3, 5-dichlorophenyl) propanoic acid 【1-s2.0-S1347861320300633-main, 1-s2.0-S1936523321002047-main】。
> * JPH203 以競爭方式抑制 LAT1 【1-s2.0-S1347861320300633-main】。
> * JPH203 的 $\text{IC}_{50}$ 值為 $60\text{--}140\text{ nM}$ 【1-s2.0-S1347861320300633-main】。
> * JPH203 在多種癌細胞和腫瘤負荷動物模型中表現出抗增殖效果 【1-s2.0-S1347861320300633-main】。
> * 初步臨床試驗顯示，JPH203 對於結腸癌和膽道癌患者具有療效且可耐受 【1-s2.0-S1347861320300633-main】。
> * JPH203 的 I 期和 II 期臨床試驗已在進階實體腫瘤患者中開展 【1-s2.0-S19 …（完整內容見 JSON）

---

### ⚠️ Q10 · contradiction_check

**問題**：Do the papers agree on the hydrolytic stability of boroxine structures? Describe any agreements or differences between them.

- detected_paper：`None`
- 選出論文：['Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation', '41467_2024_Article_45464', 's41467-024-45464-z (1)', 'a-method-for-the-deprotection-of-alkylpinacolyl-boronate-esters']
- gold_papers：['41467_2024_Article_45464', 'Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.618
- 延遲：1204.1s　問題標記：{'conflicts': 0, 'unsupported': 16}

**答案預覽**：

> ## 【論文直接依據】
> 
> **來自【s41467-024-45464-z】：**
> * 傳統硼氧環（boroxines）通常被認為具有水解不穩定性 [s41467-024-45464-z]。
> * 傳統硼氧環在接觸水或潮濕環境時會迅速水解為硼酸 [s41467-024-45464-z]。
> * 這種水解會導致傳統硼氧環的功能喪失 [s41467-024-45464-z]。
> * 先前提高穩定性的策略包括引入電子捐贈基團 [s41467-024-45464-z]。
> * 先前提高穩定性的策略包括引入龐大基團 [s41467-024-45464-z]。
> * 先前提高穩定性的策略包括與 N-供體配體形成加合物 [s41467-024-45464-z]。
> * 先前提高穩定性的策略包括透過將三個硼酸單元整合到柔性大環中來採用熵穩定化（entropy stabilization） [s41467-024-45464-z]。
> * 這些先前的策略未能從根本上解決水解不穩定性問題 [s41467-024-45464-z]。
> * 2-羥基苯硼酸 (HO-PBA) 可以形成具有水穩定性的硼氧環結構 [s41467-024-45464-z]。
> * HO-PBA 在室溫下會自發脫水形成二聚體 [s41467-024-45464-z]。
> * 接觸水後，HO-PBA 二聚體在室溫下迅速轉化為 HO-PBA 三聚體–H2O 複合物 [s41467-024-45464-z]。
> * 這些 HO-PBA 三聚體–H2O 複合物是水穩定的硼氧環結構 [s41467-024-45464-z]。
> * 由這些結構交聯的 Poly(PEGMEA-co-AMA-(HO-PBA)) 水凝膠在 pH=2 溶液中保持穩定 [s41467-024-45464-z]。
> * 由這些結構交聯的 Poly(PEGMEA-co-AMA-(HO-PBA …（完整內容見 JSON）

---

### ✅ Q11 · out_of_scope

**問題**：What are the phase III clinical trial overall survival results for BNCT in glioblastoma patients?

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', '1-s2.0-S0378517325007926-main', 'cas0106-0279']
- gold_papers：[]
- 論文選擇命中率：N/A　檢索覆蓋率：N/A　grounding：—
- 延遲：375.2s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ⚠️ **資料來源說明**：本地學術文獻資料庫中未找到與此問題直接相關的內容。以下回答來自模型自身知識，非論文原文，請謹慎參考並自行查證。
> 
> 針對您提出的關於「硼中子捕獲治療（Boron Neutron Capture Therapy, BNCT）在膠質母細胞瘤（Glioblastoma, GBM）患者中的第三期臨床試驗總生存期（Overall Survival, OS）結果」之問題，以下是基於目前學術知識的詳細分析。
> 
> ### 核心結論：關於第三期臨床試驗的現狀
> 
> 首先必須明確指出：**截至目前的公開學術紀錄中，針對膠質母細胞瘤（GBM）的 BNCT 治療，尚未有大規模、隨機對照且已完成並發表之「標準第三期臨床試驗（Phase III Clinical Trial）」的 OS 數據。**
> 
> 大多數關於 BNCT 在 GBM 中的研究仍處於第一期（安全性評估）或第二期（初步療效評估）階段。BNCT 的特殊性在於其對中子源（Neutron Source）的高度依賴，且過去多依賴核反應爐，而近年才逐漸轉向醫院可用的加速器型中子源（Accelerator-based BNCT, AB-BNCT），這限制了大規模多中心第三期試驗的推行。
> 
> ### 現有臨床研究（第一、二期）之 OS 趨勢分析
> 
> 雖然缺乏第三期數據，但我們可以從已發表的第二期或小規模臨床研究中觀察到 OS 的表現：
> 
> 1.  **復發性膠質母細胞瘤 (Recurrent GBM)：**
>     在針對復發性 GBM 的小型試驗中，BNCT 顯示出一定的生存期延長潛力。部分研究報告指出，接受 BNCT 的患者中位總生存期（Median OS）可達到 **12 至 20 個月**不等。相較於僅接受化療或姑息治療的復發患者，BNCT 展現了較好的局部控制率。
> 
> 2.  **初診膠質母細胞瘤 (Newly Diagnosed …（完整內容見 JSON）

---

### ⚠️ Q12 · false_premise

**問題**：Since BPA is administered orally in BNCT, what oral bioavailability values are reported in these papers?

- detected_paper：`None`
- 選出論文：['cas0106-0279', 'Chem Biol Drug Des - 2013 - Achilli - Folic Acid‐Conjugated 4‐Amino‐Phenylboronate  a Boron‐Containing Compound Designed']
- gold_papers：[]
- 論文選擇命中率：N/A　檢索覆蓋率：N/A　grounding：0.6
- 延遲：719.7s　問題標記：{'conflicts': 0, 'unsupported': 4}

**答案預覽**：

> ## 【論文直接依據】
> 
> * BNCT 載體需要透過靜脈注射給藥【Chem Biol Drug Des - 2013】。
> * 研究旨在鑑定負責對-硼苯丙氨酸 (p-boronophenylalanine, BPA) 攝取的芳香族氨基酸轉運蛋白【cas0106-0279】。
> * 透過在非洲爪蟾卵母細胞 (*Xenopus oocytes*) 中表達這些轉運蛋白，來鑑定並檢驗能夠運輸 BPA 的轉運蛋白【cas0106-0279】。
> * BPA 對於血細胞和造血前驅細胞具有高生物相容性【Chem Biol Drug Des - 2013】。
> * 在涉及中性粒細胞和血小板的測試中，BPA 並未表現出在 FPB 中觀察到的促炎或促血栓效應【Chem Biol Drug Des - 2013】。
> 
> ## 【跨文獻推論】
> 
> * 跨論文推論（基於【Chem Biol Drug Des - 2013】和【cas0106-0279】）：提供的文獻中未報告 BPA 的任何口服生物利用度數值。此外，問題的前提（BPA 為口服給藥）與【Chem Biol Drug Des - 2013】中明確指出 BNCT 載體必須透過靜脈注射給藥的陳述之間存在矛盾。
> 
> ## 【知識延伸與推測】
> 
> * 模型推測（文獻依據不足）：這些論文中缺乏口服生物利用度數據，加上提及「靜脈注射」以及對 BPA 攝取之特定「轉運蛋白」的研究，表明 BPA 可能具有較差的胃腸道吸收或低口服生物利用度。在臨床實踐中，對-硼苯丙氨酸通常透過靜脈注射給藥，以確保治療濃度能到達腫瘤部位，這與【Chem Biol Drug Des - 2013】中提到的給藥途徑一致。
> 
> ---
> 📋 **答案品質報告**
> 
> 📊 **分段論文依據率：**
> 
>   ⚠️ 【論文直接依據】：60.0%（3/5 句）
>   ❌ 【跨文獻推論】：0.0%（0/1 句）   …（完整內容見 JSON）

---
