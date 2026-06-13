# Eval Report — `rerank_on`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-06-13 11:46
- 題數：4

## 彙總

| 指標 | 值 |
|------|-----|
| 平均 grounding 分數 | 0.58 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | N/A |
| 平均總延遲 | 11821.1s |
| 平均 planning 延遲 | 4.7s |
| 平均 retrieval 延遲 | 268.6s |
| 平均 grounding 延遲 | 10071.6s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ⚠️ | Q01 | single_paper | 100.0% | N/A | 0.6 | 11105.0s | C0/U4 |
| ⚠️ | Q02 | single_paper | 100.0% | N/A | 0.429 | 9729.1s | C0/U9 |
| ⚠️ | Q03 | figure_dependent | 100.0% | N/A | 0.789 | 19299.8s | C0/U6 |
| ⚠️ | Q04 | table_lookup | 100.0% | N/A | 0.5 | 7150.3s | C0/U2 |

## 逐題細節

### ⚠️ Q01 · single_paper

**問題**：In the study of JPH203's preincubation inhibitory effect on LAT1, how does preincubation change JPH203's inhibitory potency? Give the reported values.

- detected_paper：`None`
- 選出論文：['1-s2.0-S1347861320300633-main', '1-s2.0-S0021979710012403-main', '1-s2.0-S1936523321002047-main', 's41421-024-00697-6']
- gold_papers：['1-s2.0-S1347861320300633-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：N/A　grounding：0.6
- 延遲：11105.0s　問題標記：{'conflicts': 0, 'unsupported': 4}

**答案預覽**：

> ## 【論文直接依據】
> * 預孵育（Preincubation）在與共同孵育（co-incubation）結合時，能顯著增強 JPH203 的抑制效力，產生協同效應 【1-s2.0-S1347861320300633-main】。
> * 當加入預孵育時，$\text{IC}_{50}$ 值從 $99.2 \pm 11.0\text{ nM}$（僅共同孵育）降低至 $34.2 \pm 3.6\text{ nM}$ 【1-s2.0-S1347861320300633-main】。
> * 在預孵育與共同孵育的結合分析中，$\text{K}_i$ 值從 $0.46\text{ mM}$（僅共同孵育）降低至 $0.37\text{ mM}$ 【1-s2.0-S1347861320300633-main】。
> * 預孵育的抑制效果具有時間依賴性；具體而言，使用 $10\text{ mM}$ JPH203 處理 30、60 和 120 分鐘，可將亮氨酸（leucine）的攝取活性分別降低至對照組的 42%、32% 和 28% 【1-s2.0-S1347861320300633-main】。
> * 使用 $0.3\text{ mM}$ JPH203 預孵育 120 分鐘，可將活性水平降低至 DMSO 處理細胞的 30% 【1-s2.0-S1347861320300633-main】。
> 
> ## 【跨文獻推論】
> * 跨論文推論（基於 【1-s2.0-S1347861320300633-main】）：預孵育提高了 JPH203 的整體抑制效率，而未改變其基本作用機制。這源於儘管 $\text{IC}_{50}$ 和 $\text{K}_i$ 值降低（表明效力更高），但抑制作用仍為競爭性，這可由共同孵育和預孵育處理中一致的 $\text{V}_{\max}$ 值以及增加的 $\text{K}_m$ 值來證 …（完整內容見 JSON）

---

### ⚠️ Q02 · single_paper

**問題**：What is the hybrid process used for the asymmetric synthesis of optically pure L-p-boronophenylalanine, and what are its key steps?

- detected_paper：`None`
- 選出論文：['bbb0683', 'CMDC-20-e202500059']
- gold_papers：['bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：N/A　grounding：0.429
- 延遲：9729.1s　問題標記：{'conflicts': 0, 'unsupported': 9}

**答案預覽**：

> ## 【論文直接依據】
> 
> 用於光學純 L-p-硼苯丙氨酸 (L-BPA) 不對稱合成的混合製程結合了「對映選擇性烷基化」以及隨後的「酶促水解」[bbb0683]。關鍵步驟如下：
> 
> *   **步驟 1：保護基合成：** 4-溴甲基苯硼酸 (1) 在乾燥 THF 中與 2,2-二甲基-1,3-丙二醇反應，產出環狀硼酸酯 (2)，產率為 79% [bbb0683]。具體而言，將 5.99 g (27.9 mmol) 的物質 1 和 2.91 g (27.9 mmol) 的 2,2-二甲基-1,3-丙二醇溶解於 30 ml 乾燥 THF 中，並在室溫下攪拌 10 分鐘 [bbb0683]。
> *   **步驟 2：立體選擇性烷基化：** (2R)-(-)-2,5-二氫-2-異丙基-3,6-二甲氧基吡嗪 (bislactimether, 3) 使用 n-BuLi 進行鋰化，然後在 -78°C 的乾燥 THF 中與環狀硼酸酯 (2) 反應，生成加成產物 4 [bbb0683]。具體條件包括將 1.84 g (10.0 mmol) 的物質 3 溶解於 36 ml 乾燥 THF 中，在 30 分鐘內加入 n-BuLi (1.68 M in hexane, 5.95 ml, 10 mmol)，等待 1 小時，然後在 30 分鐘內加入由 2.83 g (10.0 mmol) 物質 2 溶解於 6 ml 乾燥 THF 的溶液，隨後在 -78°C 下攪拌 15 小時 [bbb0683]。產物 4 具有 2S 構型，立體選擇性為 74% e.e. (或 72% 非對映異構體過量) [bbb0683]。
> *   **步驟 3：L-BPA 甲酯 (6) 的合成：** 這是一個兩階段過程，從產物 4 到酯 6 的總產率為 85% [bbb0683]：
>     *   第一階段 (4 $\rightarro …（完整內容見 JSON）

---

### ⚠️ Q03 · figure_dependent

**問題**：According to the cryo-EM structure, how does JPH203 bind to and inhibit the LAT1-4F2hc complex?

- detected_paper：`None`
- 選出論文：['s41421-024-00697-6', '1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 's41467-024-45464-z (1)', 'LAT1 ChemComm 2026']
- gold_papers：['s41421-024-00697-6']
- 論文選擇命中率：100.0%　檢索覆蓋率：N/A　grounding：0.789
- 延遲：19299.8s　問題標記：{'conflicts': 0, 'unsupported': 6}

**答案預覽**：

> 根據提供的事實，以下是關於 JPH203 如何結合並抑制 LAT1-4F2hc 複合物的全面回答。
> 
> ## 【論文直接依據】
> **結合位點與分子結構**
> * JPH203 結合於 LAT1 的傳統底物結合口袋中 【s41421-024-00697-6】。
> * 該分子由兩個主要部分組成：具有苯丙氨酸（phenylalanine）骨架的親水性頭部基團，以及具有 5-amino-2-(3-aminophenyl) benzoxazole 骨架的疏水性尾部基團 【s41421-024-00697-6】。
> 
> **分子間相互作用**
> * **親水性頭部：** $\alpha$-氨基和 $\alpha$-羧基與 TM1 和 TM6 未捲曲（或展開）區域的主鏈原子形成氫鍵網絡 【s41421-024-00697-6】。
> * **核心：** 二鹵酪胺酸（dihalo-tyrosine）核心上的氯原子與 Tyr259 形成鹵鍵 【s41421-024-00697-6】。
> * **疏水性尾部：** 
>     * 尾部嵌入由 TM1 (Ile63, Ile64, Ile68)、TM3 (Ile140, Ile147, Ile148)、TM6 (Phe252, Gly255) 和 TM10 (Ile397, Phe400, Phe403) 組成的疏水性口袋中 【s41421-024-00697-6】。
>     * 尾部的氨基與 TM10 上 Phe400 的主鏈形成氫鍵 【s41421-024-00697-6】。
>     * 尾部的苯環與 Phe252 的側鏈發生 T 型 $\pi$-$\pi$ 相互作用 【s41421-024-00697-6】。
> 
> **結構影響與抑制機制**
> * JPH203 將 LAT1-4F2hc 複合物鎖定在面向外部（outward-facing）的結構中（透過低溫電子顯微 …（完整內容見 JSON）

---

### ⚠️ Q04 · table_lookup

**問題**：What are the main degradation products and impurities of BPA identified by HPLC, and under which storage conditions do they form?

- detected_paper：`None`
- 選出論文：['1-s2.0-S0731708511003451-main', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water']
- gold_papers：['1-s2.0-S0731708511003451-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：N/A　grounding：0.5
- 延遲：7150.3s　問題標記：{'conflicts': 0, 'unsupported': 2}

**答案預覽**：

> ## 【論文直接依據】
> **降解產物：**
> - **酪胺酸 (Tyrosine)：** 在鹼性和氧化條件下由 BPA 降解而形成 【1-s2.0-S0731708511003451-main】。在氧化條件下，降解速度非常快 【1-s2.0-S0731708511003451-main】。在鹼性水溶液中，此過程可能會因接觸空氣而加速，表明氧化作用發揮了影響 【1-s2.0-S0731708511003451-main】。其透過 HPLC 在 230 nm 下鑑定，保留時間為 4.5 分鐘 【1-s2.0-S0731708511003451-main】。
> - **苯丙胺酸 (Phenylalanine)：** 在含有 BPA 和甘露醇 (mannitol) 的冷凍乾燥藥品中形成 【1-s2.0-S0731708511003451-main】。這是一個緩慢且與溫度相關的過程；例如，在 40 °C 下儲存 6 個月後，產量約為 1%（相對於 BPA 質量） 【1-s2.0-S0731708511003451-main】。其透過 HPLC 在 256 nm 下鑑定，保留時間為 11.0 分鐘 【1-s2.0-S0731708511003451-main】。
> - **母體芳環 (Parent arene)：** 在鹼性水溶液條件下透過原脫硼反應 (protodeborylation) 形成 【1-s2.0-S0731708511003451-main】。
> 
> **雜質（合成中間體）：**
> - **(2-(4-Bromophenyl)-[1,3]dioxane (BrPD)：** BPA 的合成中間體，透過 HPLC 在 256 nm 下鑑定，保留時間為 17.3 分鐘 【1-s2.0-S0731708511003451-main】。
> - **4-Formylbenzeneboronic  …（完整內容見 JSON）

---
