# Eval Report — `rerank_off`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-06-12 15:35
- 題數：4

## 彙總

| 指標 | 值 |
|------|-----|
| 平均 grounding 分數 | 0.651 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | N/A |
| 平均總延遲 | 8131.3s |
| 平均 planning 延遲 | 4.7s |
| 平均 retrieval 延遲 | 251.2s |
| 平均 grounding 延遲 | 6177.5s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ✅ | Q01 | single_paper | 100.0% | N/A | 0.857 | 5896.3s | C0/U3 |
| ❌ | Q02 | single_paper | 100.0% | N/A | 0.286 | 10163.4s | C0/U9 |
| ✅ | Q03 | figure_dependent | 100.0% | N/A | 0.889 | 8637.9s | C0/U3 |
| ⚠️ | Q04 | table_lookup | 100.0% | N/A | 0.571 | 7827.5s | C0/U5 |

## 逐題細節

### ✅ Q01 · single_paper

**問題**：In the study of JPH203's preincubation inhibitory effect on LAT1, how does preincubation change JPH203's inhibitory potency? Give the reported values.

- detected_paper：`None`
- 選出論文：['1-s2.0-S1347861320300633-main', '1-s2.0-S0021979710012403-main', '1-s2.0-S1936523321002047-main', 's41421-024-00697-6']
- gold_papers：['1-s2.0-S1347861320300633-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：N/A　grounding：0.857
- 延遲：5896.3s　問題標記：{'conflicts': 0, 'unsupported': 3}

**答案預覽**：

> ## 【論文直接依據】
> * 預孵育（Preincubation）在與共同孵育（co-incubation）結合時，能顯著增強 JPH203 的抑制效力，產生協同效應 [Chunk 1, Chunk 3]。
> * 報告的 $\text{IC}_{50}$ 值如下：
>     * 僅共同孵育：$99.2 \pm 11.0\text{ nM}$ [Chunk 1, Chunk 3]。
>     * 預孵育結合共同孵育：$34.2 \pm 3.6\text{ nM}$ [Chunk 1, Chunk 3]。
>     * 僅預孵育：$193 \pm 50\text{ nM}$ [Chunk 3]。
> * 報告的 $\text{K}_i$ 值如下：
>     * 僅共同孵育：$0.46\text{ mM}$ [Chunk 1, Chunk 3]。
>     * 預孵育加共同孵育：$0.37\text{ mM}$ [Chunk 1, Chunk 3]。
> * 預孵育與共同孵育之間的這種協同效應導致 $\text{IC}_{50}$ 值降低三倍，且 $\text{K}_i$ 值降低 20% [Chunk 1]。
> 
> ## 【跨文獻推論】
> * 跨論文推論（基於 [Chunk 1, Chunk 3]）：雖然僅預孵育的效力低於僅共同孵育（比較 $\text{IC}_{50}$ 值 $193 \pm 50\text{ nM}$ 與 $99.2 \pm 11.0\text{ nM}$），但這兩個過程的結合所產生的效力超過了任何一種單獨方法（$\text{IC}_{50}$ 為 $34.2 \pm 3.6\text{ nM}$）。這表明預孵育以某種方式對系統或轉運蛋白進行了啟動（primes），從而顯著優化了隨後共同孵育期間的抑制作用。
> 
> ## 【知識延伸與推測】
> * 模型推測（文獻依據不足）：觀察到預孵育後效力增加 …（完整內容見 JSON）

---

### ❌ Q02 · single_paper

**問題**：What is the hybrid process used for the asymmetric synthesis of optically pure L-p-boronophenylalanine, and what are its key steps?

- detected_paper：`None`
- 選出論文：['bbb0683', 'CMDC-20-e202500059']
- gold_papers：['bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：N/A　grounding：0.286
- 延遲：10163.4s　問題標記：{'conflicts': 0, 'unsupported': 9}

**答案預覽**：

> ## 【論文直接依據】
> 
> 光學純 L-p-硼苯丙氨酸 (L-BPA) 不對稱合成的混合製程結合了兩個主要階段：「不對稱烷基化」與「酶促水解」【bbb0683】。與使用手性膦的不對稱氫化或外消旋前體的酶促拆分相比，該方法的特點是效率更高、反應步驟更少，且操作與後處理更簡單【bbb0683】。
> 
> 該製程的關鍵步驟如下：
> 
> *   **步驟 1：起始物料保護：** 4-溴甲基苯硼酸 (1) 在乾燥 THF 中於室溫下與 2,2-二甲基-1,3-丙二醇反應 10 分鐘，生成環狀硼酸酯 (化合物 2)，產率為 79% 【bbb0683】。具體而言，在 30 ml 乾燥 THF 中使用 5.99 g (27.9 mmol) 的物質 1 和 2.91 g (27.9 mmol) 的 2,2-二甲基-1,3-丙二醇【bbb0683】。
> *   **步驟 2：不對稱烷基化：** 在氬氣環境下的乾燥 THF 中，於 -78°C 將鋰化的 (2R)-(-)-2,5-二氫-2-異丙基-3,6-二甲氧基吡嗪 (3) 與溴化物 2 反應，生成具有 2S 構型的加成產物 4。立體選擇性為 74% e.e.（或 72% 非對映異構體過量） 【bbb0683】。此步驟使用 n-BuLi（1.68M 於己烷中）作為鹼/催化劑，後處理涉及磷酸鹽緩衝液 (pH 7.0)、乙醚、飽和食鹽水和硫酸鈉【bbb0683】。詳細步驟包括將 1.84 g (10.0 mmol) 的 3 溶解於 36 ml 乾燥 THF 中，在 30 分鐘內加入 5.95 ml (10 mmol) 的 n-BuLi 並攪拌 1 小時，然後加入由 2.83 g (10.0 mmol) 化合物 2 溶解於 6 ml 乾燥 THF 的溶液中，並在 -78°C 下攪拌 15 小時【bbb0683】。
> *   **步驟 3：L-BPA 甲酯 (6) 的 …（完整內容見 JSON）

---

### ✅ Q03 · figure_dependent

**問題**：According to the cryo-EM structure, how does JPH203 bind to and inhibit the LAT1-4F2hc complex?

- detected_paper：`None`
- 選出論文：['s41421-024-00697-6', '1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 's41467-024-45464-z (1)', 'LAT1 ChemComm 2026']
- gold_papers：['s41421-024-00697-6']
- 論文選擇命中率：100.0%　檢索覆蓋率：N/A　grounding：0.889
- 延遲：8637.9s　問題標記：{'conflicts': 0, 'unsupported': 3}

**答案預覽**：

> 根據提供的文本，以下是關於 JPH203 如何結合並抑制 LAT1-4F2hc 複合物的全面回答。
> 
> ## 【論文直接依據】
> *   **結合位點與分子結構：** JPH203 結合於 LAT1 的傳統底物結合口袋內 【s41421-024-00697-6】。其結構由親水性頭部基團（苯丙氨酸骨架）和疏水性尾部基團（5-amino-2-(3-aminophenyl) benzoxazole 骨架）組成 【s41421-024-00697-6】。
> *   **結合交互作用：**
>     *   **親水性頭部：** $\alpha$-氨基和 $\alpha$-羧基與 TM1 和 TM6 去螺旋區域的主鏈原子形成氫鍵網絡 【s41421-024-00697-6】。
>     *   **核心：** 二鹵代酪氨酸（tyrosine）核心上的氯原子與 Tyr259 形成鹵鍵 【s41421-024-00697-6】。
>     *   **疏水性尾部：** 尾部嵌入由 TM1 (Ile63, Ile64, Ile68)、TM3 (Ile140, Ile147, Ile148)、TM6 (Phe252, Gly255) 和 TM10 (Ile397, Phe400, Phe403) 的殘基組成的疏水口袋中 【s41421-024-00697-6】。具體而言，尾部的氨基與 Phe400 (TM10) 的主鏈形成氫鍵，且其苯環與 Phe252 的側鏈發生 T 型 $\pi$-$\pi$ 交互作用 【s41421-024-00697-6】。
> *   **結構影響與抑制機制：** 
>     *   JPH203 將 LAT1-4F2hc 複合物穩定在面向外（outward-facing）的構象中（解析度為 3.30 Å，JPH203 的聚焦精修解析度為 3.25 Å） 【s41421-024-00 …（完整內容見 JSON）

---

### ⚠️ Q04 · table_lookup

**問題**：What are the main degradation products and impurities of BPA identified by HPLC, and under which storage conditions do they form?

- detected_paper：`None`
- 選出論文：['1-s2.0-S0731708511003451-main', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water']
- gold_papers：['1-s2.0-S0731708511003451-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：N/A　grounding：0.571
- 延遲：7827.5s　問題標記：{'conflicts': 0, 'unsupported': 5}

**答案預覽**：

> ## 【論文直接依據】
> 
> **降解產物：**
> * **酪胺酸 (Tyrosine)**：在鹼性和氧化條件下形成，且在氧化條件下的降解速度非常快 [Chunk 1, Chunk 2]。具體的形成條件包括在 $55^\circ\text{C}$ 的 $100\text{ mM NaOH}$ 中孵育 24 小時，或暴露於 $6\text{ mM H}_2\text{O}_2$ [Chunk 2]。其透過 HPLC 鑑定，保留時間為 4.5 分鐘，並在 230 nm 處進行定量 [Chunk 1, Chunk 2]。
> * **苯丙胺酸 (Phenylalanine)**：形成於含有 BPA 和甘露醇 (mannitol) 的冷凍乾燥藥品中 [Chunk 1]。具體而言，在含有 $100\text{ mg/ml}$ BPA 和 $110\text{ mg/ml}$ 甘露醇且 pH 為 $8.0 \pm 0.1$ 的樣本中，於 $40^\circ\text{C}$ 下儲存 6 個月會產生約 1% 的苯丙胺酸（相對於 BPA 質量）[Chunk 1, Chunk 2]。其透過 HPLC 鑑定，保留時間為 11.0 分鐘，並在 256 nm 處進行定量 [Chunk 1]。
> * **母體芳環 (Parent arene)**：在水相鹼性條件下，透過原脫硼反應 (protodeborylation) 形成，這是最常見的結果 [Chunk 2]。
> 
> **雜質：**
> * **BrPD (2-(4-Bromophenyl)-[1,3]dioxane)**：一種合成雜質，檢出限為 $0.5\ \mu\text{g/ml}$（或標稱工作濃度 NWC 的 0.1%），透過 HPLC 鑑定其保留時間為 17.3 分鐘，並在 256 nm 處檢測 [Chunk 1]。
> * **FBBA (4-Formylb …（完整內容見 JSON）

---
