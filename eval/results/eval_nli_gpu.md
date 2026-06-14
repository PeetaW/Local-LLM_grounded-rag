# Eval Report — `nli_gpu`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-06-14 03:27
- 題數：3

## 彙總

| 指標 | 值 |
|------|-----|
| 平均 grounding 分數 | 0.294 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | N/A |
| 平均總延遲 | 12951.1s |
| 平均 planning 延遲 | 4.9s |
| 平均 retrieval 延遲 | 287.1s |
| 平均 grounding 延遲 | 10572.0s |
| └ 其中 NLI | 9152.2s |
| └ 其中 gemma4 | 1404.9s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ⚠️ | Q01 | single_paper | 100.0% | N/A | 0.667 | 13515.1s | C0/U5 |
| ❌ | Q02 | single_paper | 100.0% | N/A | 0.0 | 8403.2s | C0/U7 |
| ❌ | Q03 | figure_dependent | 100.0% | N/A | 0.214 | 16935.1s | C0/U16 |

## 逐題細節

### ⚠️ Q01 · single_paper

**問題**：In the study of JPH203's preincubation inhibitory effect on LAT1, how does preincubation change JPH203's inhibitory potency? Give the reported values.

- detected_paper：`None`
- 選出論文：['1-s2.0-S1347861320300633-main', '1-s2.0-S0021979710012403-main', '1-s2.0-S1936523321002047-main', 's41421-024-00697-6']
- gold_papers：['1-s2.0-S1347861320300633-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：N/A　grounding：0.667
- 延遲：13515.1s　問題標記：{'conflicts': 0, 'unsupported': 5}

**答案預覽**：

> ## 【論文直接依據】
> * 預孵育（Preincubation）在與共同孵育（co-incubation）結合時，能顯著增強 JPH203 的抑制效力 【1-s2.0-S1347861320300633-main】。
> * 報告的 $\text{IC}_{50}$ 值如下：單獨共同孵育為 $99.2 \pm 11.0\text{ nM}$，而加入預孵育後則降低至 $34.2 \pm 3.6\text{ nM}$ 【1-s2.0-S1347861320300633-main】。
> * 報告的 $\text{K}_i$ 值如下：單獨共同孵育為 $0.46\text{ mM}$，而加入預孵育後則降低至 $0.37\text{ mM}$ 【1-s2.0-S1347861320300633-main】。
> * 預孵育與共同孵育之間的協同作用導致 $\text{IC}_{50}$ 值降低三倍，且 $\text{K}_i$ 值降低 20% 【1-s2.0-S1347861320300633-main】。
> * 單獨預孵育在 HT-29 細胞中（於 120 分鐘時）表現出濃度依賴性的抑制作用，其 $\text{IC}_{50}$ 為 $193 \pm 50\text{ nM}$ 【1-s2.0-S1347861320300633-main】。
> * 單獨預孵育表現出時間依賴性的抑制作用（使用 $10\text{ mM}$ JPH203），其中亮氨酸（leucine）的攝取活性在 30、60 和 120 分鐘後分別降低至 43%、32% 和 27% 【1-s2.0-S1347861320300633-main】。
> 
> ## 【跨文獻推論】
> * 跨論文推論（基於 【1-s2.0-S1347861320300633-main】）：預孵育起到了啟動機制的作用，顯著增加了 LAT1 轉運蛋白對 JPH203 的 …（完整內容見 JSON）

---

### ❌ Q02 · single_paper

**問題**：What is the hybrid process used for the asymmetric synthesis of optically pure L-p-boronophenylalanine, and what are its key steps?

- detected_paper：`None`
- 選出論文：['bbb0683', 'CMDC-20-e202500059']
- gold_papers：['bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：N/A　grounding：0.0
- 延遲：8403.2s　問題標記：{'conflicts': 0, 'unsupported': 7}

**答案預覽**：

> ## 【論文直接依據】
> 光學純 L-p-硼苯丙氨酸 (L-BPA) 的不對稱合成雜化製程結合了對映選擇性烷基化與隨後的酶促水解 [bbb0683]。與先前的方法相比，此方法具有反應步驟較少、操作與後處理更簡單以及光學純度高等優勢 [bbb0683]。該製程使用了市售的手性輔助劑 (2R)-(-)-2,5-dihydro-2-isopropyl-3,6-dimethoxypyrazine [bbb0683]。
> 
> 合成的關鍵步驟如下：
> 1.  **保護基合成**：使用 2,2-dimethyl-1,3-propanediol 對 4-bromomethylphenylboronic acid (1) 進行保護，以形成環狀硼酸酯 (2)，產率為 79% [bbb0683]。具体而言，將 5.99 g (27.9 mmol) 的起始原料與 2.91 g (27.9 mmol) 的 2,2-dimethyl-1,3-propanediol 在 30 ml 的乾燥 THF 中於室溫下攪拌 10 分鐘 [bbb0683]。
> 2.  **對映選擇性烷基化**：在氬氣氛圍下，於 -78°C 的乾燥 THF 中，使鋰化的 (2R)-(-)-2,5-dihydro-2-isopropyl-3,6-dimethoxypyrazine (3) 與溴化物 (2) 反應，生成具有 2S 構型的加成物 (4)，其立體選擇性為 74% e.e. [bbb0683]。操作過程包括將 1.84 g (10.0 mmol) 的化合物 3 溶解於 36 ml 乾燥 THF 中，在 30 分鐘內加入 5.95 ml n-BuLi 並攪拌 1 小時，隨後在 30 分鐘內加入由 2.83 g (10.0 mmol) 化合物 2 溶解於 6 ml 乾燥 THF 的溶液中並攪拌 15 小時 [bbb0683]。反應物以 pH …（完整內容見 JSON）

---

### ❌ Q03 · figure_dependent

**問題**：According to the cryo-EM structure, how does JPH203 bind to and inhibit the LAT1-4F2hc complex?

- detected_paper：`None`
- 選出論文：['s41421-024-00697-6', '1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 's41467-024-45464-z (1)', 'LAT1 ChemComm 2026']
- gold_papers：['s41421-024-00697-6']
- 論文選擇命中率：100.0%　檢索覆蓋率：N/A　grounding：0.214
- 延遲：16935.1s　問題標記：{'conflicts': 0, 'unsupported': 16}

**答案預覽**：

> 根據提供的事實，以下是關於 JPH203 如何結合並抑制 LAT1-4F2hc 複合物的全面解答。
> 
> ## 【論文直接依據】
> *   **結合位點與結構：** JPH203 結合於 LAT1 的傳統底物結合口袋中 [Fact 1]。其分子結構由親水性頭部（苯丙氨酸骨架）和疏水性尾部（5-amino-2-(3-aminophenyl) benzoxazole 骨架）組成 [Fact 2]。
> *   **結合相互作用（親水性頭部）：** 
>     *   頭部的 $\alpha$-氨基和 $\alpha$-羧基與 TM1 和 TM6 解旋區域的主鏈原子形成氫鍵網絡 [Fact 3]。
>     *   頭部二鹵酪氨酸（dihalo-tyrosine）核心上的氯原子與 Tyr259 形成鹵鍵 [Fact 4]。
> *   **結合相互作用（疏水性尾部）：** 
>     *   尾部嵌入由 TM1 (Ile63, Ile64, Ile68)、TM3 (Ile140, Ile147, Ile148)、TM6 (Phe252, Gly255) 和 TM10 (Ile397, Phe400, Phe403) 組成的疏水性口袋中 [Fact 5]。
>     *   尾部的氨基與 TM10 上 Phe400 的主鏈形成氫鍵 [Fact 6]。
>     *   尾部的苯環與 Phe252 的側鏈發生 T 型 $\pi$-$\pi$ 相互作用 [Fact 7]。
> *   **結構影響與抑制：** 
>     *   JPH203 將 LAT1-4F2hc 複合物鎖定在朝外（outward-facing）的結構中（解析度為 3.30 Å，經重點精修後提升至 3.25 Å）[Fact 8, Fact 9]。
>     *   JPH203 的延伸尾部阻止了 Phe400 的移動 [Fact 10]。
>     …（完整內容見 JSON）

---
