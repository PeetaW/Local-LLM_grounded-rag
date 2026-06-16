# Eval Report — `nli_cuda`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-06-14 23:43
- 題數：2

## 彙總

| 指標 | 值 |
|------|-----|
| 平均 grounding 分數 | 0.512 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | N/A |
| 平均總延遲 | 2815.6s |
| 平均 planning 延遲 | 5.1s |
| 平均 retrieval 延遲 | 322.3s |
| 平均 grounding 延遲 | 715.5s |
| └ 其中 NLI | 8.6s |
| └ 其中 gemma4 | 687.4s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ✅ | Q01 | single_paper | 100.0% | N/A | 0.857 | 2232.6s | C0/U3 |
| ❌ | Q02 | single_paper | 100.0% | N/A | 0.167 | 3398.5s | C0/U9 |

## 逐題細節

### ✅ Q01 · single_paper

**問題**：In the study of JPH203's preincubation inhibitory effect on LAT1, how does preincubation change JPH203's inhibitory potency? Give the reported values.

- detected_paper：`None`
- 選出論文：['1-s2.0-S1347861320300633-main', '1-s2.0-S0021979710012403-main', '1-s2.0-S1936523321002047-main', 's41421-024-00697-6']
- gold_papers：['1-s2.0-S1347861320300633-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：N/A　grounding：0.857
- 延遲：2232.6s　問題標記：{'conflicts': 0, 'unsupported': 3}

**答案預覽**：

> ## 【論文直接依據】
> 在一項使用 LAT1 陽性人類結腸癌 HT-29 細胞的研究中，JPH203 對於 LAT1 功能的預孵育（preincubation）抑制效果特徵如下【1-s2.0-S1347861320300633-main】：
> 
> *   **抑制性質：** 預孵育對 LAT1 功能表現出濃度與時間依賴性的抑制效果【1-s2.0-S1347861320300633-main】。
> *   **濃度依賴性：** 當細胞預孵育 120 分鐘時，在 0.03, 0.1, 0.3, 和 30 mM 的濃度下，leucine 攝取量分別顯著下降至對照組的 63%, 51%, 42%, 和 25%（在 0.001 mM 時未觀察到顯著變化）。預孵育的 $\text{IC}_{50}$ 為 $193 \pm 50\text{ nM}$【1-s2.0-S1347861320300633-main】。
> *   **時間依賴性：** 使用 $10\text{ mM}$ JPH203，預孵育 30, 60, 和 120 分鐘分別將 leucine 攝取量降低至對照組的 43%, 32%, 和 27%【1-s2.0-S1347861320300633-main】。
> *   **協同效能增強：** 預孵育與共同孵育（co-incubation）結合時，可顯著增強抑制效能【1-s2.0-S1347861320300633-main】。報告數值如下：
>     *   **單獨共同孵育：** $\text{IC}_{50} = 99.2 \pm 11.0\text{ nM}$ 且 $K_i = 0.46\text{ mM}$【1-s2.0-S1347861320300633-main】。
>     *   **預孵育 + 共同孵育：** $\text{IC}_{50} = 34.2 \pm 3.6 …（完整內容見 JSON）

---

### ❌ Q02 · single_paper

**問題**：What is the hybrid process used for the asymmetric synthesis of optically pure L-p-boronophenylalanine, and what are its key steps?

- detected_paper：`None`
- 選出論文：['bbb0683', 'CMDC-20-e202500059']
- gold_papers：['bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：N/A　grounding：0.167
- 延遲：3398.5s　問題標記：{'conflicts': 0, 'unsupported': 9}

**答案預覽**：

> ## 【論文直接依據】
> 光學純 L-p-硼苯丙氨酸 (L-BPA) 不對稱合成的混合製程由「對映選擇性烷基化」隨後接「酶促水解」組成 [bbb0683]。關鍵步驟如下：
> 
> *   **保護步驟：** 將 4-溴甲基苯硼酸 (1) 與 2,2-二甲基-1,3-丙二醇在乾燥 THF 中於室溫下攪拌 10 分鐘。使用 EtOAc 和己烷重新結晶後，二羥基硼基被保護為環狀硼酸酯 (2)，產率為 79% [bbb0683]。
> *   **對映選擇性烷基化：** $(2R)-(-)$-2,5-二氫-2-異丙基-3,6-二甲氧基吡嗪 (bislactimether, 3) 在 $-78^\circ\text{C}$ 的 THF 中使用 n-BuLi 進行鋰化，然後與化合物 2 反應，並在氬氣環境下於 $-78^\circ\text{C}$ 攪拌 15 小時。這產生了具有 2S 構型的加成產物 4，產率為 41%，立體選擇性為 $72\%$ d.e. 或 $74\%$ e.e. [bbb0683]。
> *   **L-BPA 甲酯 (6) 的合成：** 這是一個兩階段過程：
>     1. 加成產物 4 在 $5^\circ\text{C}$ 和室溫下用 $0.1\text{ N HCl}$ 處理，並通過蒸餾除去 D-Val-OMe 以獲得粗產物 5 [bbb0683]。
>     2. 粗產物 5 在 THF 中與 $1\text{ N HCl}$ 攪拌 2 小時，並通過 ODS 柱色譜純化以獲得 L-BPA 甲酯 (6)。從化合物 4 到 6 的總產率為 85% [bbb0683]。
> *   **酶促水解：** 將 L-BPA 甲酯 (6) 與胰凝乳蛋白酶 (chymotrypsin) 溶解於水中。使用 $0.1\text{ N LiOH}$ 將 pH 調節至 5.0，並在維持 $\text …（完整內容見 JSON）

---
