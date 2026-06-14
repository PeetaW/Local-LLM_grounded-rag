# Eval Report — `nli_batched`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-06-14 20:45
- 題數：3

## 彙總

| 指標 | 值 |
|------|-----|
| 平均 grounding 分數 | 0.45 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | N/A |
| 平均總延遲 | 15462.4s |
| 平均 planning 延遲 | 5.0s |
| 平均 retrieval 延遲 | 285.1s |
| 平均 grounding 延遲 | 13634.5s |
| └ 其中 NLI | 12506.4s |
| └ 其中 gemma4 | 1112.9s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ⚠️ | Q01 | single_paper | 100.0% | N/A | 0.714 | 13247.6s | C0/U5 |
| ❌ | Q02 | single_paper | 100.0% | N/A | 0.0 | 7628.0s | C0/U6 |
| ⚠️ | Q03 | figure_dependent | 100.0% | N/A | 0.636 | 25511.7s | C0/U7 |

## 逐題細節

### ⚠️ Q01 · single_paper

**問題**：In the study of JPH203's preincubation inhibitory effect on LAT1, how does preincubation change JPH203's inhibitory potency? Give the reported values.

- detected_paper：`None`
- 選出論文：['1-s2.0-S1347861320300633-main', '1-s2.0-S0021979710012403-main', '1-s2.0-S1936523321002047-main', 's41421-024-00697-6']
- gold_papers：['1-s2.0-S1347861320300633-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：N/A　grounding：0.714
- 延遲：13247.6s　問題標記：{'conflicts': 0, 'unsupported': 5}

**答案預覽**：

> ## 【論文直接依據】
> * 預孵育（Preincubation）在與共同孵育（co-incubation）結合時，能顯著增強 JPH203 的抑制效力 【1-s2.0-S1347861320300633-main】。
> * 報告的 $\text{IC}_{50}$ 值如下：
>     * 僅共同孵育：$99.2 \pm 11.0\text{ nM}$ 【1-s2.0-S1347861320300633-main】。
>     * 預孵育 + 共同孵育：$34.2 \pm 3.6\text{ nM}$ 【1-s2.0-S1347861320300633-main】。
> * 報告的 $\text{K}_i$ 值如下：
>     * 僅共同孵育：$0.46\text{ mM}$ 【1-s2.0-S1347861320300633-main】。
>     * 預孵育 + 共同孵育：$0.37\text{ mM}$ 【1-s2.0-S1347861320300633-main】。
> * 預孵育與共同孵育的協同效應導致 $\text{IC}_{50}$ 值降低三倍，且 $\text{K}_i$ 值降低 20% 【1-s2.0-S1347861320300633-main】。
> * 僅預孵育表現出對 LAT1 功能的濃度與時間依賴性抑制，其 $\text{IC}_{50}$ 值為 $193 \pm 50\text{ nM}$ 【1-s2.0-S1347861320300633-main】。
> 
> ## 【跨文獻推論】
> * 跨論文推論（基於 【1-s2.0-S1347861320300633-main】）：雖然僅預孵育的效力低於共同孵育（比較預孵育的 $\text{IC}_{50}$ 為 $193 \pm 50\text{ nM}$ 與共同孵育的 $99.2 \pm 11.0\text{ nM}$），但其主要 …（完整內容見 JSON）

---

### ❌ Q02 · single_paper

**問題**：What is the hybrid process used for the asymmetric synthesis of optically pure L-p-boronophenylalanine, and what are its key steps?

- detected_paper：`None`
- 選出論文：['bbb0683', 'CMDC-20-e202500059']
- gold_papers：['bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：N/A　grounding：0.0
- 延遲：7628.0s　問題標記：{'conflicts': 0, 'unsupported': 6}

**答案預覽**：

> ## 【論文直接依據】
> 用於光學純 L-p-硼苯丙氨酸 (L-BPA) 不對稱合成的混合製程，結合了「對映選擇性烷基化」與隨後的「酵素水解」【bbb0683】。與先前報導的方法相比，此製程的特點是反應步驟較少，操作與後處理更簡單【bbb0683】。
> 
> 該混合製程的關鍵步驟如下：
> 1.  **保護反應**：將 4-溴甲基苯硼酸 (1) 與 2,2-二甲基-1,3-丙二醇在乾燥 THF 中於室溫下攪拌 10 分鐘，以產生產率為 79% 的環狀硼酸酯 (2)【bbb0683】。
> 2.  **對映選擇性烷基化**：在 -78°C 的乾燥 THF 與氬氣環境中，使用 n-BuLi（1.68M 於己烷中）將 (2R)-(-)-2,5-二氫-2-異丙基-3,6-二甲氧基吡嗪 (3) 鋰化，隨後加入溴化物 2。攪拌 15 小時後，生成具有 2S 配置的加成產物 4，實現了 74% e.e. 的立體選擇性【bbb0683】。
> 3.  **酸處理與分離**：在 5°C 下加入 0.1 N 鹽酸 (HCl)，並於室溫下攪拌 8 小時，將產物 4 轉化為 D-缬氨酸甲酯 (D-Val-OMe) 與氨基酸酯 5 的等莫量混合物。透過蒸餾除去 D-Val-OMe 以獲得粗產物 5，並使用 3% 氨水將 pH 值調節至 8.5【bbb0683】。
> 4.  **去保護反應**：將粗產物 5 與 1 N HCl 及 THF 在室溫下攪拌 2 小時，並透過 ODS 管柱層析法（使用水作為洗脫劑）純化，以合成 L-BPA 甲酯 (6)，相對於產物 4 的產率為 85%【bbb0683】。
> 5.  **酵素水解**：使用糜蛋白酶 (Chymotrypsin) 作為水中的催化劑，並使用 0.1 N 氫氧化鋰 (LiOH) 將 pH 值調節至 5.0。在 25°C 下攪拌 30 小時後，獲得產率為 79% 且光學純度 …（完整內容見 JSON）

---

### ⚠️ Q03 · figure_dependent

**問題**：According to the cryo-EM structure, how does JPH203 bind to and inhibit the LAT1-4F2hc complex?

- detected_paper：`None`
- 選出論文：['s41421-024-00697-6', '1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 's41467-024-45464-z (1)', 'LAT1 ChemComm 2026']
- gold_papers：['s41421-024-00697-6']
- 論文選擇命中率：100.0%　檢索覆蓋率：N/A　grounding：0.636
- 延遲：25511.7s　問題標記：{'conflicts': 0, 'unsupported': 7}

**答案預覽**：

> 根據提供的事實，以下是關於 JPH203 如何結合並抑制 LAT1-4F2hc 複合物的全面解答。
> 
> ## 【論文直接依據】
> *   **結合位點與結構：** JPH203 在向外開放（outward-facing）的結構中，結合於 LAT1 的傳統底物結合口袋內 【s41421-024-00697-6】。低溫電子顯微鏡（cryo-EM）的分辨率為 3.30 Å，透過聚焦精修（focused refinement），JPH203 的密度圖分辨率提升至 3.25 Å 【s41421-024-00697-6】。
> *   **分子組成：** 該分子由親水性頭部（苯丙氨酸骨架）和疏水性尾部（5-amino-2-(3-aminophenyl) benzoxazole 骨架）組成 【s41421-024-00697-6】。
> *   **頭部交互作用：** 
>     *   頭部的 $\alpha$-氨基和 $\alpha$-羧基與 TM1 和 TM6 未捲曲/未折疊區域的主鏈原子形成氫鍵網絡 【s41421-024-00697-6】。
>     *   頭部雙鹵代酪氨酸核心上的氯原子與 Tyr259 形成鹵鍵（halogen bond） 【s41421-024-00697-6】。
> *   **尾部交互作用：** 
>     *   疏水性尾部嵌入由 TM1 (Ile63, Ile64, Ile68)、TM3 (Ile140, Ile147, Ile148)、TM6 (Phe252, Gly255) 和 TM10 (Ile397, Phe400, Phe403) 組成的口袋中 【s41421-024-00697-6】。
>     *   尾部的氨基與 TM10 上 Phe400 的主鏈形成氫鍵 【s41421-024-00697-6】。
>     *   尾部的苯環與 Phe252 的側鏈發生  …（完整內容見 JSON）

---
