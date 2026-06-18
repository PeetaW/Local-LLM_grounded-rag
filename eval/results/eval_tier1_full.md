# Eval Report — `tier1_full`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-06-17 09:47
- 題數：12

## 彙總

| 指標 | 值 |
|------|-----|
| 平均 grounding 分數 | 0.514 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 97.5% |
| 平均總延遲 | 2593.8s |
| 平均 planning 延遲 | 6.9s |
| 平均 retrieval 延遲 | 350.0s |
| 平均 grounding 延遲 | 872.5s |
| └ 其中 NLI | 12.9s |
| └ 其中 gemma4 | 842.9s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ⚠️ | Q01 | single_paper | 100.0% | 100.0% | 0.6 | 2353.0s | C0/U4 |
| ❌ | Q02 | single_paper | 100.0% | 100.0% | 0.25 | 3276.4s | C0/U7 |
| ✅ | Q03 | figure_dependent | 100.0% | 100.0% | 0.818 | 2374.0s | C0/U5 |
| ⚠️ | Q04 | table_lookup | 100.0% | 100.0% | 0.75 | 2488.3s | C0/U5 |
| ❌ | Q05 | single_paper | 100.0% | 100.0% | 0.25 | 3053.3s | C0/U11 |
| ✅ | Q06 | multi_chunk | 100.0% | 75.0% | 0.833 | 1788.6s | C0/U3 |
| ❌ | Q07 | figure_dependent | 100.0% | 100.0% | 0.167 | 2892.5s | C0/U9 |
| ⚠️ | Q08 | cross_paper | 100.0% | 100.0% | 0.5 | 3817.4s | C0/U11 |
| ✅ | Q09 | cross_paper | 100.0% | 100.0% | 0.8 | 2953.9s | C0/U7 |
| ⚠️ | Q10 | contradiction_check | 100.0% | 100.0% | 0.688 | 3558.9s | C0/U9 |
| ✅ | Q11 | out_of_scope | N/A | N/A | — | 359.0s | C0/U0 |
| ❌ | Q12 | false_premise | N/A | N/A | 0.0 | 2209.7s | C0/U3 |

## 逐題細節

### ⚠️ Q01 · single_paper

**問題**：In the study of JPH203's preincubation inhibitory effect on LAT1, how does preincubation change JPH203's inhibitory potency? Give the reported values.

- detected_paper：`None`
- 選出論文：['1-s2.0-S1347861320300633-main', '1-s2.0-S0021979710012403-main', '1-s2.0-S1936523321002047-main', 's41421-024-00697-6']
- gold_papers：['1-s2.0-S1347861320300633-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.6
- 延遲：2353.0s　問題標記：{'conflicts': 0, 'unsupported': 4}

**答案預覽**：

> ## 【論文直接依據】
> * 預孵育（Preincubation）在與共同孵育（co-incubation）結合時，能顯著增強 JPH203 的抑制效力 [1-s2.0-S1347861320300633-main]。
> * 報告的抑制效力數值如下：
>     * **$\text{IC}_{50}$ 值：** 在單獨的共同孵育抑制分析中，$\text{IC}_{50}$ 為 $99.2 \pm 11.0\text{ nM}$。當加入預孵育（預孵育加上共同孵育）時，$\text{IC}_{50}$ 降低至 $34.2 \pm 3.6\text{ nM}$ [1-s2.0-S1347861320300633-main]。
>     * **$\text{K}_i$ 值：** 在共同孵育分析中，$\text{K}_i$ 為 $0.46\text{ mM}$，而在預孵育加上共同孵育的分析中則降低至 $0.37\text{ mM}$ [1-s2.0-S1347861320300633-main]。
> * 預孵育與共同孵育之間的協同作用導致 $\text{IC}_{50}$ 下降三倍，且 $\text{K}_i$ 降低了 20% [1-s2.0-S1347861320300633-main]。
> * 單獨預孵育同樣表現出抑制效果，且該效果具有濃度依賴性（$\text{IC}_{50}$ 為 $193 \pm 50\text{ nM}$）與時間依賴性 [1-s2.0-S1347861320300633-main]。
> 
> ## 【跨文獻推論】
> * 跨論文推論（基於 [1-s2.0-S1347861320300633-main]）：雖然單獨預孵育的效力低於單獨共同孵育（$\text{IC}_{50}$ 分別為 $193\text{ nM}$ 與 $99.2\text{ nM}$），但兩者的結合產生了協同 …（完整內容見 JSON）

---

### ❌ Q02 · single_paper

**問題**：What is the hybrid process used for the asymmetric synthesis of optically pure L-p-boronophenylalanine, and what are its key steps?

- detected_paper：`None`
- 選出論文：['bbb0683', 'CMDC-20-e202500059']
- gold_papers：['bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.25
- 延遲：3276.4s　問題標記：{'conflicts': 0, 'unsupported': 7}

**答案預覽**：

> ## 【論文直接依據】
> 光學純 L-p-硼苯丙氨酸 (L-BPA) 的不對稱合成雜化製程結合了對映選擇性烷基化與隨後的酶促水解，起始於一種市售的手性輔助劑 【bbb0683】。與先前報導的方法相比，該製程的特點是操作更簡單、後處理更容易且反應步驟較少 【bbb0683】。
> 
> 合成的關鍵步驟如下：
> *   **步驟 1：硼酸的保護。** 將 4-bromomethylphenylboronic acid (1) 與 2,2-dimethyl-1,3-propanediol 在乾燥 THF 中於室溫下攪拌 10 分鐘，將二羥基硼基保護為環狀硼酸酯 (2)，產率為 79% 【bbb0683】。
> *   **步驟 2：對映選擇性烷基化。** 使用 n-BuLi (1.68M in hexane) 對 (2R)-(-)-2,5-dihydro-2-isopropyl-3,6-dimethoxypyrazine (3) 進行鋰化。隨後在氬氣氛圍下，於 -78°C 將 Bromide 2 加入乾燥 THF 中，以產生具有 2S 構型的加成產物 4，其立體選擇性為 74% e.e. 【bbb0683】。產物 4 隨後使用 pH 7.0 的磷酸鹽緩衝液處理 【bbb0683】。
> *   **步驟 3：酸處理與去保護。** 此步驟分為兩個階段：
>     1.  產物 4 與 THF 中的 0.1 N HCl 反應（於 5°C 加入，隨後在室溫下攪拌 8 小時），產生 D-valine methyl ester (D-Val-OMe) 與氨基酸酯 5 的等量混合物。透過蒸餾除去 D-Val-OMe，並使用 3% ammonia 將 pH 調節至 8.5 【bbb0683】。
>     2.  粗產物 5 與 1 N HCl 及 THF 在室溫下攪拌 2 小時，以獲得 L-BPA methyl es …（完整內容見 JSON）

---

### ✅ Q03 · figure_dependent

**問題**：According to the cryo-EM structure, how does JPH203 bind to and inhibit the LAT1-4F2hc complex?

- detected_paper：`None`
- 選出論文：['s41421-024-00697-6', '1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 's41467-024-45464-z (1)', 'LAT1 ChemComm 2026']
- gold_papers：['s41421-024-00697-6']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.818
- 延遲：2374.0s　問題標記：{'conflicts': 0, 'unsupported': 5}

**答案預覽**：

> ## 【論文直接依據】
> * **結合位點與結構**：JPH203 結合於 LAT1 的傳統底物結合口袋中 【s41421-024-00697-6】。其分子結構由親水性頭部（苯丙氨酸骨架）和疏水性尾部（5-amino-2-(3-aminophenyl) benzoxazole 骨架）組成 【s41421-024-00697-6】。
> * **結合相互作用**：
>     * **親水性頭部**：$\alpha$-氨基和 $\alpha$-羧基與 TM1 和 TM6 在其去捲曲區域（uncoiled regions）的主鏈原子形成氫鍵網絡 【s41421-024-00697-6】。
>     * **核心**：雙鹵代酪氨酸（double-halogenated tyrosine）核心上的氯原子與 Tyr259 形成鹵鍵 【s41421-024-00697-6】。
>     * **疏水性尾部**：尾部嵌入由 TM1 (Ile63, Ile64, Ile68)、TM3 (Ile140, Ile147, Ile148)、TM6 (Phe252, Gly255) 和 TM10 (Ile397, Phe400, Phe403) 的殘基組成的疏水口袋中 【s41421-024-00697-6】。具體而言，尾部的氨基與 TM10 上 Phe400 的主鏈形成氫鍵，且其苯環與 Phe252 的側鏈發生 T 型 $\pi$-$\pi$ 相互作用 【s41421-024-00697-6】。
> * **結構影響與抑制**： 
>     * JPH203 將 LAT1-4F2hc 複合物穩定在面向外（outward-facing）的構象中（解析度為 $3.30\ \text{\AA}$，其中 JPH203 區域精修至 $3.25\ \text{\AA}$） 【s41421-024-00697-6】。
>     * …（完整內容見 JSON）

---

### ⚠️ Q04 · table_lookup

**問題**：What are the main degradation products and impurities of BPA identified by HPLC, and under which storage conditions do they form?

- detected_paper：`None`
- 選出論文：['1-s2.0-S0731708511003451-main', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water']
- gold_papers：['1-s2.0-S0731708511003451-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.75
- 延遲：2488.3s　問題標記：{'conflicts': 0, 'unsupported': 5}

**答案預覽**：

> ## 【論文直接依據】
> 
> **降解產物及其形成條件：**
> * **酪胺酸 (Tyrosine)：** 在鹼性和氧化條件下形成；具體而言，在 $6\text{ mM H}_2\text{O}_2$ 氧化下降解速度非常快，且在 $100\text{ mM NaOH}$ 中於 $55^\circ\text{C}$ 下孵育 24 小時後可觀察到 [Chunk 1, Chunk 2]。
> * **苯丙胺酸 (Phenylalanine)：** 在 BPA/甘露醇 (mannitol) 凍乾藥物（BPA $100\text{ mg/ml}$，甘露醇 $110\text{ mg/ml}$，pH $8.0 \pm 0.1$）於 $40^\circ\text{C}$ 下儲存 6 個月時形成；此過程緩慢且具有溫度依賴性 [Chunk 1, Chunk 2]。
> * **母體芳環 (Parent arene)：** 在水相鹼性條件下透過原脫硼反應 (protodeborylation) 形成，這是此類情況下最常見的結果 [Chunk 2]。
> 
> **鑑定出的雜質：**
> * **BrPD (2-(4-Bromophenyl)-[1,3]dioxane)** [Chunk 1]。
> * **FBBA (4-Formylbenzeneboronic acid)** [Chunk 1]。
> * **BDPA (2-tert-Butoxycarbonylamino-3-[4-(5,5-dimethyl-[1,3,2]dioxaborinan-2-yl)-phenyl]acrylic acid methyl ester)** [Chunk 1]。
> 
> **HPLC 鑑定參數：**
> * **酪胺酸 (Tyrosine)：** 保留時間 (RT) 為 4.5 分鐘，在 $230\text{ nm}$ 處定量 [Chunk  …（完整內容見 JSON）

---

### ❌ Q05 · single_paper

**問題**：What is the solvent-free green protocol for N-Boc protection of amines using picric acid as a catalyst, and what reaction conditions are used?

- detected_paper：`None`
- 選出論文：['1-s2.0-S2773223124000268-main']
- gold_papers：['1-s2.0-S2773223124000268-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.25
- 延遲：3053.3s　問題標記：{'conflicts': 0, 'unsupported': 11}

**答案預覽**：

> ## 【論文直接依據】
> - **催化劑與試劑**：該方案使用苦味酸 (picric acid) 作為 Brønsted 酸催化劑，並使用二叔丁基二碳酸酯 ($(Boc)_2O$) 作為保護試劑 【1-s2.0-S2773223124000268-main】。
> - **最適反應條件**：反應在室溫 (rt) 下的無溶劑 (neat) 條件下進行，使用 0.2 當量的苦味酸催化劑，反應時間為 60 分鐘 【1-s2.0-S2773223124000268-main】。
> - **產率與適用範圍**：該方法最高產率可達 98%，在最適條件下，芳香族一級胺的產率可達 95% 【1-s2.0-S2773223124000268-main】。其適用於芳香族胺、雜環芳香族胺、開鏈脂肪族胺以及環狀脂肪族胺 【1-s2.0-S2773223124000268-main】。
> - **建議機理**：苦味酸活化 $(Boc)_2O$ 中羰基的氧原子；隨後，胺的氮原子攻擊羰基碳，接著進行質子取代並消除 $\text{CO}_2$ 和叔丁醇 (t-BuOH)，從而形成 N-Boc 保護化合物 【1-s2.0-S2773223124000268-main】。
> - **溫度與時間影響**： 
>     - 在 -10 °C 和 0 °C 時未觀察到產物；在 10 °C 時，產率僅為 20% 【1-s2.0-S2773223124000268-main】。
>     - 縮短反應時間會導致產率下降 【1-s2.0-S2773223124000268-main】。
> - **催化效率**：在沒有催化劑的情況下，反應需要 48 小時才能達到 55% 的產率 【1-s2.0-S2773223124000268-main】。
> - **其他觀察結果**：對於 2-氨基苯酚 (2-aminophenol) 的保護，使用了 1. …（完整內容見 JSON）

---

### ✅ Q06 · multi_chunk

**問題**：How does poly(vinyl alcohol) enhance the therapeutic effect of 4-L-boronophenylalanine in neutron capture therapy of thoracic tumors? Explain the proposed mechanism with supporting data.

- detected_paper：`None`
- 選出論文：['1-s2.0-S0378517325007926-main', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'CMDC-20-e202500059']
- gold_papers：['1-s2.0-S0378517325007926-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：75.0%　grounding：0.833
- 延遲：1788.6s　問題標記：{'conflicts': 0, 'unsupported': 3}

**答案預覽**：

> ## 【論文直接依據】
> - 聚乙烯醇 (PVA) 透過硼酸酯與 4-L-硼苯丙氨酸 (BPA) 形成複合物，產生「PVA-BPA」，其保留了針對大中性氨基酸轉運蛋白 1 (LAT1) 至關重要的類苯丙氨酸結構 [1-s2.0-S0378517325007926-main]。
> - PVA-BPA 與 LAT1 相互作用，並透過「LAT1 介導的胞吞作用」進入癌細胞，隨後定位於內體 (endosomes) 和溶酶體 (lysosomes) 中 [1-s2.0-S0378517325007926-main]。
> - 這種特定的攝取機制增加了細胞對 BPA 的攝取量，並延緩其從細胞質中的流出，從而延長了 BPA 在腫瘤內的滯留時間 [1-s2.0-S0378517325007926-main]。
> - 透過對 LAT1 的主動靶向，PVA-BPA 能在一個小時內高效地積聚並留在腫瘤中，同時限制在正常器官中的積聚 [1-s2.0-S0378517325007926-main]。
> - 使用山梨醇 (sorbitol) 作為增溶劑，開發了一種稱為「PVA-sorbitol-BPA」的優化配方，與原型的 PVA-BPA 複合物相比，該配方將潛在副作用降低至可忽略不計的水平 [1-s2.0-S0378517325007926-main]。
> - 在模擬胸腔內惡性胸膜間皮瘤 (MPM) 的小鼠模型中，與傳統 BPA 相比，全身性給藥 PVA-sorbitol-BPA 在目標腫瘤中實現了更高的積聚量和更長的滯留時間，顯著提高了硼中子捕獲治療 (BNCT) 期間的生存率 [1-s2.0-S0378517325007926-main]。
> 
> ## 【跨文獻推論】
> - **跨論文推論（基於 [Fact 1]、[Fact 2] 和 [Fact 3]）：** PVA 透過改變其細胞進入機制來增強 BPA 的遞送。 …（完整內容見 JSON）

---

### ❌ Q07 · figure_dependent

**問題**：What is the water-stable boroxine structure reported, and what role do the dynamic covalent bonds play in its fluoride binding and hydrogel formation?

- detected_paper：`None`
- 選出論文：['41467_2024_Article_45464', 's41467-024-45464-z (1)']
- gold_papers：['41467_2024_Article_45464']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.167
- 延遲：2892.5s　問題標記：{'conflicts': 0, 'unsupported': 9}

**答案預覽**：

> ## 【論文直接依據】
> - **水穩定硼氧環結構**：報導的結構源自 2-羥基苯硼酸 (HO-PBA)。其形成過程涉及 HO-PBA 在室溫下自發脫水，形成含有動態共價鍵 (DCBs) 並具有聚集誘導發光增強 (AIEE) 活性的二聚體；在室溫下接觸水後，這些二聚體會迅速轉化為硼氧環結構（具體為三聚體-$\text{H}_2\text{O}$ 複合物） 【s41467-024-45464-z】。
> - **穩定性**：此硼氧環結構在水中高度穩定。HO-PBA 三聚體的穩定範圍為 $2.4 < \text{pH} < 9$，而 $\text{CH}_3\text{-HO-PBA}$ 三聚體則為 $2.4 < \text{pH} < 10$ 【s41467-024-45464-z】。
> - **動態共價鍵 (DCB) 特性**：此結構中的 B–O 鍵在室溫水溶液中會發生快速的交換反應，而傳統的硼氧環結構則需要在甲苯溶液中加熱至 $60\text{ °C}$ 並維持 8 小時才能實現交換 【s41467-024-45464-z】。
> - **氟離子結合**：該硼氧環結構能選擇性地識別水溶液中的氟離子 ($\text{F}^-$)，且其結合親和力顯著高於受體苯硼酸 (PBA) 【s41467-024-45464-z】。在 $^1\text{H NMR}$ 實驗中，向 HO-PBA 三聚體加入 $\text{F}^-$ 使峰數從四組增加到十多組，而加入 $\text{Cl}^-$, $\text{Br}^-$, 或 $\text{I}^-$ 則無變化 【s41467-024-45464-z】。
> - **水凝膠形成**：硼氧環結構作為交聯劑連接 `poly(PEGMEA-co-AMA-(HO-PBA))` 聚合物鏈，當該聚合物與水混合時會形成水凝膠 【s41467-024-45464-z】。 …（完整內容見 JSON）

---

### ⚠️ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.5
- 延遲：3817.4s　問題標記：{'conflicts': 0, 'unsupported': 11}

**答案預覽**：

> ## 【論文直接依據】
> 
> **同位素富集與成本效益**
> * $^{10}\text{B}$ 的成本比普通硼高出 1,000 倍以上；因此，引入硼原子後的總步驟數和總產率是合成的關鍵參數 【CMDC-20-e202500059】。
> * 使用 $\text{B}_2\text{pin}_2$ 的合成路徑被認為是不理想的，因為成本高昂且會浪費 $^{10}\text{B}$ 核心 【CMDC-20-e202500059】。
> * 利用二鹵代苯的路徑在過去十年中已大部分被放棄，因為其涉及的步驟過多 【CMDC-20-e202500059】。
> 
> **合成路徑效率與可擴展性**
> * Pd 催化的交叉耦合路徑效率很高，但存在重金屬污染的風險；FDA 將此類污染物在注射劑量中的限制設定為 $10\ \mu\text{g day}^{-1}$ 【CMDC-20-e202500059】。
> * 使用三烷基硼酸酯的非 Pd 催化替代路徑顯示出不同的產率：n-BuLi 法的產率為 $53\text{--}58\%$（使用 $^{10}\text{B}(\text{OBu})_3$ 時為 $62\%$），i-PrMgCl 法的產率為 $78\%$，而 Turbo Grignard ($\text{i-PrMgCl/LiCl}$) 法的產率在 $80\text{--}85\%$ 之間 【CMDC-20-e202500059】。
> * 森田製藥（Morita Pharmaceutical Company）的專利合成被認為非常有前景，儘管其倒數第二步仍需要 Pd 催化劑 【CMDC-20-e202500059】。
> * 在 $\text{HCl}$ 去保護過程中，加入 $\text{NaIO}_4$ 可以防止殘留的硼酸酯，但會引入毒性風險和潛在的產物污染 【CMDC-20-e202500059】。
> 
> **立體化學 …（完整內容見 JSON）

---

### ✅ Q09 · cross_paper

**問題**：Several papers target LAT1 for cancer therapy. What different therapeutic strategies targeting LAT1 are described across the literature, and how do they differ in mechanism?

- detected_paper：`None`
- 選出論文：['LAT1 ChemComm 2026', '1-s2.0-S0378517325007926-main', '1-s2.0-S1347861320300633-main', 's41421-024-00697-6', '1-s2.0-S1936523321002047-main']
- gold_papers：['1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 'LAT1 ChemComm 2026', 's41421-024-00697-6']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.8
- 延遲：2953.9s　問題標記：{'conflicts': 0, 'unsupported': 7}

**答案預覽**：

> ## 【論文直接依據】
> 
> **針對 LAT1 的治療策略**
> *   針對 LAT1 的策略包括使用 L-BPA（p-硼苯丙氨酸）、PVA-BPA、PVA-sorbitol-BPA、選擇性抑制劑（如 JPH203）、mRNA 敲低 (knockdown)、基因干擾以及自組裝肽 【1-s2.0-S0378517325007926-main, 1-s2.0-S1347861320300633-main, 1-s2.0-S1936523321002047-main, s41421-024-00697-6】。
> 
> **作用機制**
> *   **L-BPA：** 它透過 LAT1 運送到細胞質中，並伴隨細胞內氨基酸（如麩醯胺酸）的流出。當胞外 BPA 濃度降低時，它會與胞外酪胺酸進行交換 【1-s2.0-S0378517325007926-main】。
> *   **PVA-BPA：** 此策略利用硼酸酯與多個 BPA 分子形成複合物。這些複合物透過「LAT1 介導的內吞作用」被癌細胞吸收，並定位於內體/溶酶體中，從而增加細胞吸收量、延緩 BPA 流出並延長在腫瘤中的滯留時間 【1-s2.0-S0378517325007926-main】。
> *   **PVA-sorbitol-BPA：** 這是 PVA-BPA 的優化版本，將 pH 值從 9.2–9.5 調整至 7.6，以將腎臟副作用降低至可忽略的程度。它在惡性胸膜間皮瘤 (MPM) 胸腔腫瘤模型中顯示出高累積量並提高了生存率 【1-s2.0-S0378517325007926-main】。
> *   **選擇性抑制 (JPH203)：** 
>     *   JPH203 是一種酪胺酸衍生物，能選擇性地競爭性抑制 LAT1 的運輸活性而非 LAT2 【1-s2.0-S1347861320300633-main, s41421-024-0 …（完整內容見 JSON）

---

### ⚠️ Q10 · contradiction_check

**問題**：Do the papers agree on the hydrolytic stability of boroxine structures? Describe any agreements or differences between them.

- detected_paper：`None`
- 選出論文：['Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation', '41467_2024_Article_45464', 's41467-024-45464-z (1)', 'a-method-for-the-deprotection-of-alkylpinacolyl-boronate-esters']
- gold_papers：['41467_2024_Article_45464', 'Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation']
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.688
- 延遲：3558.9s　問題標記：{'conflicts': 0, 'unsupported': 9}

**答案預覽**：

> ## 【論文直接依據】
> 
> **硼氧環的通用穩定性**
> * 傳統觀點認為，硼氧環在水溶液或潮濕環境中具有水解不穩定性，會迅速水解為硼酸並失去其功能 【s41467-024-45464-z (1)】。
> * 先前的增穩策略——包括引入電子捐贈基團、大位阻基團、N-供體配體加合物，或將三個硼酸單元整合到柔性大環中以實現熵穩定（entropy stabilization）——均未能從根本上解決水解問題 【s41467-024-45464-z (1)】。
> 
> **HO-PBA 衍生硼氧環的穩定性**
> * 2-羥基苯硼酸（HO-PBA）可形成在水中穩定的硼氧環結構 【s41467-024-45464-z (1)】。
> * 其形成過程涉及在室溫下自發脫水形成二聚體，隨後在接觸水時迅速轉化為「HO-PBA 三聚體–$\text{H}_2\text{O}$ 複合物」 【s41467-024-45464-z (1)】。
> * 這些 HO-PBA 三聚體在室溫下可保持穩定長達 7 天，這一結果已通過 $^1\text{H NMR}$、ESI-Q-TOF MS 和 UV-Raman 測量得到證實 【s41467-024-45464-z (1)】。
> * HO-PBA 三聚體的 pH 穩定範圍為 $2.4 < \text{pH} < 9$，而 $\text{CH}_3\text{-HO-PBA}$ 三聚體在 $2.4 < \text{pH} < 10$ 之間穩定 【s41467-024-45464-z (1)】。
> * 以 HO-PBA 硼氧環結構交聯的水凝膠 [poly(PEGMEA-co-AMA-(HO-PBA))] 在酸性 ($\text{pH}=2$) 和鹼性 ($\text{pH}=10$) 溶液中均保持穩定 【s41467-024-45464-z (1)】。
> * HO-PBA 硼氧環中的 B–O 鍵 …（完整內容見 JSON）

---

### ✅ Q11 · out_of_scope

**問題**：What are the phase III clinical trial overall survival results for BNCT in glioblastoma patients?

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', '1-s2.0-S0378517325007926-main', 'cas0106-0279']
- gold_papers：[]
- 論文選擇命中率：N/A　檢索覆蓋率：N/A　grounding：—
- 延遲：359.0s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ⚠️ **資料來源說明**：本地學術文獻資料庫中未找到與此問題直接相關的內容。以下回答來自模型自身知識，非論文原文，請謹慎參考並自行查證。
> 
> 針對您提出的關於「硼中子捕獲治療（Boron Neutron Capture Therapy, BNCT）在膠質母細胞瘤（Glioblastoma, GBM）患者中的第三期臨床試驗整體生存期（Overall Survival, OS）結果」之問題，以下是基於目前學術知識的詳細分析與回答。
> 
> ### 核心結論：缺乏大規模第三期臨床試驗數據
> 
> 首先必須明確指出：**截至目前，全球範圍內尚未有針對膠質母細胞瘤（GBM）且具有高度權威性、大規模隨機對照的「第三期（Phase III）」臨床試驗結果來定義 BNCT 的標準整體生存期（OS）。**
> 
> 大多數關於 BNCT 治療 GBM 的研究仍處於第一期（安全性評估）或第二期（初步療效評估）階段。這主要是由於 BNCT 需要極其特殊的基礎設施（如核反應爐或加速器中子源），導致樣本數難以擴大至第三期試驗所需的規模。
> 
> ### 現有研究之分析（第一期與第二期數據）
> 
> 雖然缺乏第三期數據，但根據已發表的第二期臨床試驗及回顧性研究，BNCT 在 GBM 患者中的表現可概括如下：
> 
> #### 1. 生存期的趨勢
> 在多項針對復發性 GBM 或新診斷 GBM 的小型試驗中，BNCT 顯示出一定的生存獲益。部分研究指出，接受 BNCT 的患者其中位整體生存期（Median OS）有所提升，但這些數據通常是與「歷史對照組」而非「同步隨機對照組」進行比較。
> 
> #### 2. 與標準治療（Stupp Protocol）的對比
> 目前的 GBM 標準治療（手術 $\rightarrow$ 同時放化療 $\rightarrow$ 輔助化療）的中位 OS 約為 14.6 個月。部分 BNCT 的初步研究顯示，對於特定亞群患者 …（完整內容見 JSON）

---

### ❌ Q12 · false_premise

**問題**：Since BPA is administered orally in BNCT, what oral bioavailability values are reported in these papers?

- detected_paper：`None`
- 選出論文：['cas0106-0279', 'Chem Biol Drug Des - 2013 - Achilli - Folic Acid‐Conjugated 4‐Amino‐Phenylboronate  a Boron‐Containing Compound Designed']
- gold_papers：[]
- 論文選擇命中率：N/A　檢索覆蓋率：N/A　grounding：0.0
- 延遲：2209.7s　問題標記：{'conflicts': 0, 'unsupported': 3}

**答案預覽**：

> ## 【論文直接依據】
> 所提供的事實中，均未報告 BPA 的口服生物利用度數值。
> 
> ## 【跨文獻推論】
> 文獻依據不足。雖然 [Fact 5] 提到鑑定出負責 BPA 攝取的芳香族氨基酸轉運蛋白，但缺乏關於給藥途徑或定量藥代動力學數值的數據，以對口服生物利用度做出推論。
> 
> ## 【知識延伸與推測】
> 模型推測（文獻依據不足）：提供的文本側重於 BPA 與 FPB 的生物相容性比較，以及其透過轉運蛋白攝取的細胞機制。[Unverified] 在臨床實踐中，BPA 通常透過靜脈注射而非口服給藥，以確保 BNCT 的精確劑量；因此，口服生物利用度數值可能不是這些特定研究的主要焦點。雖然 BPA 利用芳香族氨基酸轉運蛋白 ([Fact 5]) 的事實表明它在理論上可以在胃腸道中被吸收 [Unverified]，但在缺乏藥代動力學數據的情況下，其實際的口服生物利用度仍然未知。
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
> ⚠️  **以下 3 個陳述未找到明確論文依據，請謹慎參考：**
> 
> - [Direct Paper Evidence] None of the provided facts report oral bioavailability values for BPA.（信心度：10.5%，最近似來源：Chem Biol Drug Des - 2013-150e0782）
> - [Cross-Literature Inference] Insufficient literature …（完整內容見 JSON）

---
