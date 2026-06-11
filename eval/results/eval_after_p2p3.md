# Eval Report — `after_p2p3`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-06-11 14:06
- 題數：12

## 彙總

| 指標 | 值 |
|------|-----|
| 平均 grounding 分數 | 0.31 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | N/A |
| 平均總延遲 | 4334.9s |
| 平均 planning 延遲 | 5.3s |
| 平均 retrieval 延遲 | 306.6s |
| 平均 grounding 延遲 | 5537.2s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ✅ | Q01 | single_paper | 100.0% | N/A | — | 417.7s | C0/U0 |
| ❌ | Q02 | single_paper | 100.0% | N/A | 0.167 | 7490.7s | C0/U6 |
| ✅ | Q03 | figure_dependent | 100.0% | N/A | 1.0 | 6855.1s | C0/U3 |
| ⚠️ | Q04 | table_lookup | 100.0% | N/A | 0.75 | 8426.6s | C0/U5 |
| ❌ | Q05 | single_paper | 100.0% | N/A | 0.25 | 5233.9s | C0/U10 |
| ❌ | Q06 | multi_chunk | 100.0% | N/A | 0.0 | 2989.7s | C0/U2 |
| ✅ | Q07 | figure_dependent | 100.0% | N/A | — | 285.8s | C0/U0 |
| ❌ | Q08 | cross_paper | 100.0% | N/A | 0.0 | 2007.3s | C0/U1 |
| ❌ | Q09 | cross_paper | 100.0% | N/A | 0.0 | 17160.5s | C0/U8 |
| ✅ | Q10 | contradiction_check | 100.0% | N/A | — | 506.4s | C0/U0 |
| ✅ | Q11 | out_of_scope | N/A | N/A | — | 339.1s | C0/U0 |
| ✅ | Q12 | false_premise | N/A | N/A | — | 305.9s | C0/U0 |

## 逐題細節

### ✅ Q01 · single_paper

**問題**：In the study of JPH203's preincubation inhibitory effect on LAT1, how does preincubation change JPH203's inhibitory potency? Give the reported values.

- detected_paper：`None`
- 選出論文：['1-s2.0-S1347861320300633-main', '1-s2.0-S0021979710012403-main', '1-s2.0-S1936523321002047-main', 's41421-024-00697-6']
- gold_papers：['1-s2.0-S1347861320300633-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：N/A　grounding：—
- 延遲：417.7s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ⚠️ **資料來源說明**：本地學術文獻資料庫中未找到與此問題直接相關的內容。以下回答來自模型自身知識，非論文原文，請謹慎參考並自行查證。
> 
> 關於 JPH203 對於 LAT1（Large Neutral Amino Acid Transporter 1, SLC7A5）的預孵育（Preincubation）抑制效果，根據目前的藥理學知識與相關研究趨勢，詳細分析如下：
> 
> ### 1. 預孵育對抑制效能的影響
> 在針對 LAT1 的抑制劑研究中，JPH203 被觀察到具有明顯的**時間依賴性（Time-dependent）**或**慢結合動力學（Slow-binding kinetics）**特徵。
> 
> 一般而言，**預孵育會顯著增強 JPH203 的抑制效能（即降低其 $\text{IC}_{50}$ 值）**。這意味著當 JPH203 與 LAT1 蛋白在底物（如亮氨酸 L-leucine）加入前先接觸一段時間，其對轉運功能的抑制效果會比直接同時加入時更強。
> 
> ### 2. 機制分析
> 這種現象通常歸因於以下幾點：
> *   **慢結合動力學**：JPH203 可能在與 LAT1 結合後，經歷了一個構象調整過程（Conformational change），從一個初步的鬆散結合狀態轉變為更穩固、親和力更高的複合體。
> *   **狀態特異性結合**：LAT1 是一種交替訪問轉運蛋白（Alternating-access transporter）。預孵育可能允許 JPH203 在底物競爭壓力較小的情況下，優先佔據特定的構象狀態（例如 inward-facing 或 outward-facing），從而提高其表觀親和力。
> 
> ### 3. 關於報告數值的說明
> 由於您提到的本地資料庫未找到直接文獻，且不同研究使用的細胞株（如 HEK293, MDA-MB-231）以及實驗條件（溫度、緩衝液 …（完整內容見 JSON）

---

### ❌ Q02 · single_paper

**問題**：What is the hybrid process used for the asymmetric synthesis of optically pure L-p-boronophenylalanine, and what are its key steps?

- detected_paper：`None`
- 選出論文：['bbb0683', 'CMDC-20-e202500059']
- gold_papers：['bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：N/A　grounding：0.167
- 延遲：7490.7s　問題標記：{'conflicts': 0, 'unsupported': 6}

**答案預覽**：

> ## 【論文直接依據】
> 
> 光學純 L-p-硼苯丙氨酸 (L-BPA) 的不對稱合成雜化製程結合了「對映選擇性烷基化」以及隨後的「酶促水解」[bbb0683]。此製程旨在比以往的方法更高效，提供更簡單的操作與後處理、更少的反應步驟，並避免了不對稱氫化後需要額外分離/純化的需求，或獲取 $\alpha$-苯甲醯胺肉桂酸衍生物底物之困難 [bbb0683]。
> 
> 此合成的關鍵步驟如下：
> 
> 1.  **起始原料保護**：將 4-溴甲基苯硼酸 (1) 與 2,2-二甲基-1,3-丙二醇在乾燥 THF 中於室溫下攪拌 10 分鐘，以產生環狀硼酸酯 (2)，產率為 79% [bbb0683]。
> 2.  **對映選擇性烷基化**：在氬氣環境下，於 -78°C 的乾燥 THF 中，使用 n-BuLi（1.68M 於己烷中）將 (2R)-(-)-2,5-二氫-2-異丙基-3,6-二甲氧基吡嗪 (3) 進行鋰化。隨後加入溴化物 (2) 並在 -78°C 下攪拌 15 小時，以產生具有 2S 構型的加成產物 4（立體選擇性：74% e.e. 或 72% 非對映異構體過量）[bbb0683]。詳細操作包括在 30 分鐘內加入 n-BuLi，等待 1 小時，然後在 30 分鐘內加入物質 2 的溶液；後處理包括加入磷酸鹽緩衝液 (pH 7.0, 30ml) 並升溫至室溫 [bbb0683]。
> 3.  **L-BPA 甲酯 (6) 的合成**：分為兩個階段 [bbb0683]：
>     *   **階段 1**：將產物 4 溶解於 THF 中；在 5°C 下加入 0.1 N HCl 並在室溫下攪拌 8 小時，產生 D-Val-OMe 與氨基酸甲酯 5 的等量混合物。透過蒸餾 (50°C, 1 mmHg) 去除 D-Val-OMe 以獲得粗產物 5 [bbb0683]。
>     *   **階段 2**：將粗 …（完整內容見 JSON）

---

### ✅ Q03 · figure_dependent

**問題**：According to the cryo-EM structure, how does JPH203 bind to and inhibit the LAT1-4F2hc complex?

- detected_paper：`None`
- 選出論文：['s41421-024-00697-6', '1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 's41467-024-45464-z (1)', 'LAT1 ChemComm 2026']
- gold_papers：['s41421-024-00697-6']
- 論文選擇命中率：100.0%　檢索覆蓋率：N/A　grounding：1.0
- 延遲：6855.1s　問題標記：{'conflicts': 0, 'unsupported': 3}

**答案預覽**：

> 根據提供的事實，以下是關於 JPH203 如何結合並抑制 LAT1-4F2hc 複合體的全面回答。
> 
> ## 【論文直接依據】
> *   **結合位點與結構：** JPH203 結合於 LAT1 的傳統底物結合口袋中 [Fact 1]。其分子結構由具有苯丙氨酸骨架的親水性頭部，以及具有 5-amino-2-(3-aminophenyl) benzoxazole 骨架的疏水性尾部組成 [Fact 2]。
> *   **親水性頭部交互作用：** 頭部的 $\alpha$-氨基和 $\alpha$-羧基與 TM1 和 TM6 解旋區域的主鏈原子形成氫鍵網絡；此外，雙鹵代酪氨酸核心上的氯原子與 Tyr259 形成鹵鍵 [Fact 3]。
> *   **疏水性尾部交互作用：** 尾部嵌入由 TM1 (Ile63, Ile64, Ile68)、TM3 (Ile140, Ile147, Ile148)、TM6 (Phe252, Gly255) 和 TM10 (Ile397, Phe400, Phe403) 的殘基組成的疏水口袋中 [Fact 4]。具體而言，尾部的氨基與 TM10 上 Phe400 的主鏈形成氫鍵，且其苯環與 Phe252 的側鏈發生 T 型 $\pi$-$\pi$ 交互作用 [Fact 5]。
> *   **結構分析：** 冷凍電子顯微鏡 (Cryo-EM) 解析了結合 JPH203 的 LAT1-4F2hc 複合體之向外開放構象 (outward-facing conformation)，整體解析度為 3.30 Å，JPH203 的聚焦精修解析度為 3.25 Å [Fact 6]。
> *   **選擇性與親和力：** Phe400 和 Tyr259 是決定高選擇性的關鍵殘基；在 LAT2 中，這些對應於 Val391 和 Phe250，導致鹵鍵的缺失 [Fact 7]。WT LA …（完整內容見 JSON）

---

### ⚠️ Q04 · table_lookup

**問題**：What are the main degradation products and impurities of BPA identified by HPLC, and under which storage conditions do they form?

- detected_paper：`None`
- 選出論文：['1-s2.0-S0731708511003451-main', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water']
- gold_papers：['1-s2.0-S0731708511003451-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：N/A　grounding：0.75
- 延遲：8426.6s　問題標記：{'conflicts': 0, 'unsupported': 5}

**答案預覽**：

> ## 【論文直接依據】
> 
> **透過 HPLC 鑑定之降解產物與雜質：**
> * **酪胺酸 (Tyrosine)：** 定量波長為 230 nm，保留時間為 4.5 分鐘 [Chunk 1]。
> * **苯丙胺酸 (Phenylalanine)：** 定量波長為 256 nm，保留時間為 11.0 分鐘 [Chunk 1]。
> * **BrPD (2-(4-Bromophenyl)-[1,3]dioxane)：** 一種合成中間體雜質，於 256 nm 處檢出，保留時間為 17.3 分鐘 [Chunk 1]。
> * **FBBA (4-Formylbenzeneboronic acid)：** 一種合成中間體雜質，於 256 nm 處檢出，保留時間為 23.7 分鐘 [Chunk 1]。
> * **BDPA (2-tert-Butoxycarbonylamino-3-[4-(5,5-dimethyl-[1,3,2]dioxaborinan-2-yl)-phenyl]acrylic acid methyl ester)：** 一種合成中間體雜質，於 270 nm 處檢出；其與 FBBA 共洗脫 (co-elutes) [Chunk 1]。
> 
> **儲存與生成條件：**
> * **酪胺酸 (Tyrosine) 的生成：** 發生於鹼性及氧化條件下，且在氧化條件下的降解速度極快 [Chunk 1, Chunk 2]。具體而言，當暴露於 $100\text{ mM NaOH}$（鹼性）或 $6\text{ mM H}_2\text{O}_2$（氧化）並於 $55^\circ\text{C}$ 下放置 24 小時時會產生酪胺酸 [Chunk 2]。
> * **苯丙胺酸 (Phenylalanine) 的生成：** 發生於冷凍乾燥的 BPA/甘露醇 (mannitol) 藥品中。此過程緩慢且具有溫度依 …（完整內容見 JSON）

---

### ❌ Q05 · single_paper

**問題**：What is the solvent-free green protocol for N-Boc protection of amines using picric acid as a catalyst, and what reaction conditions are used?

- detected_paper：`None`
- 選出論文：['1-s2.0-S2773223124000268-main']
- gold_papers：['1-s2.0-S2773223124000268-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：N/A　grounding：0.25
- 延遲：5233.9s　問題標記：{'conflicts': 0, 'unsupported': 10}

**答案預覽**：

> ## 【論文直接依據】
> 使用苦味酸（picric acid）作為催化劑的胺類 N-Boc 保護無溶劑綠色方案具有以下特點：
> 
> *   **催化劑與試劑**：該反應使用苦味酸作為 Brønsted 酸催化劑 [1-s2.0-S2773223124000268-main]，並使用二叔丁基二碳酸酯 ($\text{(Boc)}_2\text{O}$ / Boc anhydride) 作為試劑 [1-s2.0-S2773223124000268-main]。
> *   **反應條件**：該過程在室溫 (rt) 下，於完全無溶劑（neat）的條件下進行 [1-s2.0-S2773223124000268-main]。
> *   **最適參數**：
>     *   最適催化劑用量為 0.2 equivalents，儘管使用 0.1、0.5 和 1.0 equivalents 也能獲得極佳的產率 [1-s2.0-S2773223124000268-main]。
>     *   最適反應時間為 60 minutes [1-s2.0-S2773223124000268-main]。
> *   **性能與範圍**：
>     *   該方法最高可達到 98% 的產率 [1-s2.0-S2773223124000268-main]。
>     *   在最適條件下（0.2 eq 催化劑，室溫，60 min），芳香族初級胺的產率為 95% [1-s2.0-S2773223124000268-main]。
>     *   該方案適用於各種胺類底物，包括芳香族胺、雜環芳香族胺、開鏈脂肪族胺和環狀脂肪族胺 [1-s2.0-S2773223124000268-main]。
> *   **擬議機制**：反應分三個階段進行：(1) 苦味酸活化 $\text{Boc}$ anhydride 的羰基氧；(2) 胺的氮原子攻擊 …（完整內容見 JSON）

---

### ❌ Q06 · multi_chunk

**問題**：How does poly(vinyl alcohol) enhance the therapeutic effect of 4-L-boronophenylalanine in neutron capture therapy of thoracic tumors? Explain the proposed mechanism with supporting data.

- detected_paper：`None`
- 選出論文：['1-s2.0-S0378517325007926-main', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'CMDC-20-e202500059']
- gold_papers：['1-s2.0-S0378517325007926-main']
- 論文選擇命中率：100.0%　檢索覆蓋率：N/A　grounding：0.0
- 延遲：2989.7s　問題標記：{'conflicts': 0, 'unsupported': 2}

**答案預覽**：

> 由於提供的來源資料不包含實質資訊（僅由錯誤訊息與「Empty Response」組成），因此無法提供基於特定論文的證據導向答案。
> 
> 因此，由於缺乏足夠的文獻基礎以填充內容，所有章節均被省略。
> 
> ---
> 📋 **答案品質報告**
> 
> ❌ **整體論文依據率**：0.0%　低（建議重新查詢或縮小問題範圍）
> 
> ⚠️  **以下 2 個陳述未找到明確論文依據，請謹慎參考：**
> 
> - Because the provided source material contains no substantial information (it consists of error messages and "Empty Response"), it is impossible to provide evidence-based answers derived from specific papers.（信心度：3.0%，最近似來源：cas0106-0279-f23bbb8b）
> - Consequently, all sections are omitted as there is insufficient literature basis to populate them.（信心度：0.1%，最近似來源：bbb0683-5edb73bf）
> 
> <!-- grounding_score=0.000 -->
> ---

---

### ✅ Q07 · figure_dependent

**問題**：What is the water-stable boroxine structure reported, and what role do the dynamic covalent bonds play in its fluoride binding and hydrogel formation?

- detected_paper：`None`
- 選出論文：['41467_2024_Article_45464', 's41467-024-45464-z (1)']
- gold_papers：['41467_2024_Article_45464']
- 論文選擇命中率：100.0%　檢索覆蓋率：N/A　grounding：—
- 延遲：285.8s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ⚠️ **資料來源說明**：本地學術文獻資料庫中未找到與此問題直接相關的內容。以下回答來自模型自身知識，非論文原文，請謹慎參考並自行查證。
> 
> 這是一個高度專業的超分子化學（Supramolecular Chemistry）與材料科學問題。雖然您提到的具體文獻未在本地資料庫中直接命中，但根據硼化學（Boron Chemistry）以及動態共價鍵（Dynamic Covalent Bonds, DCBs）的通用學術原理，我可以為您提供詳細的分析。
> 
> 這類研究通常聚焦於克服硼氧環（Boroxine）在水溶液中極易水解為硼酸（Boronic acid）的缺陷。以下是針對該問題的學術解答：
> 
> ### 1. 水穩定硼氧環（Water-stable Boroxine）的結構特徵
> 
> 一般的硼氧環是由三個 $\text{B}$ 原子和三個 $\text{O}$ 原子構成的六元環 ($\text{B}_3\text{O}_3$)。在常規情況下，水分子會迅速親核攻擊 $\text{B}$ 中心的空 $\text{p}$ 軌域，導致環狀結構崩潰。
> 
> 為了實現「水穩定性」，研究者通常採取以下兩種策略之一（**此部分為模型根據學術常理之推測，建議查閱原始文獻確認具體分子式**）：
> *   **空間位阻效應（Steric Hindrance）：** 在硼原子周圍引入體積龐大的取代基（例如：間三甲基苯基 Mesityl 或其他具有正交取代的芳香環）。這些龐大的基團能形成物理屏障，阻止水分子接近 $\text{B}$ 中心，從而動力學上抑制水解。
> *   **電子效應與配位穩定化（Electronic Stabilization）：** 通過引入特定的電子供體或利用鄰近基團的協同作用（如內部配位），增加 $\text{B}$ 原子的電子密度，降低其路易斯酸性，使其對水分子的親和力下降。
> 
> ### 2. 動態 …（完整內容見 JSON）

---

### ❌ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- 論文選擇命中率：100.0%　檢索覆蓋率：N/A　grounding：0.0
- 延遲：2007.3s　問題標記：{'conflicts': 0, 'unsupported': 1}

**答案預覽**：

> 由於提供的來源文本不包含實質內容（僅顯示「Empty Response」或系統故障訊息），因此無法從指定的論文中找到可用證據來回答該問題。
> 
> ---
> 📋 **答案品質報告**
> 
> ❌ **整體論文依據率**：0.0%　低（建議重新查詢或縮小問題範圍）
> 
> ⚠️  **以下 1 個陳述未找到明確論文依據，請謹慎參考：**
> 
> - Due to the fact that the provided source text contains no substantive content (stating only "Empty Response" or system failure messages), there is no evidence available from the specified papers to answer the question.（信心度：8.8%，最近似來源：synthesis-and-biological--6465c2e8）
> 
> <!-- grounding_score=0.000 -->
> ---

---

### ❌ Q09 · cross_paper

**問題**：Several papers target LAT1 for cancer therapy. What different therapeutic strategies targeting LAT1 are described across the literature, and how do they differ in mechanism?

- detected_paper：`None`
- 選出論文：['LAT1 ChemComm 2026', '1-s2.0-S0378517325007926-main', '1-s2.0-S1347861320300633-main', 's41421-024-00697-6', '1-s2.0-S1936523321002047-main']
- gold_papers：['1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 'LAT1 ChemComm 2026', 's41421-024-00697-6']
- 論文選擇命中率：100.0%　檢索覆蓋率：N/A　grounding：0.0
- 延遲：17160.5s　問題標記：{'conflicts': 0, 'unsupported': 8}

**答案預覽**：

> 根據提供的輸入，目前沒有可供分析的學術論文或提取的事實（來源文本僅包含顯示「Empty Response」或「Generation Failed」的錯誤訊息）。因此，我無法提供基於特定文獻的證據導向答案。
> 
> ## 【論文直接依據】
> （無可用內容；輸入中未提供任何論文。）
> 
> ## 【跨文獻推論】
> 文獻基礎不足。
> 
> ## 【知識延伸與推測】
> 模型推測（文獻基礎不足）：由於未提供特定論文，我將根據關於腫瘤學中 LAT1 (SLC7A5) 的一般學術知識進行外推：
> 
> *   **治療策略：** [Unverified] 在一般癌症研究中，LAT1 是被標記的目標，因為它在各種腫瘤中經常過度表達，以滿足對必需氨基酸 (EAAs) 的高需求。（原因：已知事實列表中未提供證據。）
>     *   **小分子抑制：** [Unverified] 使用競爭性抑制劑來阻斷 EAAs（如 leucine 和 phenylalanine）的運輸，從而誘導代謝壓力。（原因：已知事實列表中未提供證據。）
>     *   **針對伴隨蛋白 (CD98hc)：** [Unverified] 由於 LAT1 與 CD98hc 以異二聚體形式運作，相關策略可能涉及破壞此交互作用，或使用針對 CD98hc 的抗體以防止 LAT1 到達細胞膜。（原因：已知事實列表中未提供證據。）
>     *   **PROTACs：** [Unverified] 利用蛋白水解靶向嵌合體 (Proteolysis Targeting Chimeras) 完全降解 LAT1 蛋白，而非僅抑制其活性。（原因：已知事實列表中未提供證據。）
> *   **機制差異：** 
>     *   [Unverified] 直接抑制劑主要針對運輸功能，導致「氨基酸飢餓」。（原因：已知事實列表中未提供證據。）
>     *   [Unverified]  …（完整內容見 JSON）

---

### ✅ Q10 · contradiction_check

**問題**：Do the papers agree on the hydrolytic stability of boroxine structures? Describe any agreements or differences between them.

- detected_paper：`None`
- 選出論文：['Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation', '41467_2024_Article_45464', 's41467-024-45464-z (1)', 'a-method-for-the-deprotection-of-alkylpinacolyl-boronate-esters']
- gold_papers：['41467_2024_Article_45464', 'Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation']
- 論文選擇命中率：100.0%　檢索覆蓋率：N/A　grounding：—
- 延遲：506.4s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ⚠️ **資料來源說明**：本地學術文獻資料庫中未找到與此問題直接相關的內容。以下回答來自模型自身知識，非論文原文，請謹慎參考並自行查證。
> 
> 關於硼氧環（Boroxine, $\text{B}_3\text{O}_3$ 結構）的水解穩定性，學術界在**基本化學原理**上具有高度共識，但在**實際應用場景（如小分子與聚合物/框架材料）的穩定程度**上則存在顯著的討論差異。
> 
> 以下詳細分析其一致之處與分歧之處：
> 
> ### 1. 學術界的一致觀點 (Points of Agreement)
> 
> 絕大多數文獻均認同以下基本化學特性：
> 
> *   **熱力學不穩定性（相對於水）：** 硼氧環被普遍認為在水環境中是不穩定的。其核心反應是一個可逆的平衡過程：一個硼氧環分子與三個水分子反應，會水解回三個對應的硼酸（Boronic acid）分子。
>     $$\text{(R-BO)}_3 + 3\text{H}_2\text{O} \rightleftharpoons 3\text{R-B(OH)}_2$$
> *   **可逆性：** 文獻一致認為此過程是高度可逆的。透過脫水反應（例如加熱或使用脫水劑），硼酸可以重新縮合形成硼氧環。
> *   **親電性質：** 硼原子在硼氧環中保持 $\text{sp}^2$ 混成且具有空 $\text{p}$ 軌域，這使其成為強親電中心，極易受到水分子（親核試劑）的攻擊，這是導致其水解的根本原因。
> 
> ### 2. 文獻中的差異與討論焦點 (Points of Difference/Nuance)
> 
> 雖然基本原理一致，但在探討「如何提高穩定性」或「在特定結構中是否穩定」時，不同論文的結論有所不同：
> 
> #### A. 立體阻礙與電子效應 (Steric and Electronic Effects)
> *   **共識部分：** 增加硼原子周圍的立體阻礙（例如引入 …（完整內容見 JSON）

---

### ✅ Q11 · out_of_scope

**問題**：What are the phase III clinical trial overall survival results for BNCT in glioblastoma patients?

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', '1-s2.0-S0378517325007926-main', 'cas0106-0279']
- gold_papers：[]
- 論文選擇命中率：N/A　檢索覆蓋率：N/A　grounding：—
- 延遲：339.1s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ⚠️ **資料來源說明**：本地學術文獻資料庫中未找到與此問題直接相關的內容。以下回答來自模型自身知識，非論文原文，請謹慎參考並自行查證。
> 
> 針對您提出的關於硼中子捕獲治療（Boron Neutron Capture Therapy, BNCT）在膠質母細胞瘤（Glioblastoma, GBM）患者中第三期臨床試驗整體生存期（Overall Survival, OS）結果的問題，以下是基於目前學術知識的詳細分析。
> 
> ### 核心結論：關於第三期臨床試驗的現狀
> 
> 首先必須明確指出：**截至目前，全球範圍內尚未有針對膠質母細胞瘤（GBM）且具有高度權威性、大規模隨機對照的第三期（Phase III）臨床試驗結果被公認地發表並改變目前的標準治療指南。**
> 
> 大多數關於 BNCT 治療 GBM 的數據來源於第一期（安全性評估）或第二期（初步療效評估）臨床試驗，以及部分回顧性研究或小規模的前瞻性隊列研究。因此，目前無法提供一個統一的、基於第三期試驗的 OS 標準數值。
> 
> ---
> 
> ### 現有臨床數據分析（主要為 Phase I/II 及 Pilot Studies）
> 
> 雖然缺乏大規模第三期試驗，但從已發表的第二期或初步研究中，可以觀察到以下趨勢：
> 
> #### 1. 復發性膠質母細胞瘤 (Recurrent GBM)
> 對於已經接受過標準治療（手術、放療、化療）後復發的患者，BNCT 通常被視為挽救療法（Salvage Therapy）。
> *   **生存期表現：** 在部分研究中，復發性 GBM 患者在接受 BNCT 後的中位整體生存期（Median OS）約在 **6 至 12 個月** 之間。
> *   **對比意義：** 此數值通常高於僅接受姑息治療或單純化療的復發患者，但由於缺乏隨機對照組，其統計學顯著性仍有爭議。
> 
> #### 2. 初診膠質母細胞瘤 (Newly Diagn …（完整內容見 JSON）

---

### ✅ Q12 · false_premise

**問題**：Since BPA is administered orally in BNCT, what oral bioavailability values are reported in these papers?

- detected_paper：`None`
- 選出論文：['cas0106-0279', 'Chem Biol Drug Des - 2013 - Achilli - Folic Acid‐Conjugated 4‐Amino‐Phenylboronate  a Boron‐Containing Compound Designed']
- gold_papers：[]
- 論文選擇命中率：N/A　檢索覆蓋率：N/A　grounding：—
- 延遲：305.9s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ⚠️ **資料來源說明**：本地學術文獻資料庫中未找到與此問題直接相關的內容。以下回答來自模型自身知識，非論文原文，請謹慎參考並自行查證。
> 
> 針對您提出的關於硼中子捕獲治療（Boron Neutron Capture Therapy, BNCT）中所使用的 BPA（L-Boronophenylalanine，硼苯丙氨酸）口服生物利用度（Oral Bioavailability）的問題，由於您未提供具體的參考文獻，我將基於目前的學術知識庫為您解答。
> 
> ### 學術分析與回答
> 
> 在 BNCT 的臨床實踐中，**BPA 通常是以靜脈注射（Intravenous infusion）的方式給藥**，而非口服。這是為了確保硼化合物能迅速且精確地達到血液循環，並透過 L-型氨基酸轉運蛋白 1（LAT1）高效地積聚在腫瘤細胞中，以維持較高的腫瘤與正常組織之硼濃度比（T/N ratio）。
> 
> 關於 BPA 的口服生物利用度，目前的學術現況如下：
> 
> #### 1. 口服給藥的罕見性與挑戰
> 在大多數已發表的 BNCT 臨床試驗中，極少使用口服路徑。主要原因在於：
> *   **吸收率的不確定性**：BPA 作為一種氨基酸類比物，其在腸道的吸收效率可能受到競爭性氨基酸的影響。
> *   **藥代動力學（PK）控制**：BNCT 要求在照射前將腫瘤內的硼濃度提升至峰值，靜脈注射能提供更可預測的血藥濃度曲線。
> 
> #### 2. 生物利用度的數值分析
> 由於口服 BPA 並非主流給藥方式，**目前缺乏一個被廣泛認可的、標準化的「口服生物利用度百分比」數值**。在少數探討口服路徑的基礎研究或動物實驗中，其生物利用度通常低於靜脈注射（定義上 IV 為 100%）。
> 
> 若您在特定論文中看到相關數據，請注意以下幾點：
> *   **劑型影響**：BPA 常以 BPA-果糖複合物（BPA-fructose complex …（完整內容見 JSON）

---
