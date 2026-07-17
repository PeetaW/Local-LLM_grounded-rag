# Eval Report — `baseline_v8_structured_product_full`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-17 10:08
- 題數：12

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.583 |
| Correctness judge 覆蓋 | 12/12（N/A 0） |
| 平均翻譯忠實度（LLM-judge） | 0.806 |
| Translation judge 覆蓋 | 9/12（N/A 3） |
| 平均 grounding 分數 | 0.861 |
| 平均論文選擇命中率 | 100.0% |
| 平均檢索覆蓋率 | 94.7% |
| 平均總延遲 | 515.9s |
| 平均 planning 延遲 | 6.9s |
| 平均 retrieval 延遲 | 6.2s |
| └ Phase A embed/vector/BM25 | 6.2s |
| └ Phase B 子答案生成 | 0.0s |
| 平均 grounding 延遲 | 15.5s |
| └ 其中 NLI | 2.0s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | correctness | 翻譯忠實度 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|-------------|------------|---------|---------|-----------|------|------|
| ❌ | Q01 | single_paper | 0.25 | 1.0 | 100.0% | 100.0% | 0.875 | 563.2s | C0/U1 |
| ✅ | Q02 | single_paper | 1.0 | 0.75 | 100.0% | 100.0% | 1.0 | 391.1s | C0/U0 |
| ⚠️ | Q03 | figure_dependent | 0.5 | 0.75 | 100.0% | 100.0% | 0.833 | 666.0s | C0/U2 |
| ❌ | Q04 | table_lookup | 0.0 | 0.75 | 100.0% | 100.0% | 1.0 | 603.9s | C0/U0 |
| ⚠️ | Q05 | single_paper | 1.0 | 1.0 | 100.0% | 100.0% | 0.778 | 516.0s | C0/U2 |
| ❌ | Q06 | multi_chunk | 0.0 | N/A | 100.0% | 80.0% | — | 226.2s | C0/U0 |
| ❌ | Q07 | figure_dependent | 0.25 | 0.75 | 100.0% | 66.7% | 0.714 | 515.1s | C0/U2 |
| ✅ | Q08 | cross_paper | 1.0 | 0.75 | 100.0% | 100.0% | 1.0 | 755.3s | C0/U0 |
| ⚠️ | Q09 | cross_paper | 0.5 | 0.5 | 100.0% | 100.0% | 0.8 | 904.0s | C0/U1 |
| ⚠️ | Q10 | contradiction_check | 1.0 | 1.0 | 100.0% | 100.0% | 0.75 | 541.8s | C0/U1 |
| ✅ | Q11 | out_of_scope | 1.0 | N/A | N/A | N/A | — | 282.1s | C0/U0 |
| ⚠️ | Q12 | false_premise | 0.5 | N/A | N/A | N/A | — | 226.4s | C0/U0 |

## 逐題細節

### ❌ Q01 · single_paper

**問題**：In the study of JPH203's preincubation inhibitory effect on LAT1, how does preincubation change JPH203's inhibitory potency? Give the reported values.

- detected_paper：`None`
- 選出論文：['1-s2.0-S1347861320300633-main', '1-s2.0-S0021979710012403-main', '1-s2.0-S1936523321002047-main', 's41421-024-00697-6']
- gold_papers：['1-s2.0-S1347861320300633-main']
- correctness candidate：`answer_for_judge`
- correctness：0.25　raw：2/5　judge：`structured_fact_audit_v1`
- judge reason：covered 2/5; missing F2, F3, F4
- translation fidelity：1.0　raw：5/5　judge：`translation_fidelity_v1`
- translation reason：The translation accurately conveys all scientific terminology, numerical data, experimental conditions, and logical relationships without omissions or mistranslations.
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.875
- 延遲：563.2s　問題標記：{'conflicts': 0, 'unsupported': 1}

**答案預覽**：

> ⚠️ **資料可能不足**：檢索內容僅部分涵蓋此問題，以下回答可能不完整，請謹慎參考並自行查證。
>
> JPH203-LAT1 組合中的預孵育 (preincubation) 抑制效果與共孵育 (co-incubation) 的抑制效能相比，展現出顯著的增強效果，但其持續時間相對較短【1-s2.0-S1347861320300633-main】。在這些分析的動力學參數 (kinetic parameters) 方面，共孵育 (co-incubation) 抑制分析中濃度為 0.01, 0.1 及 1 mM 的結果顯示 Vmax 值一致且 Km 值增加【1-s2.0-S1347861320300633-main】。預孵育加共孵育 (pre-plus co-incubation) 抑制分析同樣展現出相似的 Vmax 值一致性與 Km 值增加，這與在實驗條件下 JPH203 的抑制作用為競爭性 (competitive) 的特性一致【1-s2.0-S1347861320300633-main】。
>
> 關於與預孵育 (preincubation) 處理條件相關的分子變化，JPH203 (10 mM, 2 h) 顯著降低了 HT-29 細胞中的 LAT1 mRNA 與蛋白質水平【1-s2.0-S1347861320300633-main】。在使用 JPH203 (10 mM) 或 0.5% DMSO (control) 處理 2 h 後，透過即時定量聚合酶連鎖反應 (real-time PCR) 對 JPH203 預孵育對 LAT1 mRNA 水平的影響進行定量【1-s2.0-S1347861320300633-main】。在此定量分析中，mRNA 水平以 $\beta$-actin 進行標準化 (normalized)，並繪製為相對於 $\beta$-actin mRNA 水平的倍數變化 …（完整內容見 JSON）

---

### ✅ Q02 · single_paper

**問題**：What is the hybrid process used for the asymmetric synthesis of optically pure L-p-boronophenylalanine, and what are its key steps?

- detected_paper：`None`
- 選出論文：['bbb0683', 'CMDC-20-e202500059']
- gold_papers：['bbb0683']
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 6/6
- translation fidelity：0.75　raw：4/5　judge：`translation_fidelity_v1`
- translation reason：The translation retains excessive English technical terms (e.g., "enantioselective alkylation", "adduct", "chymotrypsin") within the Chinese text, which constitutes a minor wording imprecision in scientific fidelity despite preserving all factual data.
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：1.0
- 延遲：391.1s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> 方法依據：
> - 光學純的 L-p-boronophenylalanine (L-BPA) 是透過一個包含 enantioselective alkylation (對映選擇性烷基化) 以及隨後的 enzymatic hydrolysis (酶水解) 的 hybrid process 合成的。[bbb0683]
> - 在 -78°C 的 THF 中，將 lithiated (鋰化的) (2R)-(-)-2,5-dihydro-2-isopropyl-3,6-dimethoxypyrazine (bislactimether, 3) 與 protected 4-bromomethylbenzeneboronic acid derivative 2 (亦稱為 4-bromo-methylbenzeneboronate (2)) 進行 enantioselective alkylation (對映選擇性烷基化)，產生了 adduct 4。[bbb0683]
> - Adduct 4 的對映體過量值 (e.e.) 為 74%。[bbb0683]
> - 將 adduct 4 以 hydrochloric acid (鹽酸) 進行 stepwise treatment (逐步處理)，得到了 L-BPA methyl ester 6。[bbb0683]
> - L-BPA methyl ester 6 使用 chymotrypsin 進行水解，以提供光學純的 L-BPA。[bbb0683]
>
> ---
> 📋 **答案品質報告**
>
> ✅ **整體論文依據率**：100.0%　高（答案高度忠實於論文內容）
>
> ✅ **所有陳述均有論文依據**
>
>
> <!-- grounding_score=1.000 -->
> ---

---

### ⚠️ Q03 · figure_dependent

**問題**：According to the cryo-EM structure, how does JPH203 bind to and inhibit the LAT1-4F2hc complex?

- detected_paper：`None`
- 選出論文：['s41421-024-00697-6', '1-s2.0-S1347861320300633-main']
- gold_papers：['s41421-024-00697-6']
- correctness candidate：`answer_for_judge`
- correctness：0.5　raw：3/5　judge：`structured_fact_audit_v1`
- judge reason：covered 4/6; missing F4, F5
- translation fidelity：0.75　raw：4/5　judge：`translation_fidelity_v1`
- translation reason：The translation is scientifically accurate but retains excessive English terms in parentheses (e.g., "sodium-independent exchange manner", "knockdown/knockout") which reduces the fidelity of a complete Traditional Chinese scientific text.
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.833
- 延遲：666.0s　問題標記：{'conflicts': 0, 'unsupported': 2}

**答案預覽**：

> L-型氨基酸轉運蛋白 1 (L-type amino acid transporter 1, LAT1 或 SLC7A5) 與 4F2hc (或 SLC3A2) 耦合，以不依賴鈉離子的交換方式 (sodium-independent exchange manner) 促進大型中性氨基酸與甲狀腺激素跨細胞膜的轉運 【s41421-024-00697-6】。LAT1 被認為是一個抗癌靶點，因為其藥理學抑制或敲低/敲除 (knockdown/knockout) 會抑制各種癌細胞的增殖 【s41421-024-00697-6】。JPH203 由兩個部分組成：具有苯丙氨酸 (phenylalanine) 骨架的親水性頭部基團，以及具有 5-amino-2-(3-aminophenyl) benzoxazole 骨架的疏水性尾部基團 【s41421-024-00697-6】。
>
> 根據冷凍電子顯微鏡 (cryo-EM) 結構，JPH203 結合在 LAT1 的傳統底物結合口袋 (substrate-binding pocket) 中，與抑制劑 Diiodo-Tyr、JX-075、JX-078 和 JX-119 相似 【s41421-024-00697-6】。JPH203 頭部的 $\alpha$-氨基 ($\alpha$-amino group) 與 $\alpha$-羧基 ($\alpha$-carboxyl group) 與 LAT1 中 TM1 和 TM6 解旋區域 (unwound region) 的主鏈原子形成氫鍵網絡 【s41421-024-00697-6】。LAT1 中的 Phe400 與 Tyr259 對於 JPH203 的高選擇性至關重要 【s41421-024-00697-6】。在 JPH203 結合結構 (JPH203-bound structure) 中，TM10 …（完整內容見 JSON）

---

### ❌ Q04 · table_lookup

**問題**：What are the main degradation products and impurities of BPA identified by HPLC, and under which storage conditions do they form?

- detected_paper：`None`
- 選出論文：['1-s2.0-S0731708511003451-main', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water']
- gold_papers：['1-s2.0-S0731708511003451-main']
- correctness candidate：`answer_for_judge`
- correctness：0.0　raw：1/5　judge：`structured_fact_audit_v1`
- judge reason：covered 0/7; missing F1, F2, F4, F5, F7; contradicted F3, F6
- translation fidelity：0.75　raw：4/5　judge：`translation_fidelity_v1`
- translation reason：The translation is scientifically faithful but contains a minor inconsistency in terminology by leaving "aromatic hydrocarbons" untranslated with only Chinese characters for the term, whereas other technical terms like tyrosine and phenylalanine are fully translated.
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：1.0
- 延遲：603.9s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ⚠️ **資料可能不足**：檢索內容僅部分涵蓋此問題，以下回答可能不完整，請謹慎參考並自行查證。
>
> 本研究調查了 boron phenylalanine (BPA) 及其相關雜質的 HPLC 分析方法與降解路徑 【1-s2.0-S0731708511003451-main】。在水性鹼溶液中，BPA 會降解為 tyrosine (酪胺酸) 【1-s2.0-S0731708511003451-main】。BPA 在水性鹼溶液中向 tyrosine (酪胺酸) 的轉化會因鹼的暴露而加速 【1-s2.0-S0731708511003451-main】。在酸性條件下，BPA 會產生 tyrosine (酪胺酸) 【1-s2.0-S0731708511003451-main】。相反地，在鹼性條件下，BPA 會形成 aromatic hydrocarbons (芳香族<0xE7><0x83><0xB4>) 【1-s2.0-S0731708511003451-main】。當在固態下與 mannitol (甘露醇) 共存時，BPA 會部分轉化為 phenylalanine (苯丙氨酸) 【1-s2.0-S0731708511003451-main】。tyrosine (酪胺酸) 與 boric acid (硼酸) 的釋放是由一個過程所促成，該過程涉及 peroxide (過氧化物) 的初步配位（以中性物種形式隨後發生質子流失，或以陰離子形式），以形成 borate species (硼酸鹽物種) 【1-s2.0-S0731708511003451-main】。此過程隨後透過 aryl migration (芳基遷移) 繼續進行，進而促進 hydroxide (氫氧根) 的流失 【1-s2.0-S0731708511003451-main】。
>
> ---
> 📋 **答案品質報告**
>
> ✅ **整體 …（完整內容見 JSON）

---

### ⚠️ Q05 · single_paper

**問題**：What is the solvent-free green protocol for N-Boc protection of amines using picric acid as a catalyst, and what reaction conditions are used?

- detected_paper：`None`
- 選出論文：['1-s2.0-S2773223124000268-main']
- gold_papers：['1-s2.0-S2773223124000268-main']
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 7/7
- translation fidelity：1.0　raw：5/5　judge：`translation_fidelity_v1`
- translation reason：The translation accurately conveys all scientific details, including chemical names, reaction mechanisms, numerical data, units, and logical relationships without any omissions or mistranslations.
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.778
- 延遲：516.0s　問題標記：{'conflicts': 0, 'unsupported': 2}

**答案預覽**：

> 一種使用苦味酸 (picric acid) 作為催化劑，用於各種胺類 (amines) 之 N-Boc 保護的無溶劑且環保的綠色方案已被開發 【1-s2.0-S2773223124000268-main】。此方法能有效地對所有類型的胺類進行 N-Boc 保護 【1-s2.0-S2773223124000268-main】。
>
> 此反應的建議機理涉及苦味酸 (picric acid) 活化 Boc 酐 (Boc anhydride) 中羰基 (carbonyl groups) 的氧原子，隨後胺基 (amine group) 的氮原子透過提供其孤對電子 (lone pair of electrons) 攻擊 Boc 酐 (Boc anhydride) 的羰基碳 【1-s2.0-S2773223124000268-main】。最後，透過取代質子 (protons) 並消除二氧化碳 (carbon dioxide) 和 t-BuOH 作為副產物，形成 N-Boc 保護化合物 【1-s2.0-S2773223124000268-main】。
>
> 關於反應條件，在無溶劑條件下使用 2mol% 的苦味酸 (picric acid) 催化劑，產率為 98% 【1-s2.0-S2773223124000268-main】。獲得極高產率產品的最佳條件是在無溶劑、室溫下，使用 0.2 當量 (equivalents) 的苦味酸 (picric acid) 作為催化劑反應 60 min 【1-s2.0-S2773223124000268-main】。在沒有催化劑的情況下，48 h 後僅獲得 55 % 的產品 【1-s2.0-S2773223124000268-main】。
>
> 溫度影響顯示，在 0 °C 和 −10 °C 時未觀察到產品，而 10 °C 時僅獲得 20 % 的產品 【1-s2.0-S27 …（完整內容見 JSON）

---

### ❌ Q06 · multi_chunk

**問題**：How does poly(vinyl alcohol) enhance the therapeutic effect of 4-L-boronophenylalanine in neutron capture therapy of thoracic tumors? Explain the proposed mechanism with supporting data.

- detected_paper：`None`
- 選出論文：['1-s2.0-S0378517325007926-main', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'CMDC-20-e202500059']
- gold_papers：['1-s2.0-S0378517325007926-main']
- correctness candidate：`answer_for_judge`
- correctness：0.0　raw：1/5　judge：`structured_fact_audit_v1`
- judge reason：covered 0/7; missing F1, F2, F3, F4, F5, F6, F7
- translation fidelity：N/A　raw：N/A　judge：`N/A`
- 論文選擇命中率：100.0%　檢索覆蓋率：80.0%　grounding：—
- 延遲：226.2s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ⚠️ **誠實棄答**：檢索到的內容與此問題相關，但並不包含可直接回答的資訊。為避免編造不存在的數據，本系統選擇不作答。建議換個問法，或確認該主題是否涵蓋於文獻庫中。

---

### ❌ Q07 · figure_dependent

**問題**：What is the water-stable boroxine structure reported, and what role do the dynamic covalent bonds play in its fluoride binding and hydrogel formation?

- detected_paper：`None`
- 選出論文：['41467_2024_Article_45464', 'Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation']
- gold_papers：['41467_2024_Article_45464']
- correctness candidate：`answer_for_judge`
- correctness：0.25　raw：2/5　judge：`structured_fact_audit_v1`
- judge reason：covered 2/8; missing F2, F3, F6, F7, F8; contradicted F5
- translation fidelity：0.75　raw：4/5　judge：`translation_fidelity_v1`
- translation reason：The translation is scientifically faithful and accurate, but the final clause introduces a specific claim about "hydrogel formation" (形成水凝膠) which appears to be an interpretive addition not explicitly present in the source text's list of general application categories.
- 論文選擇命中率：100.0%　檢索覆蓋率：66.7%　grounding：0.714
- 延遲：515.1s　問題標記：{'conflicts': 0, 'unsupported': 2}

**答案預覽**：

> ⚠️ **資料可能不足**：檢索內容僅部分涵蓋此問題，以下回答可能不完整，請謹慎參考並自行查證。
>
> 根據來自【Source 41467_2024_Article_45464】的提供數據，以下資訊探討了水穩定硼氧環 (water-stable boroxine) 結構及其相關性質：
>
> **水穩定硼氧環 (Water-Stable Boroxine) 結構**
> 一種源自於 **2-hydroxyphenylboronic acid** 的水穩定硼氧環 (water-stable boroxine) 結構被發現了 [Source: 41467_2024_Article_45464]。此發現解決了由於水解不穩定性 (hydrolytic instability)，導致在水相介質 (aqueous media) 中利用硼氧環 (boroxines) 通常是一項艱鉅任務的挑戰 [Source: 41467_2024_Article_45464]。
>
> **形成與動態共價鍵 (Dynamic Covalent Bonds)**
> 在環境條件下，**2-hydroxyphenylboronic acid** 會發生自發性脫水反應，形成一種以動態共價鍵 (dynamic covalent bonds) 為特徵的二聚體 (dimer) [Source: 41467_2024_Article_45464]。此過程與聚集誘導增強發光 (aggregation-induced enhanced emission) 活性相關 [Source: 41467_2024_Article_45464]。
>
> **在結合與材料形成中的角色**
> 關於這些結構在結合與材料應用中的角色，硼氧環 (boroxines) 被指出是 **陰離子受體 (anion receptors)** 的重要結構，儘管提供的文本中未詳細說明具 …（完整內容見 JSON）

---

### ✅ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 7/7
- translation fidelity：0.75　raw：4/5　judge：`translation_fidelity_v1`
- translation reason：The translation is scientifically faithful but retains English terms like "chymotrypsin-catalysed enzymatic hydrolysis" and "workup" within the Chinese text, which constitutes a minor wording imprecision for a complete Traditional Chinese scientific translation.
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：1.0
- 延遲：755.3s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ⚠️ **資料可能不足**：檢索內容僅部分涵蓋此問題，以下回答可能不完整，請謹慎參考並自行查證。
>
> 比較框架：
> - 路徑：【bbb0683】報導了對映選擇性烷基化 (enantioselective alkylation)，隨後進行 chymotrypsin-catalysed enzymatic hydrolysis (chymotrypsin 催化的酶水解)，從而以高對映體過量值 (high e.e.) 產出光學純的 L-BPA 【bbb0683】。
> - 回顧/比較來源：【CMDC-20-e202500059】報導 4-borono-L-phenylalanine 的合成已透過多種路徑實現 【CMDC-20-e202500059】。
> - 回顧維度：該回顧強調了每種方法在成本效益 (cost-effectiveness)、可擴展性 (scalability) 和安全性 (safety) 方面的局限性 【CMDC-20-e202500059】。
>
> 核心權衡（高純度/同位素富集與可擴展性及成本效益）：
> - 高純度/同位素富集：在 L-BPA 合成中，生產富含硼-10 (10B) 的高純度物質是一項挑戰 【CMDC-20-e202500059】。
> - 可擴展性：混合合成法 (hybrid synthesis method) 在操作與後處理 (workup) 的便捷性以及反應步驟較少方面具有優勢 【bbb0683】。
> - 成本效益：在製備同位素富集化合物 (isotopically enriched compounds) 時，主要成本通常來自於同位素起始原料 (isotope starting material) 【CMDC-20-e202500059】。
>
> ---
> 📋 **答案品質報告**
>
> ✅ **整體論文依據率**：100.0%　高（答案高度忠實於論文內容）
>
> ✅ **所有 …（完整內容見 JSON）

---

### ⚠️ Q09 · cross_paper

**問題**：Several papers target LAT1 for cancer therapy. What different therapeutic strategies targeting LAT1 are described across the literature, and how do they differ in mechanism?

- detected_paper：`None`
- 選出論文：['LAT1 ChemComm 2026', '1-s2.0-S0378517325007926-main', '1-s2.0-S1347861320300633-main', 's41421-024-00697-6', '1-s2.0-S1936523321002047-main']
- gold_papers：['1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 'LAT1 ChemComm 2026', 's41421-024-00697-6']
- correctness candidate：`answer_for_judge`
- correctness：0.5　raw：3/5　judge：`structured_fact_audit_v1`
- judge reason：covered 5/7; missing F4, F5
- translation fidelity：0.5　raw：3/5　judge：`translation_fidelity_v1`
- translation reason：The translation retains English terms "X-irradiation", "leucine", and "isoleucine" in the target text instead of translating them to their standard Traditional Chinese equivalents (X射線照射、亮氨酸、異亮氨酸), which constitutes a notable omission of scientific terminology.
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.8
- 延遲：904.0s　問題標記：{'conflicts': 0, 'unsupported': 1}

**答案預覽**：

> Comparison scaffold:
> - Route: 【LAT1 ChemComm 2026】報導了一種基於自組裝肽 (self-assembling peptide) 的策略，通過抑制 LAT1 介導的氨基酸運輸並伴隨細胞膜破壞，選擇性地抑制了 MCF-7 和 MDA-MB-231 細胞的增殖 【LAT1 ChemComm 2026】。
> - Route: 【1-s2.0-S0378517325007926-main】報導了利用硼酸酯 (boronate esters) 提高了治療效力；且無論細胞週期如何，均能提高攝取效率 【1-s2.0-S0378517325007926-main】。
> - Route: 【1-s2.0-S1347861320300633-main】報導了一種酪胺酸衍生物 (tyrosine derivative)，其 IC50 值为 60e140 nM；在多種類型的癌細胞以及腫瘤負擔動物中具有抗增殖效果 【1-s2.0-S1347861320300633-main】。
> - Route: 【1-s2.0-S1936523321002047-main】報導了 LAT1 抑制劑 JPH203，通過增強輻射誘導的細胞衰老使癌細胞對輻射敏感；在 X-irradiation 後降低細胞內 leucine 和 isoleucine 水平，導致 mTOR 活性的下調 【1-s2.0-S1936523321002047-main】。
>
> Central trade-off: 針對 LAT1 的治療策略各異，包括使用如 JPH203 等酪胺酸衍生物 (tyrosine derivatives) 的競爭性抑制（其展現出臨床耐受性，且能通過耗盡 leucine 和 isoleucine 以下調 mTOR 活性，從而使細胞對輻射敏感）；如 LﬀVLKK-4Phe 等將運輸抑制 …（完整內容見 JSON）

---

### ⚠️ Q10 · contradiction_check

**問題**：Do the papers agree on the hydrolytic stability of boroxine structures? Describe any agreements or differences between them.

- detected_paper：`None`
- 選出論文：['Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation', '41467_2024_Article_45464', 'a-method-for-the-deprotection-of-alkylpinacolyl-boronate-esters']
- gold_papers：['41467_2024_Article_45464', 'Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation']
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 4/4
- translation fidelity：1.0　raw：5/5　judge：`translation_fidelity_v1`
- translation reason：The translation accurately preserves all technical terms, chemical structures, citation references, and logical relationships without any omissions or semantic errors.
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.75
- 延遲：541.8s　問題標記：{'conflicts': 0, 'unsupported': 1}

**答案預覽**：

> 比較框架：
> - 路徑 (Route)：【Ono】報導了分子內 Boroxine 形成，產生了與 boroxine 4 相比具有更強抗水解能力的 boroxine 2 【Ono】。
> - 綜述/比較來源 (Review/comparison source)：【41467_2024_Article_45464】報導 Boroxine 結構的合成已透過多種路徑實現 【41467_2024_Article_45464】。
> - 綜述維度 (Review dimensions)：該綜述強調了每種方法在電子捐贈基團 (electron-donating groups)、龐大基團 (bulky groups)、與 N-donor ligands 的加合物 (adducts) 以及透過將三個 boronic acid 單元整合到柔性大環 (flexible macrocycle) 中的熵穩定策略 (entropic stabilization strategy) 方面的局限性 【41467_2024_Article_45464】。
> - 核心權衡 (Central trade-off)：雖然 boroxines 通常具有水解不穩定性的特徵，但可以透過特定的結構修飾來實現或增強穩定性，例如使用 2-hydroxyphenylboronic acid 以形成水穩定二聚體 (water-stable dimer)，或透過使用柔性大環 (flexible macrocycles) 的熵穩定 (entropic stabilization) 來實現 【Ono, 41467_2024_Article_45464】。
>
> ---
> 📋 **答案品質報告**
>
> ⚠️ **整體論文依據率**：75.0%　中（部分陳述需要確認）
>
> ⚠️  **以下 1 個陳述未找到明確論文依據，請謹慎參考：**
>
> - Review d …（完整內容見 JSON）

---

### ✅ Q11 · out_of_scope

**問題**：What are the phase III clinical trial overall survival results for BNCT in glioblastoma patients?

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', '1-s2.0-S0378517325007926-main', 'cas0106-0279']
- gold_papers：[]
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 2/2
- translation fidelity：N/A　raw：N/A　judge：`N/A`
- 論文選擇命中率：N/A　檢索覆蓋率：N/A　grounding：—
- 延遲：282.1s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ⚠️ **誠實棄答**：檢索到的內容與此問題相關，但並不包含可直接回答的資訊。為避免編造不存在的數據，本系統選擇不作答。建議換個問法，或確認該主題是否涵蓋於文獻庫中。

---

### ⚠️ Q12 · false_premise

**問題**：Since BPA is administered orally in BNCT, what oral bioavailability values are reported in these papers?

- detected_paper：`None`
- 選出論文：['cas0106-0279', 'Chem Biol Drug Des - 2013 - Achilli - Folic Acid‐Conjugated 4‐Amino‐Phenylboronate  a Boron‐Containing Compound Designed']
- gold_papers：[]
- correctness candidate：`answer_for_judge`
- correctness：0.5　raw：3/5　judge：`structured_fact_audit_v1`
- judge reason：covered 2/3; contradicted F2
- translation fidelity：N/A　raw：N/A　judge：`N/A`
- 論文選擇命中率：N/A　檢索覆蓋率：N/A　grounding：—
- 延遲：226.4s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ⚠️ **誠實棄答**：檢索到的內容與此問題相關，但並不包含可直接回答的資訊。為避免編造不存在的數據，本系統選擇不作答。建議換個問法，或確認該主題是否涵蓋於文獻庫中。

---
