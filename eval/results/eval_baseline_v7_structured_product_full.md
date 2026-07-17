# Eval Report — `baseline_v7_structured_product_full`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-16 10:11
- 題數：12

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.667 |
| Correctness judge 覆蓋 | 12/12（N/A 0） |
| 平均 grounding 分數 | 0.867 |
| 平均論文選擇命中率 | 97.5% |
| 平均檢索覆蓋率 | 97.5% |
| 平均總延遲 | 551.3s |
| 平均 planning 延遲 | 6.8s |
| 平均 retrieval 延遲 | 6.0s |
| └ Phase A embed/vector/BM25 | 5.9s |
| └ Phase B 子答案生成 | 0.0s |
| 平均 grounding 延遲 | 15.8s |
| └ 其中 NLI | 1.5s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | 選擇命中 | 檢索覆蓋 | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|---------|---------|-----------|------|------|
| ✅ | Q01 | single_paper | 100.0% | 100.0% | 1.0 | 607.2s | C0/U0 |
| ✅ | Q02 | single_paper | 100.0% | 100.0% | 1.0 | 479.3s | C0/U0 |
| ❌ | Q03 | figure_dependent | 100.0% | 100.0% | 0.25 | 624.6s | C0/U3 |
| ⚠️ | Q04 | table_lookup | 100.0% | 100.0% | 0.667 | 598.6s | C0/U1 |
| ✅ | Q05 | single_paper | 100.0% | 100.0% | 1.0 | 568.6s | C0/U0 |
| ✅ | Q06 | multi_chunk | 100.0% | 75.0% | 1.0 | 492.9s | C0/U0 |
| ✅ | Q07 | figure_dependent | 100.0% | 100.0% | 1.0 | 477.1s | C0/U0 |
| ✅ | Q08 | cross_paper | 100.0% | 100.0% | 1.0 | 692.3s | C0/U0 |
| ⚠️ | Q09 | cross_paper | 75.0% | 100.0% | 0.75 | 982.4s | C0/U1 |
| ✅ | Q10 | contradiction_check | 100.0% | 100.0% | 1.0 | 545.6s | C0/U0 |
| ✅ | Q11 | out_of_scope | N/A | N/A | — | 324.9s | C0/U0 |
| ✅ | Q12 | false_premise | N/A | N/A | — | 222.2s | C0/U0 |

## 逐題細節

### ✅ Q01 · single_paper

**問題**：In the study of JPH203's preincubation inhibitory effect on LAT1, how does preincubation change JPH203's inhibitory potency? Give the reported values.

- detected_paper：`None`
- 選出論文：['1-s2.0-S1347861320300633-main', '1-s2.0-S0021979710012403-main', '1-s2.0-S1936523321002047-main', 's41421-024-00697-6']
- gold_papers：['1-s2.0-S1347861320300633-main']
- correctness candidate：`answer_for_judge`
- correctness：0.25　raw：2/5　judge：`legacy_holistic`
- judge reason：The candidate fails to provide the specific IC50 values requested in the question and instead discusses unrelated kinetic parameters (Km, Vmax) and mRNA/protein expression levels not mentioned as key facts for potency comparison.
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：1.0
- 延遲：607.2s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ⚠️ **資料可能不足**：檢索內容僅部分涵蓋此問題，以下回答可能不完整，請謹慎參考並自行查證。
>
> JPH203-LAT1 組合的預孵育 (preincubation) 抑制效果與同時孵育 (co-incubation) 的抑制效能相比，顯示出顯著的增強效果，但持續時間相對較短 【1-s2.0-S1347861320300633-main】。在動力學特性 (kinetic properties) 方面，同時抑制分析 (co-inhibition assays) 顯示 Vmax 值保持一致且 Km 值增加，這與 JPH203 在實驗條件下具有競爭性 (competitive) 抑制特性的發現一致 【1-s2.0-S1347861320300633-main】。預孵育加同時孵育 (pre-plus co-incubation) 的抑制分析顯示出與同時孵育分析相似的 Vmax 值一致性和 Km 值增加，維持了這種競爭性特徵 (competitive profile) 【1-s2.0-S1347861320300633-main】。
>
> 關於這些動力學觀察的特定濃度，JPH203 對 LAT1 的抑制在 HT-29 細胞中於 0.01, 0.1, 和 1 mM 的濃度下具有競爭性 【1-s2.0-S1347861320300633-main】。然而，使用較高濃度的 JPH203 (10 mM) 預孵育 2 h 會顯著降低 HT-29 細胞中的 LAT1 mRNA 和蛋白質水平 【1-s2.0-S1347861320300633-main】。LAT1 mRNA 水平的效果是在使用 JPH203 (10 mM) 或 0.5% DMSO (control/對照組) 處理 2 h 後，透過 real-time PCR 定量，且 mRNA 水平相對於 b-actin 進行標準化 (normal …（完整內容見 JSON）

---

### ✅ Q02 · single_paper

**問題**：What is the hybrid process used for the asymmetric synthesis of optically pure L-p-boronophenylalanine, and what are its key steps?

- detected_paper：`None`
- 選出論文：['bbb0683', 'CMDC-20-e202500059']
- gold_papers：['bbb0683']
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 6/6
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：1.0
- 延遲：479.3s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> 方法證據：
> - 光學純的 L-p-boronophenylalanine (L-BPA) 是透過一個包含 enantioselective alkylation (對映選擇性烷基化) 以及隨後的 enzymatic hydrolysis (酶水解) 的 hybrid process (混合製程) 合成的。[bbb0683]
> - 此合成涉及使 lithiated (鋰化的) (2R)-( - )-2,5-dihydro-2-isopropyl-3,6-dimethoxypyrazine 與 4-bromomethylbenzeneboronate (2) 反應，以 74% e.e. 產出 adduct 4 (加成物 4)。[bbb0683]
> - 將 adduct 4 以 hydrochloric acid (鹽酸) 進行 stepwise treatment (逐步處理)，可產生 L-BPA methyl ester 6。[bbb0683]
> - 使用 chymotrypsin (胰凝乳蛋白酶) 對 L-BPA methyl ester 6 進行 hydrolysis (水解)，可獲得光學純的 L-BPA。[bbb0683]
> - lithiated (鋰化的) (2R)-( - )-2,5-dihydro-2-isopropyl-3,6-dimethoxypyrazine (bislactimether, 3) 與 bromide 2 的 enantioselective alkylation (對映選擇性烷基化) 是在 THF 中於 -78°C 下進行的。[bbb0683]
>
> ---
> 📋 **答案品質報告**
>
> ✅ **整體論文依據率**：100.0%　高（答案高度忠實於論文內容）
>
> ✅ **所有陳述均有論文依據**
>
>
> <!-- grounding_score=1.000  …（完整內容見 JSON）

---

### ❌ Q03 · figure_dependent

**問題**：According to the cryo-EM structure, how does JPH203 bind to and inhibit the LAT1-4F2hc complex?

- detected_paper：`None`
- 選出論文：['s41421-024-00697-6', '1-s2.0-S1347861320300633-main']
- gold_papers：['s41421-024-00697-6']
- correctness candidate：`answer_for_judge`
- correctness：0.75　raw：4/5　judge：`legacy_holistic`
- judge reason：The candidate correctly identifies the binding pocket and hydrogen bond network but omits key specific structural details present in the reference, such as the halogen bond with Tyr259 involving the chlorine atom on the bi-halogenated tyrosine core.
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.25
- 延遲：624.6s　問題標記：{'conflicts': 0, 'unsupported': 3}

**答案預覽**：

> ⚠️ **資料可能不足**：檢索內容僅部分涵蓋此問題，以下回答可能不完整，請謹慎參考並自行查證。
>
> ### LAT1 功能與靶點狀態 (LAT1 Function and Target Status)
> L-型氨基酸轉運蛋白 1 (L-type amino acid transporter 1, LAT1 或 SLC7A5) 與 4F2hc (或 SLC3A2) 結合，以非鈉依賴性的交換方式 (sodium-independent exchange manner) 促進大型中性氨基酸與甲狀腺激素 (thyroid hormones) 跨細胞膜的轉運 【s41421-024-00697-6】。LAT1 被認為是一個重要的抗癌靶點，因為其藥理學抑制或敲低/敲除 (knockdown/knockout) 能抑制多種癌細胞的增殖 【s41421-024-00697-6】。
>
> ### JPH203 結構與結合機制 (JPH203 Structure and Binding Mechanism)
> JPH203 由兩個不同的部分組成：一個具有苯丙氨酸骨架 (phenylalanine backbone) 的親水性頭部基團 (hydrophilic head group)，以及一個具有 5-amino-2-(3-aminophenyl) benzoxazole 骨架的疏水性尾部基團 (hydrophobic tail group) 【s41421-024-00697-6】。關於結合模式，JPH203 結合在 LAT1 的傳統底物結合口袋 (substrate-binding pocket) 內，與抑制劑 Diiodo-Tyr, JX-075, JX-078 及 JX-119 相似 【s41421-024-00697-6】。JPH203 頭部的 $\alpha$-氨基 ($\alpha$-am …（完整內容見 JSON）

---

### ⚠️ Q04 · table_lookup

**問題**：What are the main degradation products and impurities of BPA identified by HPLC, and under which storage conditions do they form?

- detected_paper：`None`
- 選出論文：['1-s2.0-S0731708511003451-main', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water']
- gold_papers：['1-s2.0-S0731708511003451-main']
- correctness candidate：`answer_for_judge`
- correctness：0.5　raw：3/5　judge：`legacy_holistic`
- judge reason：The candidate correctly identifies tyrosine and phenylalanine as degradation products but incorrectly attributes the formation of aromatic hydrocarbons to alkaline conditions (contradicting the reference which specifies oxidative/basic conditions leading primarily to tyrosine) while omitting all key
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：0.667
- 延遲：598.6s　問題標記：{'conflicts': 0, 'unsupported': 1}

**答案預覽**：

> ⚠️ **資料可能不足**：檢索內容僅部分涵蓋此問題，以下回答可能不完整，請謹慎參考並自行查證。
>
> 根據來自【1-s2.0-S0731708511003451-main】的提供數據，BPA 的降解產物與雜質隨儲存條件（如 pH 狀態和共存物質）而有顯著差異：
>
> **不同條件下的降解產物：**
> 在酸性條件下，BPA 會產生 tyrosine（酪胺酸） 【1-s2.0-S0731708511003451-main】。相比之下，在一般鹼性條件下，BPA 會形成 aromatic hydrocarbons（芳香族<0xE7><0x83><0xB4>） 【1-s2.0-S0731708511003451-main】。然而，特別是在水鹼溶液中，BPA 會降解為 tyrosine（酪胺酸），而非僅形成 aromatic hydrocarbons（芳香族<0xE7><0x83><0xB4>） 【1-s2.0-S0731708511003451-main】。在水鹼溶液中，BPA 向 tyrosine（酪胺酸）的這種轉化會因暴露於 alka 而加速 【1-s2.0-S0731708511003451-main】。此外，當在固態下與 mannitol（甘露醇）共存時，BPA 會部分轉化為 phenylalanine（苯丙胺酸） 【1-s2.0-S0731708511003451-main】。
>
> **機制分析：**
> tyrosine（酪胺酸）和 boric acid（硼酸）的釋放是由一個過程促進的，該過程涉及 peroxide（過氧化物）（以中性物種或陰離子形式存在）的初始配位、將孤對電子捐贈至 boron（硼）的空 p-orbital（p 軌域）以形成 borate species（硼酸鹽物種）、促進 hydroxide（氫氧根）流失的 aryl migration（芳基遷移），以及隨後重新配位 …（完整內容見 JSON）

---

### ✅ Q05 · single_paper

**問題**：What is the solvent-free green protocol for N-Boc protection of amines using picric acid as a catalyst, and what reaction conditions are used?

- detected_paper：`None`
- 選出論文：['1-s2.0-S2773223124000268-main']
- gold_papers：['1-s2.0-S2773223124000268-main']
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`legacy_holistic`
- judge reason：The candidate accurately covers all key reference facts including the catalyst loading, reaction conditions (solvent-free, room temperature, time), yield percentages, substrate scope implications, and comparative data for uncatalyzed reactions.
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：1.0
- 延遲：568.6s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> 用於胺的 N-Boc 保護 (N-Boc protection of amines) 的無溶劑綠色方案 (solvent-free green protocol) 使用苦味酸 (picric acid) 作為催化劑 [1-s2.0-S2773223124000268-main]。在這些條件下，此方法能有效地對所有類型的胺進行 N-Boc 保護 [1-s2.0-S2773223124000268-main]。
>
> 提出的機理涉及苦味酸 (picric acid) 活化 Boc 酐 (Boc anhydride) 中羰基 (carbonyl groups) 的氧原子，隨後由胺基 (amine group) 的氮原子攻擊 Boc 酐 (Boc anhydride) 的羰基碳 (carbonyl carbon) [1-s2.0-S2773223124000268-main]。反應透過胺提供孤對電子 (lone pair electrons) 以形成 N-Boc 保護化合物 (N-Boc-protected compound)，取代質子 (protons) 並以二氧化碳 (carbon dioxide) 和 t-BuOH 作為副產物 (by-products) 消除 [1-s2.0-S2773223124000268-main]。
>
> 關於反應條件，在無溶劑條件下使用 2mol% 的苦味酸 (picric acid) 催化劑進行胺的 N-Boc 保護 (N-Boc protection of amines)，可達到 98% 的產率 (yield) [1-s2.0-S2773223124000268-main]。然而，獲得極高產率產品的最佳條件是在室溫 (room temperature) 下、無溶劑條件下使用 0.2 當量 (equivalents) 的苦味酸 (picric acid) …（完整內容見 JSON）

---

### ✅ Q06 · multi_chunk

**問題**：How does poly(vinyl alcohol) enhance the therapeutic effect of 4-L-boronophenylalanine in neutron capture therapy of thoracic tumors? Explain the proposed mechanism with supporting data.

- detected_paper：`None`
- 選出論文：['1-s2.0-S0378517325007926-main', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'CMDC-20-e202500059']
- gold_papers：['1-s2.0-S0378517325007926-main']
- correctness candidate：`answer_for_judge`
- correctness：0.75　raw：4/5　judge：`legacy_holistic`
- judge reason：The candidate correctly identifies the LAT1 interaction and improved retention but omits the specific mechanism of boronate ester complex formation, the role of sorbitol in reducing side effects, and the explicit survival data mentioned in the reference.
- 論文選擇命中率：100.0%　檢索覆蓋率：75.0%　grounding：1.0
- 延遲：492.9s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> 聚乙烯醇 (Poly(vinyl alcohol) (PVA)) 增強了 4-L-boronophenylalanine 在中子捕獲治療 (neutron capture therapy) 中對胸腔腫瘤的治療效果 【1-s2.0-S0378517325007926-main】。PVA 的生物相容性 (biocompatibility) 已被廣泛報導，支持其作為載體材料 (carrier material) 的使用 【1-s2.0-S0378517325007926-main】。
>
> 所提出的機制涉及與 LAT1 轉運蛋白 (LAT1 transporter) 的特定相互作用。當細胞外 BPA 濃度較高時，它會經由 LAT1 與細胞內胺基酸 (intracellular amino acid) 輕易地進行交換 【1-s2.0-S0378517325007926-main】。在細胞外 BPA 濃度較低時，細胞內 BPA 會與細胞外底物 (extracellular substrate) 進行交換 【1-s2.0-S0378517325007926-main】。對於結合物 (conjugate) 而言，PVA-BPA 與 LAT1 相互作用，隨後通過胞吞作用實現內化 (internalization through endocytosis (LAT1-mediated endocytosis)) 【1-s2.0-S0378517325007926-main】。
>
> 支持數據表明，PVA-BPA 對 LAT1 的靶向作用限制了其在正常器官中的累積 【1-s2.0-S0378517325007926-main】。此外，PVA-BPA 在皮下腫瘤模型 (subcutaneous tumor models) 中顯著增強了 BPA 的抗腫瘤活性 (antitumor activity)，證明了 …（完整內容見 JSON）

---

### ✅ Q07 · figure_dependent

**問題**：What is the water-stable boroxine structure reported, and what role do the dynamic covalent bonds play in its fluoride binding and hydrogel formation?

- detected_paper：`None`
- 選出論文：['41467_2024_Article_45464', 'Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation']
- gold_papers：['41467_2024_Article_45464']
- correctness candidate：`answer`
- correctness：0.25　raw：2/5　judge：`legacy_holistic`
- judge reason：The candidate fails to identify the specific water-stable boroxane structure (HO-PBA trimer-H2O complex) and omits all key details regarding fluoride binding affinity, pH stability ranges, and the reversible gel-to-sol transition mechanism described in the reference.
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：1.0
- 延遲：477.1s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ⚠️ **資料可能不足**：檢索內容僅部分涵蓋此問題，以下回答可能不完整，請謹慎參考並自行查證。
>
> 根據來自 Source 41467_2024_Article_45464 的提供數據：
>
> **水穩定 Boroxine 結構**
> 一種源自 2-hydroxyphenylboronic acid 的水穩定 Boroxine 結構被發現 [41467_2024_Article_45464]。在環境條件下 (ambient environments)，該化合物會發生自發性脫水，形成一種以動態共價鍵 (dynamic covalent bonds) 為特徵的二聚體 (dimer) [41467_2024_Article_45464]。此結構展現出聚集誘導增強發光 (aggregation-induced enhanced emission) 活性，以及在水相介質中的穩定性，而典型的 Boroxines 在此類環境中則面臨水解不穩定性 (hydrolytic instability) [41467_2024_Article_45464]。
>
> **動態共價鍵 (Dynamic Covalent Bonds) 的作用與應用**
> 提供之文本指出 Boroxines 是陰離子受體 (anion receptors) 的重要結構，但在這些片段中並未詳細說明動態共價鍵 (dynamic covalent bonds) 在氟化物結合機制 (fluoride binding mechanisms) 中的具體作用 [41467_2024_Article_45464]。關於水凝膠 (hydrogel) 的形成，該水穩定結構在可修復的水下黏合劑 (repairable underwater adhesive) 和分子構建 (molecular architectures) 方面具有潛在應用，儘管提供的事實 …（完整內容見 JSON）

---

### ✅ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 7/7
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：1.0
- 延遲：692.3s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ⚠️ **資料可能不足**：檢索內容僅部分涵蓋此問題，以下回答可能不完整，請謹慎參考並自行查證。
>
> 比較框架 (Comparison scaffold)：
> - 路徑 (Route)：【bbb0683】報導了對映選擇性烷基化 (enantioselective alkylation)，隨後進行 chymotrypsin-catalysed enzymatic hydrolysis (s-胰蛋白酶催化酶水解)，從而以高 e.e. 產出光學純的 L-BPA 【bbb0683】。
> - 回顧/比較來源 (Review/comparison source)：【CMDC-20-e202500059】報導 4-borono-L-phenylalanine (L-BPA) 的合成已透過多種路徑實現 【CMDC-20-e202500059】。
> - 回顧維度 (Review dimensions)：該回顧強調了每種方法在可擴展性 (scalability)、成本效益 (cost-effectiveness) 和安全性 (safety) 方面的局限性 【CMDC-20-e202500059】。
>
> 核心權衡（高純度/同位素富集與可擴展性及成本效益）：
> - 高純度/同位素富集 (High-purity/isotopic enrichment)：L-BPA 的合成反映了在生產高純度、同位素富集材料（特別是用於 BNCT 的 10B）方面面臨的挑戰 【CMDC-20-e202500059】。
> - 可擴展性 (Scalability)：混合合成法 (hybrid synthesis method) 在操作與後處理的便捷性 (ease of performance and workup) 以及反應步驟較少方面具有優勢 【bbb0683】。
> - 成本效益 (Cost-effectiveness)：在製備同位素富 …（完整內容見 JSON）

---

### ⚠️ Q09 · cross_paper

**問題**：Several papers target LAT1 for cancer therapy. What different therapeutic strategies targeting LAT1 are described across the literature, and how do they differ in mechanism?

- detected_paper：`None`
- 選出論文：['LAT1 ChemComm 2026', '1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 'LAT1 ChemComm 2026SI', 'cas0106-0279']
- gold_papers：['1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 'LAT1 ChemComm 2026', 's41421-024-00697-6']
- correctness candidate：`answer_for_judge`
- correctness：0.25　raw：2/5　judge：`legacy_holistic`
- judge reason：The candidate correctly identifies a self-assembling peptide strategy but fails to mention the specific cryo-EM structural insights required by the reference, while introducing multiple incorrect or hallucinated details (e.g., claiming JPH203 is a tyrosine derivative with clinical trial tolerability
- 論文選擇命中率：75.0%　檢索覆蓋率：100.0%　grounding：0.75
- 延遲：982.4s　問題標記：{'conflicts': 0, 'unsupported': 1}

**答案預覽**：

> 比較框架：
> - 路徑：【LAT1 ChemComm 2026】報導了基於 self-assembling peptide (自組裝肽) 的策略，結果選擇性地抑制了 MCF-7 和 MDA-MB-231 細胞的增殖 [LAT1 ChemComm 2026]。
> - 路徑：【1-s2.0-S0378517325007926-main】報導了利用 boronate esters (硼酸酯)，顯著提高了治療效力；其腫瘤選擇性 (tumor-selectivity) 似乎與 sorbitol-BPA 相當 [1-s2.0-S0378517325007926-main]。
> - 路徑：【1-s2.0-S1347861320300633-main】報導了一種 tyrosine derivative (酪胺酸衍生物) 競爭性抑制劑，其 IC50 值為 60e140 nM；在數種癌症細胞和腫瘤負擔動物 (tumor-bearing animals) 中具有抗增殖效果 [1-s2.0-S1347861320300633-main]。
>
> 核心權衡 (Central trade-off)：針對 LAT1 的治療策略在機制上有所不同：JPH203 作為一種 competitive tyrosine derivative inhibitor (競爭性酪胺酸衍生物抑制劑)，在臨床試驗中提供了耐受性並增強了放射敏感性 (radiosensitivity)；LffVLKK-4Phe 採用 self-assembling peptide (自組裝肽) 策略，將運輸抑制與細胞膜破壞相結合；而 PVA-sorbitol-BPA 則利用 boronate esters (硼酸酯) 來提高不論細胞週期如何的攝取效率，並由 18F-BPA 診斷支持。
>
> ---
> 📋 **答案品質報告**
>
> ⚠️ **整體論文依據率**：75 …（完整內容見 JSON）

---

### ✅ Q10 · contradiction_check

**問題**：Do the papers agree on the hydrolytic stability of boroxine structures? Describe any agreements or differences between them.

- detected_paper：`None`
- 選出論文：['Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation', '41467_2024_Article_45464', 'a-method-for-the-deprotection-of-alkylpinacolyl-boronate-esters']
- gold_papers：['41467_2024_Article_45464', 'Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation']
- correctness candidate：`answer_for_judge`
- correctness：0.5　raw：3/5　judge：`legacy_holistic`
- judge reason：The candidate correctly identifies the general instability and specific stabilization strategies (including entropic stabilization) but fails to explicitly state that both papers agree on this consensus, instead presenting them as a list of findings without synthesizing the "agreement" aspect requir
- 論文選擇命中率：100.0%　檢索覆蓋率：100.0%　grounding：1.0
- 延遲：545.6s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> 比較框架：
> - 評論/比較來源：【41467_2024_Article_45464】報告指出，已採用各種策略使 Boroxine 結構對水解 (hydrolysis) 具有更強的魯棒性 [41467_2024_Article_45464]。
> - 評論維度：該評論強調了水解穩定性 (hydrolytic stability) 的維度 [41467_2024_Article_45464]。
> - 核心權衡：雖然 boroxines 通常具有水解不穩定性 (hydrolytic instability) 的特徵，但可以透過結構修飾 (structural modifications) 來提高穩定性，例如引入電子捐贈基團 (electron-donating groups)、龐大基團 (bulky groups)、N-供體配體 (N-donor ligands)，或採用熵穩定化策略 (entropic stabilization strategy)（將三個硼酸單元 boronic acid units 整合到一個柔性大環 flexible macrocycle 中），且特定的前驅體 (precursors) 如 2-hydroxyphenylboronic acid 可以形成水穩定結構 (water-stable structures) [Ono, 41467_2024_Article_45464]。
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

### ✅ Q11 · out_of_scope

**問題**：What are the phase III clinical trial overall survival results for BNCT in glioblastoma patients?

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', '1-s2.0-S0378517325007926-main', 'cas0106-0279']
- gold_papers：[]
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`legacy_holistic`
- judge reason：The candidate correctly identifies the lack of available data and refuses to fabricate results, aligning perfectly with the reference's instruction.
- 論文選擇命中率：N/A　檢索覆蓋率：N/A　grounding：—
- 延遲：324.9s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ⚠️ **誠實棄答**：檢索到的內容與此問題相關，但並不包含可直接回答的資訊。為避免編造不存在的數據，本系統選擇不作答。建議換個問法，或確認該主題是否涵蓋於文獻庫中。

---

### ✅ Q12 · false_premise

**問題**：Since BPA is administered orally in BNCT, what oral bioavailability values are reported in these papers?

- detected_paper：`None`
- 選出論文：['cas0106-0279', 'Chem Biol Drug Des - 2013 - Achilli - Folic Acid‐Conjugated 4‐Amino‐Phenylboronate  a Boron‐Containing Compound Designed']
- gold_papers：[]
- correctness candidate：`answer_for_judge`
- correctness：0.75　raw：4/5　judge：`legacy_holistic`
- judge reason：The candidate correctly identifies the false premise and refuses to provide non-existent data, but fails to explicitly state that BPA is administered intravenously rather than orally as required by the reference.
- 論文選擇命中率：N/A　檢索覆蓋率：N/A　grounding：—
- 延遲：222.2s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ⚠️ **誠實棄答**：檢索到的內容與此問題相關，但並不包含可直接回答的資訊。為避免編造不存在的數據，本系統選擇不作答。建議換個問法，或確認該主題是否涵蓋於文獻庫中。

---
