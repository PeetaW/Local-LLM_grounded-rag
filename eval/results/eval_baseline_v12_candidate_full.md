# Eval Report — `baseline_v12_candidate_full`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-08-04 09:26
- 題數：12

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.938 |
| Correctness judge 覆蓋 | 12/12（N/A 0） |
| 平均翻譯忠實度（LLM-judge） | 0.9 |
| Translation judge 覆蓋 | 10/12（N/A 2） |
| 平均 grounding 分數 | 0.921 |
| 平均論文選擇命中率 | 100.0% |
| 平均 retriever candidate 覆蓋率 | 94.7% |
| 平均 Stage 2 evidence 覆蓋率 | 88.2% |
| 平均總延遲 | 422.3s |
| 平均 planning 延遲 | 29.0s |
| 平均 retrieval 延遲 | 6.3s |
| └ Phase A embed/vector/BM25 | 6.3s |
| └ Phase B 子答案生成 | 0.1s |
| 平均 grounding 延遲 | 15.4s |
| └ 其中 NLI | 0.6s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | correctness | 翻譯忠實度 | 選擇命中 | candidate recall | Stage 2 recall | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|-------------|------------|---------|------------------|----------------|-----------|------|------|
| ✅ | Q01 | single_paper | 1.0 | 1.0 | 100.0% | 100.0% | 100.0% | 1.0 | 357.4s | C0/U0 |
| ✅ | Q02 | single_paper | 1.0 | 1.0 | 100.0% | 100.0% | 100.0% | 1.0 | 392.2s | C0/U0 |
| ✅ | Q03 | figure_dependent | 1.0 | 1.0 | 100.0% | 100.0% | 100.0% | 1.0 | 310.7s | C0/U0 |
| ⚠️ | Q04 | table_lookup | 1.0 | 0.5 | 100.0% | 100.0% | 100.0% | 1.0 | 396.9s | C0/U0 |
| ✅ | Q05 | single_paper | 1.0 | 1.0 | 100.0% | 100.0% | 100.0% | 1.0 | 402.6s | C0/U0 |
| ✅ | Q06 | multi_chunk | 1.0 | 1.0 | 100.0% | 80.0% | 100.0% | 1.0 | 350.9s | C0/U0 |
| ⚠️ | Q07 | figure_dependent | 1.0 | 1.0 | 100.0% | 66.7% | 50.0% | 1.0 | 422.2s | C0/U0 |
| ✅ | Q08 | cross_paper | 1.0 | 1.0 | 100.0% | 100.0% | 100.0% | 1.0 | 456.8s | C0/U0 |
| ❌ | Q09 | cross_paper | 0.25 | 1.0 | 100.0% | 100.0% | 57.1% | 0.875 | 1217.4s | C0/U1 |
| ⚠️ | Q10 | contradiction_check | 1.0 | 0.5 | 100.0% | 100.0% | 75.0% | 0.333 | 447.8s | C0/U2 |
| ✅ | Q11 | out_of_scope | 1.0 | N/A | N/A | N/A | N/A | — | 160.2s | C0/U0 |
| ✅ | Q12 | false_premise | 1.0 | N/A | N/A | N/A | N/A | — | 152.0s | C0/U0 |

## 逐題細節

### ✅ Q01 · single_paper

**問題**：In the study of JPH203's preincubation inhibitory effect on LAT1, how does preincubation change JPH203's inhibitory potency? Give the reported values.

- detected_paper：`None`
- 選出論文：['1-s2.0-S1347861320300633-main', '1-s2.0-S0021979710012403-main', '1-s2.0-S1936523321002047-main', 's41421-024-00697-6']
- gold_papers：['1-s2.0-S1347861320300633-main']
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 5/5
- translation fidelity：1.0　raw：5/5　judge：`translation_fidelity_v2`
- translation reason：0 material and 0 minor semantic errors
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：100.0%　grounding：1.0
- 延遲：357.4s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
>
> - 總結而言，我們揭示了 JPH203 能夠透過其預孵育效應 (preincubation effect) 抑制 LAT1 功能，且該預孵育效應能協同增強共孵育抑制效應 (co-incubation inhibitory effects) [Source: 1-s2.0-S1347861320300633-main]。
> - 此外，預孵育與共孵育效應組合的 IC50 值低於單獨共孵育抑制的 IC50 值 (34.2 ± 3.6 nM vs. 99.2 ± 11.0 nM) [Source: 1-s2.0-S1347861320300633-main]。
> - 如圖 5 (Fig. 5) 所示，雖然共孵育抑制分析 (co-incubation inhibition assays) 中的 IC50 值為 99.2 ± 11.0 nM，但加入預孵育顯著增強了其抑制效能 (IC50 = 34.2 ± 3.6 nM) [Source: 1-s2.0-S1347861320300633-main]。
> - 與 IC50 分析結果一致，預孵育加共孵育抑制分析 (pre-plus co-incubation inhibition assays) 中的 Ki 值為 0.37 mM，低於共孵育抑制分析中的 Ki 值 (0.46 mM) [Source: 1-s2.0-S1347861320300633-main]。
> - 這些結果表明，JPH203 以濃度與時間依賴的方式對 HT-29 細胞中的 LAT1 功能產生預孵育抑制效應 [Source: 1-s2.0-S1347861320300633-main]。
> - 根據結果，IC50 值被確定為 193 ± 50 (nM) [Source: 1-s2.0-S1347861320300633-main]。
>
> ---
> 📋 **答案 …（完整內容見 JSON）

---

### ✅ Q02 · single_paper

**問題**：What is the hybrid process used for the asymmetric synthesis of optically pure L-p-boronophenylalanine, and what are its key steps?

- detected_paper：`None`
- 選出論文：['bbb0683', 'CMDC-20-e202500059']
- gold_papers：['bbb0683']
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 6/6
- translation fidelity：1.0　raw：5/5　judge：`translation_fidelity_v2`
- translation reason：0 material and 0 minor semantic errors
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：100.0%　grounding：1.0
- 延遲：392.2s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> 方法證據：
> - 鋰化的 (lithiated) (2R)-( - )-2,5-dihydro- 2-isoprop}'1-3,6-dimethoxypyrazine 與 4-bromo- methylbenzeneboronate (2) 反應，以 74% e.e 得到加成物 4 (adduct 4) 【bbb0683】。
> - 將 4 分步處理 (stepwise treatment) 以鹽酸 (hydrochloric acid)，得到 L-BP A methyl ester 6，隨後經 hydrolyzed with chymotrypsin (使用 chymotrypsin 水解) 以提供光學純的 L-BPA (optically pure L-BPA) 【bbb0683】。
> - 在 - 78°C 的 THF 中，將鋰化的 3 (lithiated 3) 與溴化物 2 (bromide 2) 進行對映選擇性烷基化 (enantioselective alkylation) 【bbb0683】。
> - 將 4 使用 0.1 N 鹽酸 (hydrochloric acid) 處理，得到 D-valine methyl ester (D-Val-OMe) 和具有所需立體化學 (desired stereochemistry) 的氨基酸甲酯 5 (amino acid methyl ester 5) 的等莫量混合物 (equimolar mixture)，其中 D-Val-OMe 通過蒸餾 (distillation) 除去以得到粗產物 5 (crude 5)，隨後經 deprotected by a further treatment with 1 N hydrochloric acid (通過進一步使用 1 N 鹽酸處理進行去保護)，從 4 以 85% 的產率 (yield …（完整內容見 JSON）

---

### ✅ Q03 · figure_dependent

**問題**：According to the cryo-EM structure, how does JPH203 bind to and inhibit the LAT1-4F2hc complex?

- detected_paper：`None`
- 選出論文：['s41421-024-00697-6', '1-s2.0-S1347861320300633-main']
- gold_papers：['s41421-024-00697-6']
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 6/6
- translation fidelity：1.0　raw：5/5　judge：`translation_fidelity_v2`
- translation reason：0 material and 0 minor semantic errors
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：100.0%　grounding：1.0
- 延遲：310.7s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
>
> - 在我們的結構分析中，我們觀察到 JPH203 結合於 LAT1 的傳統底物結合口袋 (traditional substrate-binding pocket) 內，與先前確定的抑制劑如 Diiodo-Tyr 或 JX-075、JX-078 以及 JX-119 相似 (圖 1b, c)13 [Source: s41421-024-00697-6]。
> - 頭部的 $\alpha$-氨基 ($\alpha$-amino group) 和 $\alpha$-羧基 ($\alpha$-carboxyl group) 在穩定分子方面起著關鍵作用，與 LAT1 中 TM1 和 TM6 展開區域 (unwound region) 的主鏈原子形成了氫鍵網絡 (hydrogen bond network) (圖 1b) [Source: s41421-024-00697-6]。
> - 此外，JPH203 的雙鹵代酪氨酸核心 (bi-halogenated tyrosine core) 上的氯原子與 Tyr259 形成了鹵鍵 (halogen bond) [Source: s41421-024-00697-6]。
> - 此外，尾部基團的苯環與 Phe252 的側鏈發生了 T 型 $\pi$-$\pi$ 交互作用 ($\text{T-shaped } \pi\text{--}\pi \text{ interaction}$) (圖 1c) [Source: s41421-024-00697-6]。
> - 疏水性尾部部分 (hydrophobic tail moiety) 緊密地擬合於一個由特定氨基酸構成的疏水口袋 (hydrophobic pocket) 中，這些氨基酸包括：TM1 上的 Ile63、Ile64 和 Ile68；TM3 上的 Ile140、Ile147 和 …（完整內容見 JSON）

---

### ⚠️ Q04 · table_lookup

**問題**：What are the main degradation products and impurities of BPA identified by HPLC, and under which storage conditions do they form?

- detected_paper：`None`
- 選出論文：['1-s2.0-S0731708511003451-main', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water']
- gold_papers：['1-s2.0-S0731708511003451-main']
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 7/7
- translation fidelity：0.5　raw：3/5　judge：`translation_fidelity_v2`
- translation reason：1 material and 0 minor semantic errors; The source states 'producing no detectable degradation', meaning the amount of degradation is zero or below detection limit. The target translates this as '未檢測到可偵測的降解' (did not detect any detectable degradation), which creates a logical contradiction ('not detected' + 'detectable'). It should be '未檢
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：100.0%　grounding：1.0
- 延遲：396.9s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
>
> - BPA 與 tyrosine 的分離在 230 nm 處定量，其保留時間分別為 5.3 和 4.5 min [Source: 1-s2.0-S0731708511003451-main]。
> - 樣品製備與降解分析 (Sample preparation and degradation assays)：BPA 藥品是由 BPA (100 mg/ml) 與 mannitol (110 mg/ml) 在 WFI 中將 pH 調節至 8.0 ± 0.1 後進行冷凍乾燥 (freeze drying) 製得，樣品瓶在黑暗中於 4, 25 和 40 ◦C 下存放數月 [Source: 1-s2.0-S0731708511003451-main]。
> - 降解 (Degradation)：作為原料粉末，BPA 是穩定的，在 55 ◦C 下存放 6 個月或在 40 ◦C 下存放 12 個月時，未檢測到可偵測的降解 [Source: 1-s2.0-S0731708511003451-main]。
> - BPA 強制降解試驗 (forced degradation tests) 使用溶解於 100 mM NaOH, 100 mM HCl 或 5% FeCl3 中的 BPA 進行，且這些樣品在 55 ◦C 下孵育 24 h。6 mM H2O2 的 BPA 溶液樣品則在 HPLC 分析前立即製備 [Source: 1-s2.0-S0731708511003451-main]。
> - 同時觀察到 BPA 在酸性以及 FeCl3 溶液（後者用於測試金屬催化降解 metal-catalysed degradation）中具有穩定性 [Source: 1-s2.0-S0731708511003451-main]。
> - 所有三種合成雜質 (synthetic impurities)  …（完整內容見 JSON）

---

### ✅ Q05 · single_paper

**問題**：What is the solvent-free green protocol for N-Boc protection of amines using picric acid as a catalyst, and what reaction conditions are used?

- detected_paper：`None`
- 選出論文：['1-s2.0-S2773223124000268-main']
- gold_papers：['1-s2.0-S2773223124000268-main']
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 7/7
- translation fidelity：1.0　raw：5/5　judge：`translation_fidelity_v2`
- translation reason：0 material and 0 minor semantic errors
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：100.0%　grounding：1.0
- 延遲：402.6s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
>
> - 利用 picric acid (苦味酸) 作為溫和且有效的催化劑，此方法能夠在無溶劑條件下實現芳香胺 (aromatic amines)、雜環芳香胺 (heterocyclic aromatic amines)、開鏈脂肪胺 (open-chain aliphatic amines) 以及環狀脂肪胺 (cyclic aliphatic amines) 的 N-Boc 保護 [Source: 1-s2.0-S2773223124000268-main]。
> - 此方法利用 di-tert-butyl dicarbonate ((Boc)2O, 二叔丁基二碳酸酯) 來保護芳香胺與脂肪胺，並使用易於獲取的 picric acid (苦味酸, 2 mol%) 作為 Brønsted 酸催化劑 (Brønsted acid catalyst) [Source: 1-s2.0-S2773223124000268-main]。
> - 此方法能夠在完全無溶劑且環保的條件下，快速且高產率（最高達 98 %）地對胺類進行保護，從而促進更清潔且更永續的合成過程 [Source: 1-s2.0-S2773223124000268-main]。
> - 基於這些篩選反應條件，我們得出結論：在室溫、無溶劑條件下，使用 0.2 當量 (equivalents) 的 picric acid 7 作為催化劑並反應 60 min，是獲得產率極佳之產物的最佳條件（Table 1, Entry 13）[Source: 1-s2.0-S2773223124000268-main]。
> - 此方法的顯著優點包括：1) 開創性地使用有機酸 (organic acid) 作為此反應的催化劑，2) 在短時間內實現高產率，3) 催化劑操作方便且具成本效益，4) 無副反應，5) 反應條件環保且綠色，以及 6) …（完整內容見 JSON）

---

### ✅ Q06 · multi_chunk

**問題**：How does poly(vinyl alcohol) enhance the therapeutic effect of 4-L-boronophenylalanine in neutron capture therapy of thoracic tumors? Explain the proposed mechanism with supporting data.

- detected_paper：`None`
- 選出論文：['1-s2.0-S0378517325007926-main', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'CMDC-20-e202500059']
- gold_papers：['1-s2.0-S0378517325007926-main']
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 7/7
- translation fidelity：1.0　raw：5/5　judge：`translation_fidelity_v2`
- translation reason：0 material and 0 minor semantic errors
- 論文選擇命中率：100.0%　candidate 覆蓋率：80.0%　Stage 2 evidence 覆蓋率：100.0%　grounding：1.0
- 延遲：350.9s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
>
> - PVA 透過 boronate esters 與多個 BPA 分子形成複合物 (complex)，保留了針對 LAT1 的關鍵類苯丙氨酸結構 (phenylalanine-like structure) (Fig. 1C)，且該複合物的形成已由 NMR 確認 [Source: 1-s2.0-S0378517325007926-main]。
> - 被稱為 PVA-BPA 的複合物與 LAT1 相互作用，並透過 LAT1 介導的胞吞作用 (LAT1-mediated endocytosis) 被攝入癌細胞中，且定位於內體/溶酶體 (endo-/lysosomes)，從而增加細胞攝取量並延緩 BPA 從細胞質 (cytosol) 的外排 (Fig. 1D) [Source: 1-s2.0-S0378517325007926-main]。
> - 被稱為 PVA-sorbitol-BPA 的優化配方與 PVA-BPA 複合物的原型相比，其可能的副作用顯著降低至可忽略的程度 [Source: 1-s2.0-S0378517325007926-main]。
> - 對照組（冷，control (cold)）、對照組（熱，control (hot)）、sorbitol-BPA 組以及 PVA-sorbitol-BPA 組自治療之日起的中位生存時間及其範圍分別為 23 (range 3–39)、16 (range 16–58)、46 (range 37–82) 以及 85 (62–85 < ) 天 [Source: 1-s2.0-S0378517325007926-main]。
> - 重要的是，靜脈注射的 PVA-sorbitol-BPA 在模擬惡性胸膜間皮瘤 (malignant pleural mesothelioma, MPM) 的胸腔腫瘤模型中表現出高效的腫瘤累積 …（完整內容見 JSON）

---

### ⚠️ Q07 · figure_dependent

**問題**：What is the water-stable boroxine structure reported, and what role do the dynamic covalent bonds play in its fluoride binding and hydrogel formation?

- detected_paper：`None`
- 選出論文：['41467_2024_Article_45464']
- gold_papers：['41467_2024_Article_45464']
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 8/8
- translation fidelity：1.0　raw：5/5　judge：`translation_fidelity_v2`
- translation reason：0 material and 0 minor semantic errors
- 論文選擇命中率：100.0%　candidate 覆蓋率：66.7%　Stage 2 evidence 覆蓋率：50.0%　grounding：1.0
- 延遲：422.2s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
>
> - 總結而言，我們發現了一種水穩定的 boroxine 結構，具有優異的 pH 穩定性以及能在室溫下進行交換的水兼容 DCBs (dynamic covalent bonds)，從根本上解決了 boroxines 長期以來的水解不穩定性挑戰 【Source: 41467_2024_Article_45464】。
> - 此外，我們開發了一種由水穩定 boroxines 交聯的水凝膠 (hydrogel)，由於 boroxine 結構優異的 pH 穩定性和動態特性，該水凝膠表現出高酸鹼穩定性以及可逆的凝膠-溶膠轉變 (gel–sol transition) 【Source: 41467_2024_Article_45464】。
> - 我們發現，在環境條件下，2-hydroxyphenylboronic acid 會發生自發脫水，形成具有動態共價鍵和聚集誘導增強發光 (aggregation-induced enhanced emission) 活性的二聚體 (dimer) 【Source: 41467_2024_Article_45464】。
> - 更有趣的是，在室溫下接觸水後，HO‑PBA 二聚體會迅速轉化為 HO‑PBA 三聚體-H2O 複合物 (HO‑PBA trimer–H2O complexes)（圖 1b），其中包含水穩定的 boroxine 結構，這一點已通過核磁共振 (NMR) 光譜、質譜 (MS)、UV–Raman 光譜以及理論計算得到證實 【Source: 41467_2024_Article_45464】。
> - 這一推測得到了以下事實的支持：如圖 7b 所示，該水凝膠在酸性 (pH=2) 和鹼性 (pH=10) 溶液中均保持穩定 【Source: 41467_2024_Article_45464】。
> - 這些發現使得 boroxines …（完整內容見 JSON）

---

### ✅ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 7/7
- translation fidelity：1.0　raw：5/5　judge：`translation_fidelity_v2`
- translation reason：0 material and 0 minor semantic errors
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：100.0%　grounding：1.0
- 延遲：456.8s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ⚠️ **資料可能不足**：檢索內容僅部分涵蓋此問題，以下回答可能不完整，請謹慎參考並自行查證。
>
> Comparison scaffold (比較框架)：
> - Route (路徑)：【bbb0683】報導了一種涉及對映選擇性烷基化 (enantioselective alkylation) 及隨後酶水解 (enzymatic hydrolysis) 的混合製程，產出光學純的 L-BPA 【bbb0683】。
> - Route detail (路徑細節)：使用 chymotrypsin 進行水解以提供光學純的 L-BPA 【bbb0683】。
> - Review/comparison source (評論/比較來源)：L-BPA 的合成已透過多種路徑進行，反映了在生產高純度、同位素富集 (isotopically enriched) 材料方面的挑戰 【CMDC-20-e202500059】。
> - Review dimensions (評論維度)：該評論強調了每種方法在可擴展性 (scalability)、成本效益 (cost-effectiveness) 和安全性方面的局限性 【CMDC-20-e202500059】。
>
> Central trade-off (high-purity/isotopic enrichment versus scalability and cost-effectiveness) (核心權衡：高純度/同位素富集與可擴展性及成本效益)：
> - High-purity/isotopic enrichment (高純度/同位素富集)：生產 L-BPA 的同位素富集 (isotopically enriched) 材料具有挑戰性 【CMDC-20-e202500059】。
> - Scalability (可擴展性)：該混合製程在操作便捷性、後處理 (workup)  …（完整內容見 JSON）

---

### ❌ Q09 · cross_paper

**問題**：Several papers target LAT1 for cancer therapy. What different therapeutic strategies targeting LAT1 are described across the literature, and how do they differ in mechanism?

- detected_paper：`None`
- 選出論文：['LAT1 ChemComm 2026', '1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 's41421-024-00697-6', 'cas0106-0279']
- gold_papers：['1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 'LAT1 ChemComm 2026', 's41421-024-00697-6']
- correctness candidate：`answer_for_judge`
- correctness：0.25　raw：2/5　judge：`structured_fact_audit_v1`
- judge reason：covered 3/7; missing F1, F3, F5, F6
- translation fidelity：1.0　raw：5/5　judge：`translation_fidelity_v2`
- translation reason：0 material and 0 minor semantic errors
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：57.1%　grounding：0.875
- 延遲：1217.4s　問題標記：{'conflicts': 0, 'unsupported': 1}

**答案預覽**：

> 比較框架：
> - 路徑：JPH203 透過共孵育 (co-incubation) 與預孵育 (preincubation) 效應抑制 LAT1 功能 【1-s2.0-S1347861320300633-main】。
> - 背景：p-Boronophenylalanine (BPA) 被用作硼中子捕獲治療 (Boron Neutron Capture Therapy, BNCT) 的硼遞送劑，其依賴於透過包括 LAT1 在內的轉運蛋白進行選擇性攝取 【cas0106-0279】。
> - 路徑：自組裝肽 (self-assembling peptide) LffVLKK-4Phe 透過靶向 LAT1 抑制癌細胞增殖 【LAT1 ChemComm 2026】。
>
> JPH203 單藥治療（預孵育 preincubation 與共孵育 co-incubation）抑制 HT-29 細胞中不依賴於 Na+ 的 leucine 攝取活性 【1-s2.0-S1347861320300633-main】。
> LffVLKK-4Phe 自組裝肽 (self-assembling peptide) 處理主要透過早期 LAT1 抑制隨後導致膜破壞 (membrane disruption)，從而引起 MCF-7 細胞的增殖抑制 【LAT1 ChemComm 2026】。
>
> JPH203 結合於 LAT1-4F2hc 複合物的底物結合口袋 (substrate-binding pocket) 內，並與 Tyr259 形成鹵鍵 (halogen bond) 【s41421-024-00697-6】。
> LffVLKK-4Phe 利用透過自組裝 (self-assembly) 的多價交互作用 (multivalent interactions)，在癌細胞膜上提供巨大的結合表面 【LAT1 ChemComm 202 …（完整內容見 JSON）

---

### ⚠️ Q10 · contradiction_check

**問題**：Do the papers agree on the hydrolytic stability of boroxine structures? Describe any agreements or differences between them.

- detected_paper：`None`
- 選出論文：['Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation', '41467_2024_Article_45464', 'a-method-for-the-deprotection-of-alkylpinacolyl-boronate-esters']
- gold_papers：['41467_2024_Article_45464', 'Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation']
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 4/4
- translation fidelity：0.5　raw：3/5　judge：`translation_fidelity_v2`
- translation reason：1 material and 0 minor semantic errors; The source states the strategy 'reports' a specific finding, but T1 omits the subject of this reporting action (the paper/author) and starts directly with the verb phrase describing the formation. While contextually implied in Chinese, the explicit attribution to the report's content as stated in S1
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：75.0%　grounding：0.333
- 延遲：447.8s　問題標記：{'conflicts': 0, 'unsupported': 2}

**答案預覽**：

> 比較框架：
> - 策略 (Strategy)：【Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation】報導了使用具有柔性連接基 (flexible linkers) 的大環三硼酸 (macrocyclic triboronic acid) 進行分子內 boroxine 形成 (intramolecular boroxine formation)，由此產生的三環 boroxine (tricyclic boroxine) 與分子間形成的 boroxines (intermolecularly formed boroxines) 相比，展現出極佳的抗水解穩定性 [Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation]。
> - 機制 (Mechanism)：【Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation】報導了 boroxines 的抗水解能力 (hydrolytic resistance) 是透過熵因子 (entropic factors) 和 Lewis 酸度 (Lewis acidity) 的降低而增強的 [Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Borox …（完整內容見 JSON）

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
- 論文選擇命中率：N/A　candidate 覆蓋率：N/A　Stage 2 evidence 覆蓋率：N/A　grounding：—
- 延遲：160.2s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ⚠️ **誠實棄答**：檢索到的內容與此問題相關，但並不包含可直接回答的資訊。為避免編造不存在的數據，本系統選擇不作答。建議換個問法，或確認該主題是否涵蓋於文獻庫中。

---

### ✅ Q12 · false_premise

**問題**：Since BPA is administered orally in BNCT, what oral bioavailability values are reported in these papers?

- detected_paper：`None`
- 選出論文：['cas0106-0279', 'Chem Biol Drug Des - 2013 - Achilli - Folic Acid‐Conjugated 4‐Amino‐Phenylboronate  a Boron‐Containing Compound Designed']
- gold_papers：[]
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 3/3
- translation fidelity：N/A　raw：N/A　judge：`N/A`
- 論文選擇命中率：N/A　candidate 覆蓋率：N/A　Stage 2 evidence 覆蓋率：N/A　grounding：—
- 延遲：152.0s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ⚠️ **前提更正**：檢索文獻沒有報告可供回答的 `oral bioavailability` 數值，因此不能把問題中的前提視為已成立，也不會臆測數值。
>
> 文獻實際報告的是不同的給藥途徑或 regimen：
>
> - Clinically, a high-dose and longer-infusion regimen (900 mg BPA ⁄kg, 6-h infusion) is reported to extend patients’ survival.(1,30) In addition to the transporter kinetics, expression levels of LAT1 and ATB0,+ in tumor tissues are thought to impact on BPA accumulation [cas0106-0279]

---
