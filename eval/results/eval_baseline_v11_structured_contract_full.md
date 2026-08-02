# Eval Report — `baseline_v11_structured_contract_full`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-08-02 14:05
- 題數：12

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.708 |
| Correctness judge 覆蓋 | 12/12（N/A 0） |
| 平均翻譯忠實度（LLM-judge） | 0.95 |
| Translation judge 覆蓋 | 10/12（N/A 2） |
| 平均 grounding 分數 | 1.0 |
| 平均論文選擇命中率 | 100.0% |
| 平均 retriever candidate 覆蓋率 | 94.7% |
| 平均 Stage 2 evidence 覆蓋率 | 88.2% |
| 平均總延遲 | 432.2s |
| 平均 planning 延遲 | 29.1s |
| 平均 retrieval 延遲 | 6.3s |
| └ Phase A embed/vector/BM25 | 6.2s |
| └ Phase B 子答案生成 | 0.1s |
| 平均 grounding 延遲 | 15.5s |
| └ 其中 NLI | 0.7s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | correctness | 翻譯忠實度 | 選擇命中 | candidate recall | Stage 2 recall | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|-------------|------------|---------|------------------|----------------|-----------|------|------|
| ✅ | Q01 | single_paper | 1.0 | 1.0 | 100.0% | 100.0% | 100.0% | 1.0 | 361.8s | C0/U0 |
| ✅ | Q02 | single_paper | 0.75 | 1.0 | 100.0% | 100.0% | 100.0% | 1.0 | 441.5s | C0/U0 |
| ⚠️ | Q03 | figure_dependent | 0.5 | 1.0 | 100.0% | 100.0% | 100.0% | 1.0 | 278.7s | C0/U0 |
| ⚠️ | Q04 | table_lookup | 0.5 | 1.0 | 100.0% | 100.0% | 100.0% | 1.0 | 364.1s | C0/U0 |
| ✅ | Q05 | single_paper | 1.0 | 1.0 | 100.0% | 100.0% | 100.0% | 1.0 | 414.9s | C0/U0 |
| ❌ | Q06 | multi_chunk | 0.25 | 1.0 | 100.0% | 80.0% | 100.0% | 1.0 | 389.1s | C0/U0 |
| ⚠️ | Q07 | figure_dependent | 1.0 | 1.0 | 100.0% | 66.7% | 50.0% | 1.0 | 444.1s | C0/U0 |
| ✅ | Q08 | cross_paper | 1.0 | 1.0 | 100.0% | 100.0% | 100.0% | 1.0 | 529.1s | C0/U0 |
| ❌ | Q09 | cross_paper | 0.25 | 1.0 | 100.0% | 100.0% | 57.1% | 1.0 | 1151.0s | C0/U0 |
| ⚠️ | Q10 | contradiction_check | 1.0 | 0.5 | 100.0% | 100.0% | 75.0% | 1.0 | 490.6s | C0/U0 |
| ✅ | Q11 | out_of_scope | 1.0 | N/A | N/A | N/A | N/A | — | 168.1s | C0/U0 |
| ❌ | Q12 | false_premise | 0.25 | N/A | N/A | N/A | N/A | — | 153.7s | C0/U0 |

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
- 延遲：361.8s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
>
> - 總結而言，我們揭示了 JPH203 能夠透過其 preincubation (預孵育) 效應來抑制 LAT1 功能，且該 preincubation 效應能協同增強 co-incubation (共同孵育) 的抑制效果 [Source: 1-s2.0-S1347861320300633-main]。
> - 此外，preincubation 與 co-incubation 效應結合後的 IC50 值低於單獨 co-incubation 抑制的效果 (34.2 ± 3.6 nM vs. 99.2 ± 11.0 nM) [Source: 1-s2.0-S1347861320300633-main]。
> - 如圖 5 所示，雖然 co-incubation 抑制分析中的 IC50 值為 99.2 ± 11.0 nM，但加入 preincubation 顯著增強了其抑制效能 (IC50 = 34.2 ± 3.6 nM) [Source: 1-s2.0-S1347861320300633-main]。
> - 與 IC50 分析結果一致，pre-plus co-incubation 抑制分析中的 Ki 值為 0.37 mM，低於 co-incubation 抑制分析中的數值 (0.46 mM) [Source: 1-s2.0-S1347861320300633-main]。
> - 這些結果表明，JPH203 對於 HT-29 細胞中 LAT1 功能的 preincubation (預孵育) 抑制效應呈濃度與時間依賴性 (concentration- and time-dependent manner) [Source: 1-s2.0-S1347861320300633-main]。
> - 根據結果，IC50 值被確定為 193 ± 50 (nM) [Source: 1 …（完整內容見 JSON）

---

### ✅ Q02 · single_paper

**問題**：What is the hybrid process used for the asymmetric synthesis of optically pure L-p-boronophenylalanine, and what are its key steps?

- detected_paper：`None`
- 選出論文：['bbb0683', 'CMDC-20-e202500059']
- gold_papers：['bbb0683']
- correctness candidate：`answer_for_judge`
- correctness：0.75　raw：4/5　judge：`structured_fact_audit_v1`
- judge reason：covered 5/6; missing F3
- translation fidelity：1.0　raw：5/5　judge：`translation_fidelity_v2`
- translation reason：0 material and 0 minor semantic errors
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：100.0%　grounding：1.0
- 延遲：441.5s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> 用於光學純 L-p-boronophenylalanine 不對稱合成 (asymmetric synthesis) 的混合化學-酶法製程 (hybrid chemo-enzymatic process) 包含一個由特定化學與酶促轉化定義的序列 【bbb0683】。其核心關鍵步驟如下：
>
> lithiated (2R)-(-)-2,5-dihydro-2-isopropyl-3,6-dimethoxypyrazine 與 4-bromomethylbenzeneboronate 的對映選擇性烷基化 (enantioselective alkylation) 以 74% e.e. 產出 adduct 4 【bbb0683】。
> 將 adduct 4 以鹽酸 (hydrochloric acid) 進行分步處理 (stepwise treatment)，得到 L-BPA methyl ester 6 【bbb0683】。
> L-BPA methyl ester 6 使用 chymotrypsin 進行水解 (hydrolysis)，可獲得光學純 L-BPA 【bbb0683】。
> 該化合物的光學純度 (optical purity) 經 HPLC 分析 (HPLC analysis) 在以下條件下確定為 100%：column 【bbb0683】
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
- translation fidelity：1.0　raw：5/5　judge：`translation_fidelity_v2`
- translation reason：0 material and 0 minor semantic errors
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：100.0%　grounding：1.0
- 延遲：278.7s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
>
> - 在我們的結構分析中，我們觀察到 JPH203 結合在傳統的底物結合口袋 (substrate-binding pocket) 內，與先前確定的 LAT1 抑制劑 (inhibitors) 如 Diiodo-Tyr 或 JX-075, JX-078 和 JX-119 相似 (Fig. 1b, c)13 [Source: s41421-024-00697-6]。
> - 頭部 (head) 的 $\alpha$-氨基 ($\alpha$-amino group) 和 $\alpha$-羧基 ($\alpha$-carboxyl group) 在穩定該分子中起著關鍵作用，與 LAT1 中 TM1 和 TM6 解旋區域 (unwound region) 的主鏈原子 (main chain atoms) 形成氫鍵網絡 (hydrogen bond network) (Fig. 1b) [Source: s41421-024-00697-6]。
> - 此外，JPH203 雙鹵代酪氨酸核心 (bi-halogenated tyrosine core) 上的氯原子 (chloride atom) 與 Tyr259 形成鹵鍵 (halogen bond) [Source: s41421-024-00697-6]。
>
> ---
> 📋 **答案品質報告**
>
> 📊 **分段論文依據率：**
>
>   ✅ 【論文直接依據】：100.0%（3/3 句）
>
> ✅ **直引依據率**：100.0%　高（直接引用高度忠實於論文）
>
> ✅ **所有陳述均有論文依據**
>
>
> <!-- grounding_score=1.000 -->
> ---

---

### ⚠️ Q04 · table_lookup

**問題**：What are the main degradation products and impurities of BPA identified by HPLC, and under which storage conditions do they form?

- detected_paper：`None`
- 選出論文：['1-s2.0-S0731708511003451-main', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water']
- gold_papers：['1-s2.0-S0731708511003451-main']
- correctness candidate：`answer_for_judge`
- correctness：0.5　raw：3/5　judge：`structured_fact_audit_v1`
- judge reason：covered 6/7; contradicted F1
- translation fidelity：1.0　raw：5/5　judge：`translation_fidelity_v2`
- translation reason：0 material and 0 minor semantic errors
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：100.0%　grounding：1.0
- 延遲：364.1s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
>
> - BPA 與 tyrosine 的分離在 230 nm 處定量，其保留時間 (retention times) 分別為 5.3 和 4.5 min [Source: 1-s2.0-S0731708511003451-main]。
> - 在強制降解 (forced degradation) 與對照條件 (control conditions) 下，BPA 峰的峰純度值 (peak purity values) >0.997 [Source: 1-s2.0-S0731708511003451-main]。
> - 降解：作為原料粉末 (raw powder)，BPA 是穩定的，在 55 °C 下儲存 6 個月或在 40 °C 下儲存 12 個月時，未產生可檢測到的降解 (detectable degradation) [Source: 1-s2.0-S0731708511003451-main]。
> - 樣品製備與降解分析：BPA 藥品 (BPA drug product) 是透過將 BPA (100 mg/ml) 與 mannitol (110 mg/ml) 冷凍乾燥 (freeze drying) 製得，在 WFI 中將 pH 調節至 8.0 ± 0.1，樣品瓶在黑暗中於 4, 25 和 40 °C 下孵育 (incubated) 數月 [Source: 1-s2.0-S0731708511003451-main]。
> - BPA 強制降解 (forced degradation) 測試使用溶解在 100 mM NaOH, 100 mM HCl 或 5% FeCl3 中的 BPA 進行，且這些樣品在 55 °C 下孵育 24 h。6 mM H2O2 的 BPA 溶液樣品在 HPLC 分析 (HPLC analysis) 前立即製備 [Source: 1-s …（完整內容見 JSON）

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
- 延遲：414.9s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
>
> - 利用苦味酸 (picric acid) 作為溫和且有效的催化劑，此方法可在無溶劑條件下實現芳香胺 (aromatic amines)、雜環芳香胺 (heterocyclic aromatic amines)、開鏈脂肪胺 (open-chain aliphatic amines) 以及環狀脂肪胺 (cyclic aliphatic amines) 的 N-Boc 保護 [Source: 1-s2.0-S2773223124000268-main]。
> - 此方法利用二叔丁基二碳酸酯 (di-tert-butyl dicarbonate, (Boc)2O) 來保護芳香胺與脂肪胺，並使用易於獲取的苦味酸 (picric acid, 2 mol%) 作為 Brønsted 酸催化劑 (Brønsted acid catalyst) [Source: 1-s2.0-S2773223124000268-main]。
> - 此方法可在完全無溶劑且環保的條件下，實現胺類的高產率 (最高達 98 %) 快速保護，從而促進更清潔且更永續的合成過程 [Source: 1-s2.0-S2773223124000268-main]。
> - 基於這些篩選反應條件，我們得出結論：在室溫、無溶劑條件下，使用 0.2 當量 (equivalents) 的苦味酸 7 (picric acid 7) 作為催化劑並反應 60 min，是獲得極高產率產物的最佳條件 (Table 1, Entry 13) [Source: 1-s2.0-S2773223124000268-main]。
> - 此方法的顯著優勢包括：1) 開創性地使用有機酸 (organic acid) 作為此反應的催化劑，2) 在短時間內實現高產率，3) 催化劑操作方便且具成本效益，4) 無副反應 (side reactions …（完整內容見 JSON）

---

### ❌ Q06 · multi_chunk

**問題**：How does poly(vinyl alcohol) enhance the therapeutic effect of 4-L-boronophenylalanine in neutron capture therapy of thoracic tumors? Explain the proposed mechanism with supporting data.

- detected_paper：`None`
- 選出論文：['1-s2.0-S0378517325007926-main', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'CMDC-20-e202500059']
- gold_papers：['1-s2.0-S0378517325007926-main']
- correctness candidate：`answer_for_judge`
- correctness：0.25　raw：2/5　judge：`structured_fact_audit_v1`
- judge reason：covered 2/7; missing F1, F2, F3, F5, F6
- translation fidelity：1.0　raw：5/5　judge：`translation_fidelity_v2`
- translation reason：0 material and 0 minor semantic errors
- 論文選擇命中率：100.0%　candidate 覆蓋率：80.0%　Stage 2 evidence 覆蓋率：100.0%　grounding：1.0
- 延遲：389.1s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ⚠️ **資料可能不足**：檢索內容僅部分涵蓋此問題，以下回答可能不完整，請謹慎參考並自行查證。
>
> ## 【論文直接依據】
>
> - 就此，我們先前報導了一項使用聚乙烯醇 (poly (vinyl alcohol) (PVA)) 來延長 BPA 之腫瘤內滯留時間的簡單技術 (Nomoto et al., 2020) [Source: 1-s2.0-S0378517325007926-main]。
> - 與硼中子捕獲治療 (boron neutron capture therapy) 相關的針狀晶體導致輸尿管損傷而引起的急性腎損傷 [Source: 1-s2.0-S0378517325007926-main]。
> - p-borono- phenylalanine 在體外 (in vitro) 透過細胞膜運輸的機制 [Source: 1-s2.0-S0378517325007926-main]。
> - 用於癌症硼中子捕獲治療 (boron neutron capture therapy) 的 FBPA PET：在大鼠異種移植模型 (rat xenograft model) 中預測腫瘤與正常組織中的 (10)B 濃度 [Source: 1-s2.0-S0378517325007926-main]。
>
> ---
> 📋 **答案品質報告**
>
> 📊 **分段論文依據率：**
>
>   ✅ 【論文直接依據】：100.0%（4/4 句）
>
> ✅ **直引依據率**：100.0%　高（直接引用高度忠實於論文）
>
> ✅ **所有陳述均有論文依據**
>
>
> <!-- grounding_score=1.000 -->
> ---

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
- 延遲：444.1s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
>
> - 總結而言，我們發現了一種水穩定的 boroxine 結構，具有極佳的 pH 穩定性以及能在室溫下進行交換的水兼容 DCBs (dynamic covalent bonds)，從根本上解決了 boroxines 長期以來的水解不穩定性挑戰 [Source: 41467_2024_Article_45464]。
> - 此外，我們開發了一種由水穩定的 boroxines 交聯而成的 hydrogel (水凝膠)，由於 boroxine 結構具有極佳的 pH 穩定性和動態特性，該 hydrogel 展現出高酸鹼穩定性以及可逆的 gel–sol transition (凝膠-溶膠轉變) [Source: 41467_2024_Article_45464]。
> - 我們發現，在 ambient environments (環境條件) 下，2-hydroxyphenylboronic acid 會發生自發脫水，形成具有動態共價鍵和聚集誘導增強發光活性 (aggregation-induced enhanced emission activity) 的 dimer (二聚體) [Source: 41467_2024_Article_45464]。
> - 更有趣的是，在接觸水後，HO‑PBA dimers 在室溫下迅速轉化為 HO‑PBA trimer–H2O complexes (圖 1b)，其中包含水穩定的 boroxine 結構，這一點已通過 nuclear magnetic resonance (NMR) spectroscopy (核磁共振光譜)、mass spectrometry (MS) (質譜)、UV–Raman spectrometry (UV-Raman 光譜) 以及理論計算得到證實 [Source: 41467_2024_Article_4546 …（完整內容見 JSON）

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
- 延遲：529.1s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ⚠️ **資料可能不足**：檢索內容僅部分涵蓋此問題，以下回答可能不完整，請謹慎參考並自行查證。
>
> 比較框架：
> - 路徑 (Route)：【bbb0683】報導了一種涉及對映選擇性烷基化 (enantioselective alkylation) 及其後續酶水解 (enzymatic hydrolysis) 的混合製程 (hybrid process)，可產出光學純 L-BPA 【bbb0683】。
> - 路徑細節 (Route detail)：使用 chymotrypsin 進行水解以產出光學純 L-BPA 【bbb0683】。
> - 回顧/比較來源 (Review/comparison source)：L-BPA 的合成已透過多種路徑進行，反映了生產高純度、同位素富集材料 (high-purity, isotopically enriched material) 的挑戰 【CMDC-20-e202500059】。
> - 回顧維度 (Review dimensions)：該回顧強調了每種方法在可擴展性 (scalability)、成本效益 (cost-effectiveness) 和安全性 (safety) 方面的局限性 【CMDC-20-e202500059】。
>
> 核心權衡（高純度/同位素富集與可擴展性及成本效益）：
> - 高純度/同位素富集：生產高純度、同位素富集的 L-BPA 材料具有挑戰性 【CMDC-20-e202500059】。
> - 可擴展性：該混合合成方法 (hybrid synthesis method) 在操作與後處理的便捷性以及反應步驟較少方面具有優勢 【bbb0683】。
> - 成本效益：在製備同位素富集化合物 (isotopically enriched compounds) 時，主要成本通常來自於同位素起始原料 【CMDC-20-e202500059】。
>  …（完整內容見 JSON）

---

### ❌ Q09 · cross_paper

**問題**：Several papers target LAT1 for cancer therapy. What different therapeutic strategies targeting LAT1 are described across the literature, and how do they differ in mechanism?

- detected_paper：`None`
- 選出論文：['LAT1 ChemComm 2026', '1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 's41421-024-00697-6', 'cas0106-0279']
- gold_papers：['1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 'LAT1 ChemComm 2026', 's41421-024-00697-6']
- correctness candidate：`answer_for_judge`
- correctness：0.25　raw：2/5　judge：`structured_fact_audit_v1`
- judge reason：covered 3/7; missing F1, F2, F5, F6
- translation fidelity：1.0　raw：5/5　judge：`translation_fidelity_v2`
- translation reason：0 material and 0 minor semantic errors
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：57.1%　grounding：1.0
- 延遲：1151.0s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> 比較框架：
> *   [route] JPH203 透過共孵育 (co-incubation) 與預孵育 (preincubation) 的抑制效應來抑制 LAT1 功能 【1-s2.0-S1347861320300633-main】。
> *   [route] 自組裝肽 (Self-assembling peptide) LffVLKK-4Phe 透過針對 LAT1 介導的氨基酸運輸活性來抑制癌細胞增殖 【LAT1 ChemComm 2026】。
>
> 證據：
> *   JPH203 的抑制作用（預孵育 preincubation 與共孵育 co-incubation）涉及與 LAT1–4F2hc 複合體的傳統底物結合口袋 (substrate-binding pocket) 內之結合 【s41421-024-00697-6】。
> *   LffVLKK-4Phe 自組裝肽 (self-assembling peptide) 治療利用了由癌細胞膜上自組裝驅動的多價交互作用 (multivalent interactions) 【LAT1 ChemComm 2026】。
> *   JPH203 可能透過黏附於運輸路徑的 cis-side 或在面向細胞質 (cytosol-facing) 的底物結合位點產生 trans-inhibition 來發揮預孵育效應 【1-s2.0-S1347861320300633-main】。
> *   p-Boronophenylalanine (BPA) 的療效依賴於由包括 LAT1 在內的轉運蛋白介導的選擇性遞送，以在 BNCT 中實現局部核分裂 (nuclear fission) 【cas0106-0279】。
>
> 核心權衡：
> 針對 LAT1 的治療策略涵蓋不同的形式 (modalities)：
> *   可透過 JPH203 與 LffVLKK-4Phe  …（完整內容見 JSON）

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
- translation reason：1 material and 0 minor semantic errors; The source specifies the boronic acid units are incorporated 'into a flexible macrocycle', but T1 omits the word 'flexible' (靈活的) when describing the macrocycle, losing a key structural detail.
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：75.0%　grounding：1.0
- 延遲：490.6s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> Comparison scaffold:
> - Strategy: 【Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation】報導將三個 boronic acid (硼酸) 單元併入一個靈活的 macrocycle (大環分子) 中，藉此增強 boroxines (硼氧環) 對於水解 (hydrolysis) 的穩定性【Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation】。
> - Strategy: 【41467_2024_Article_45464】報導 2-hydroxyphenylboronic acid (2-羥基苯硼酸) 的自發性脫水 (spontaneous dehydration)，隨後接觸水，形成具有優異 pH 穩定性和水兼容動態共價鍵 (water-compatible dynamic covalent bonds) 的 boroxine (硼氧環) 結構【41467_2024_Article_45464】。
> - Mechanism: 【Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation】報導水解抗性 (Hydrolytic resistance) 通過熵因子 (entropic factors) 和 Lewis acidity (路易斯酸度) 的降低而得到增強【Chemistry …（完整內容見 JSON）

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
- 延遲：168.1s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ⚠️ **誠實棄答**：檢索到的內容與此問題相關，但並不包含可直接回答的資訊。為避免編造不存在的數據，本系統選擇不作答。建議換個問法，或確認該主題是否涵蓋於文獻庫中。

---

### ❌ Q12 · false_premise

**問題**：Since BPA is administered orally in BNCT, what oral bioavailability values are reported in these papers?

- detected_paper：`None`
- 選出論文：['cas0106-0279', 'Chem Biol Drug Des - 2013 - Achilli - Folic Acid‐Conjugated 4‐Amino‐Phenylboronate  a Boron‐Containing Compound Designed']
- gold_papers：[]
- correctness candidate：`answer_for_judge`
- correctness：0.25　raw：2/5　judge：`structured_fact_audit_v1`
- judge reason：covered 1/3; missing F1, F2
- translation fidelity：N/A　raw：N/A　judge：`N/A`
- 論文選擇命中率：N/A　candidate 覆蓋率：N/A　Stage 2 evidence 覆蓋率：N/A　grounding：—
- 延遲：153.7s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ⚠️ **誠實棄答**：檢索到的內容與此問題相關，但並不包含可直接回答的資訊。為避免編造不存在的數據，本系統選擇不作答。建議換個問法，或確認該主題是否涵蓋於文獻庫中。

---
