# Eval Report — `baseline_v9_rootfix_smoke`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-23 20:48
- 題數：8

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.656 |
| Correctness judge 覆蓋 | 8/8（N/A 0） |
| 平均翻譯忠實度（LLM-judge） | 0.714 |
| Translation judge 覆蓋 | 7/8（N/A 1） |
| 平均 grounding 分數 | 0.929 |
| 平均論文選擇命中率 | 100.0% |
| 平均 retriever candidate 覆蓋率 | 93.3% |
| 平均 Stage 2 evidence 覆蓋率 | 68.8% |
| 平均總延遲 | 520.3s |
| 平均 planning 延遲 | 7.2s |
| 平均 retrieval 延遲 | 5.6s |
| └ Phase A embed/vector/BM25 | 5.5s |
| └ Phase B 子答案生成 | 0.1s |
| 平均 grounding 延遲 | 15.9s |
| └ 其中 NLI | 1.2s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | correctness | 翻譯忠實度 | 選擇命中 | candidate recall | Stage 2 recall | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|-------------|------------|---------|------------------|----------------|-----------|------|------|
| ❌ | Q02 | single_paper | 0.25 | 1.0 | 100.0% | 100.0% | 75.0% | 1.0 | 356.9s | C0/U0 |
| ✅ | Q03 | figure_dependent | 0.75 | 1.0 | 100.0% | 100.0% | 100.0% | 1.0 | 373.8s | C0/U0 |
| ⚠️ | Q05 | single_paper | 0.75 | N/A | 100.0% | 100.0% | 75.0% | 0.923 | 394.2s | C0/U1 |
| ✅ | Q06 | multi_chunk | 0.75 | 1.0 | 100.0% | 80.0% | 80.0% | 0.938 | 422.5s | C0/U1 |
| ⚠️ | Q07 | figure_dependent | 0.5 | 0.5 | 100.0% | 66.7% | 50.0% | 1.0 | 399.4s | C0/U0 |
| ❌ | Q08 | cross_paper | 0.75 | 0.0 | 100.0% | 100.0% | 66.7% | 1.0 | 594.7s | C0/U0 |
| ❌ | Q09 | cross_paper | 0.5 | 0.5 | 100.0% | 100.0% | 28.6% | 0.857 | 818.0s | C0/U1 |
| ⚠️ | Q10 | contradiction_check | 1.0 | 1.0 | 100.0% | 100.0% | 75.0% | 0.714 | 803.0s | C0/U2 |

## 逐題細節

### ❌ Q02 · single_paper

**問題**：What is the hybrid process used for the asymmetric synthesis of optically pure L-p-boronophenylalanine, and what are its key steps?

- detected_paper：`None`
- 選出論文：['bbb0683', 'CMDC-20-e202500059']
- gold_papers：['bbb0683']
- correctness candidate：`answer_for_judge`
- correctness：0.25　raw：2/5　judge：`structured_fact_audit_v1`
- judge reason：covered 2/6; missing F2, F3, F4, F5
- translation fidelity：1.0　raw：5/5　judge：`translation_fidelity_v2`
- translation reason：0 material and 0 minor semantic errors
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：75.0%　grounding：1.0
- 延遲：356.9s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> 方法依據：
> - 60 (4), 683684, 1996 Note Asymmetric Synthesis of Optically Pure L-p-Boronophenylalanine by a Hybrid Processt Hidekazu NAKAO, Tsuguhiro MORIMOTO, and Mitsunori KIRIHATA tt Department of Applied Biochemistry, College of Agriculture, University of Osaka Prefecture, 1-1 Gakuencho, Sakai, Osaka 593, Japan Received October 16, 1995。光學純的 L-p-boronophenylalanine (BPA) 是透過一個包含對映選擇性烷基化 (enantioselective alkylation) 以及隨後的酶水解 (enzymatic hydrolysis) 的混合製程合成的 [bbb0683]。
> - 將 4 以 hydrochloric acid (鹽酸) 逐步處理得到 L-BPA methyl ester 6，隨後使用 chymotrypsin 進行水解以製得光學純的 L-BPA [bbb0683]。
> - 另一方面，在 pH 5.0 下使用 chymotrypsin 對 6 進行酶水解 (enzymatic hydrolysis)，以 79% 的產率製得光學純的 L-BPA，mp 275-283 °C (decomp.)，[(X]~8 - 8.6 (c 0.49, 0.1 N HCl) [lit.,41 [tx]53 - 8.2( (c 0.7, 0.1 N HCl)] [bbb0683]。
> - 在 5 °C 下，向 THF (8 mL) 溶 …（完整內容見 JSON）

---

### ✅ Q03 · figure_dependent

**問題**：According to the cryo-EM structure, how does JPH203 bind to and inhibit the LAT1-4F2hc complex?

- detected_paper：`None`
- 選出論文：['s41421-024-00697-6', '1-s2.0-S1347861320300633-main']
- gold_papers：['s41421-024-00697-6']
- correctness candidate：`answer_for_judge`
- correctness：0.75　raw：4/5　judge：`structured_fact_audit_v1`
- judge reason：covered 5/6; missing F1
- translation fidelity：1.0　raw：5/5　judge：`translation_fidelity_v2`
- translation reason：0 material and 0 minor semantic errors
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：100.0%　grounding：1.0
- 延遲：373.8s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
>
> - 經過幾輪 3D 精細化（3D refinement）後，我們成功地在 3.30 Å 的解析度下揭示了結合 JPH203 的 LAT1–4F2hc 複合物的面向外結構（outward-facing structure）（圖 1a；補充圖 [Source: s41421-024-00697-6]）。
> - 在我們的結構分析中，我們觀察到 JPH203 結合在傳統的底物結合口袋（substrate-binding pocket）內，與先前確定的 LAT1 抑制劑如 Diiodo-Tyr 或 JX-075、JX-078 和 JX-119 相似（圖 1b, c）[Source: s41421-024-00697-6]。
> - 頭部的 $\alpha$-氨基 ($\alpha$-amino group) 和 $\alpha$-羧基 ($\alpha$-carboxyl group) 在穩定分子方面起著關鍵作用，與 LAT1 中 TM1 和 TM6 解旋區域（unwound region）的主鏈原子形成氫鍵網絡（hydrogen bond network）（圖 1b）[Source: s41421-024-00697-6]。
> - 此外，JPH203 的雙鹵代酪氨酸核心（bi-halogenated tyrosine core）上的氯原子與 Tyr259 形成了鹵鍵（halogen bond）[Source: s41421-024-00697-6]。
> - 疏水性尾部片段（hydrophobic tail moiety）緊密地貼合在一個由特定氨基酸形成的疏水口袋（hydrophobic pocket）中，這些氨基酸包括：TM1 上的 Ile63、Ile64 和 Ile68；TM3 上的 Ile140、Ile147、Ile148；TM6 上的 Phe252 和 Gly …（完整內容見 JSON）

---

### ⚠️ Q05 · single_paper

**問題**：What is the solvent-free green protocol for N-Boc protection of amines using picric acid as a catalyst, and what reaction conditions are used?

- detected_paper：`None`
- 選出論文：['1-s2.0-S2773223124000268-main']
- gold_papers：['1-s2.0-S2773223124000268-main']
- correctness candidate：`answer_for_judge`
- correctness：0.75　raw：4/5　judge：`structured_fact_audit_v1`
- judge reason：covered 6/7; missing F4
- translation fidelity：N/A　raw：N/A　judge：`translation_fidelity_v2`
- translation reason：invalid translation audit: top-level 'errors' must be a list
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：75.0%　grounding：0.923
- 延遲：394.2s　問題標記：{'conflicts': 0, 'unsupported': 1}

**答案預覽**：

> ## 【論文直接依據】
>
> - 使用 picric acid 作為溫和且有效的催化劑，此方法可在無溶劑條件下實現芳香胺 (aromatic amines)、雜環芳香胺 (heterocyclic aromatic amines)、開鏈脂肪胺 (open-chain aliphatic amines) 及環狀脂肪胺 (cyclic aliphatic amines) 的 N-Boc 保護 [Source: 1-s2.0-S2773223124000268-main]。
> - 此方法利用 di-tert-butyl dicarbonate ((Boc)2O) 來保護芳香胺 (aromatic amines) 與脂肪胺 (aliphatic amines)，並以易於取得的 picric acid (2 mol%) 作為 Brønsted acid catalyst [Source: 1-s2.0-S2773223124000268-main]。
> - 此方法可在完全無溶劑且環保的條件下，實現胺類的高產率（最高達 98 %）快速保護，從而促進更清潔且更永續的合成過程 [Source: 1-s2.0-S2773223124000268-main]。
> - 雖然在沒有催化劑的情況下反應仍能進行，但在 48 h 後僅獲得 55 % 的產物 (Table 1, Entry 23) [Source: 1-s2.0-S2773223124000268-main]。
> - 基於這些篩選反應條件，我們得出結論：在室溫、無溶劑條件下使用 0.2 當量 (equivalents) 的 picric acid 7 作為催化劑並反應 60 min，是獲得極高產率產物的最佳條件 (Table 1, Entry 13) [Source: 1-s2.0-S2773223124000268-main]。
> - 使用 picr …（完整內容見 JSON）

---

### ✅ Q06 · multi_chunk

**問題**：How does poly(vinyl alcohol) enhance the therapeutic effect of 4-L-boronophenylalanine in neutron capture therapy of thoracic tumors? Explain the proposed mechanism with supporting data.

- detected_paper：`None`
- 選出論文：['1-s2.0-S0378517325007926-main', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'CMDC-20-e202500059']
- gold_papers：['1-s2.0-S0378517325007926-main']
- correctness candidate：`answer_for_judge`
- correctness：0.75　raw：4/5　judge：`structured_fact_audit_v1`
- judge reason：covered 6/7; missing F6
- translation fidelity：1.0　raw：5/5　judge：`translation_fidelity_v2`
- translation reason：0 material and 0 minor semantic errors
- 論文選擇命中率：100.0%　candidate 覆蓋率：80.0%　Stage 2 evidence 覆蓋率：80.0%　grounding：0.938
- 延遲：422.5s　問題標記：{'conflicts': 0, 'unsupported': 1}

**答案預覽**：

> ## 【論文直接依據】
>
> - 我們在此報告，由 poly(vinyl alcohol) (PVA, 聚乙烯醇) 與 BNCT 中唯一經臨床批准的藥物 4-L-boronophenylalanine (BPA, 4-L-硼苯丙氨酸) 組成的複合物，對小鼠體內模擬難治性 malignant pleural mesothelioma (MPM, 惡性胸膜間皮瘤) 的胸腔腫瘤可顯示出顯著的治療效率 [Source: 1-s2.0-S0378517325007926-main]。
> - 在這方面，我們之前報導了一項使用 poly(vinyl alcohol) (PVA, 聚乙烯醇) 來延長 BPA 在腫瘤內滯留時間 (intratumoral retention time) 的簡單技術 (Nomoto et al., 2020) [Source: 1-s2.0-S0378517325007926-main]。
> - PVA 通過 boronate esters (硼酸酯) 與多個 BPA 分子形成複合物，保留了針對 LAT1 的關鍵 phenylalanine-like structure (類苯丙氨酸結構) (圖 1C)，且該複合物的形成已通過 NMR (核磁共振) 確認 [Source: 1-s2.0-S0378517325007926-main]。
> - 被稱為 PVA-BPA 的複合物與 LAT1 相互作用，並通過 LAT1-mediated endocytosis (LAT1 介導的內吞作用) 被攝入癌細胞中，且定位於 endo-/lysosomes (內體/溶酶體)，從而增加了細胞攝取量並延緩了 BPA 從 cytosol (細胞質) 的流出 (圖 1D) [Source: 1-s2.0-S0378517325007926-main]。
> - PVA-BPA 可在 1 h 內通過 …（完整內容見 JSON）

---

### ⚠️ Q07 · figure_dependent

**問題**：What is the water-stable boroxine structure reported, and what role do the dynamic covalent bonds play in its fluoride binding and hydrogel formation?

- detected_paper：`None`
- 選出論文：['41467_2024_Article_45464', 'Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation']
- gold_papers：['41467_2024_Article_45464']
- correctness candidate：`answer_for_judge`
- correctness：0.5　raw：3/5　judge：`structured_fact_audit_v1`
- judge reason：covered 4/8; missing F3, F6, F7, F8
- translation fidelity：0.5　raw：3/5　judge：`translation_fidelity_v2`
- translation reason：1 material and 0 minor semantic errors; The source sentence ends with 'Supplementary Fig' (referring to a figure), but the target translation cuts off at '補充圖', omitting the specific figure identifier or reference number that follows.
- 論文選擇命中率：100.0%　candidate 覆蓋率：66.7%　Stage 2 evidence 覆蓋率：50.0%　grounding：1.0
- 延遲：399.4s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
>
> - 有趣的是，在接觸水後，二聚體 (dimer) 會迅速轉化為具有優異 pH 穩定性和水兼容動態共價鍵 (water-compatible dynamic covalent bonds) 的 boroxine 結構 【Source: 41467_2024_Article_45464】。
> - 基於這些發現，我們報導了 boroxines 在水相介質 (aqueous media) 中對氟離子 (fluoride ions, F−) 的強結合能力，並開發了一種具有高酸鹼穩定性和可逆凝膠-溶膠轉變 (reversible gel–sol transition) 的基於 boroxine 的水凝膠 (boroxine-based hydrogel) 【Source: 41467_2024_Article_45464】。
> - 這種水穩定 boroxine 結構的發現打破了 boroxines 不適用於水環境的限制，開啟了 boroxine 化學研究的新時代 【Source: 41467_2024_Article_45464】。
> - 討論：總結來說，我們發現了一種水穩定 boroxine 結構，它具有優異的 pH 穩定性和可在室溫下進行交換的水兼容 DCBs (動態共價鍵)，從根本上解決了 boroxines 長期以來的水解不穩定性 (hydrolytic instability) 挑戰 【Source: 41467_2024_Article_45464】。
> - 憑藉這一發現，我們實現了 boroxines 在水相介質中對 F− 的選擇性識別，其親和力顯著強於廣泛使用的 F− 受體 PBA，這為開發用於 F− 檢測和分離的先進材料與技術提供了途徑 【Source: 41467_2024_Article_45464】。
> - 此外，我們開發了一種由水穩定 boro …（完整內容見 JSON）

---

### ❌ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- correctness candidate：`answer_for_judge`
- correctness：0.75　raw：4/5　judge：`structured_fact_audit_v1`
- judge reason：covered 6/7; missing F6
- translation fidelity：0.0　raw：1/5　judge：`translation_fidelity_v2`
- translation reason：7 material and 0 minor semantic errors; The source sentence explicitly cites the reference '[bbb0683]' at the end, which is omitted in the target translation.
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：66.7%　grounding：1.0
- 延遲：594.7s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> 比較框架：
> - 路徑：【bbb0683】報導了一種涉及 enantioselective alkylation and subsequent enzymatic hydrolysis（對映選擇性烷基化及隨後的酶水解）的混合製程，可產生光學純的 L-BPA 【bbb0683】。
> - 評論/比較來源：【CMDC-20-e202500059】報導了合成 4-borono-L-phenylalanine (L-BPA) 已透過多種路徑實現 【CMDC-20-e202500059】。
> - 評論維度：該評論強調了每種方法在成本效益 (cost-effectiveness)、可擴展性 (scalability) 和安全性 (safety) 方面的局限性 【CMDC-20-e202500059】。
>
> 核心權衡（高純度/同位素富集與可擴展性及成本效益）：
> - 高純度/同位素富集：10B 通常透過化學交換蒸餾 (chemical exchange distillation) 從 11B 中分離，並轉化為硼酸酯 (boronic esters) 或水解為 10B 硼酸 (10B boric acid) 【CMDC-20-e202500059】。
> - 可擴展性：混合合成法 (hybrid synthetic method) 在操作與後處理 (workup) 的簡便性以及反應步驟少方面具有優勢 【bbb0683】。
> - 成本效益：主要成本通常來自於同位素起始原料 (isotope starting material) 【CMDC-20-e202500059】。
> - 成本效益：10B 的高成本是普通硼酸的 1000 倍以上 【CMDC-20-e202500059】。
>
> ---
> 📋 **答案品質報告**
>
> ✅ **整體論文依據率**：100.0%　高（答案高度忠實於論文內容）
>
> ✅ **所有陳述均有論文 …（完整內容見 JSON）

---

### ❌ Q09 · cross_paper

**問題**：Several papers target LAT1 for cancer therapy. What different therapeutic strategies targeting LAT1 are described across the literature, and how do they differ in mechanism?

- detected_paper：`None`
- 選出論文：['LAT1 ChemComm 2026', '1-s2.0-S0378517325007926-main', '1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 's41421-024-00697-6']
- gold_papers：['1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 'LAT1 ChemComm 2026', 's41421-024-00697-6']
- correctness candidate：`answer_for_judge`
- correctness：0.5　raw：3/5　judge：`structured_fact_audit_v1`
- judge reason：covered 5/7; missing F4, F5
- translation fidelity：0.5　raw：3/5　judge：`translation_fidelity_v2`
- translation reason：1 material and 0 minor semantic errors; The source specifies that LAT1 is the primary transporter 'compared to ATB0,+ and LAT2' (implying a comparison of uptake efficiency or affinity for BPA). The target translation omits the specific context of what these other transporters are being compared against regarding their role in BPA uptake, 
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：28.6%　grounding：0.857
- 延遲：818.0s　問題標記：{'conflicts': 0, 'unsupported': 1}

**答案預覽**：

> 比較框架：
> - 策略：【1-s2.0-S1347861320300633-main】報告了 JPH203 給藥（共同孵育 co-incubation 與預孵育 preincubation），可抑制 HT-29 細胞中不依賴 Na+ 的 leucine 攝取活性 [1-s2.0-S1347861320300633-main]。
> - 策略：【1-s2.0-S1936523321002047-main】報告了 JPH203 與 X-irradiation（X 射線照射）的聯合應用，通過 mTOR 下調 (downregulation) 增強細胞衰老，且不降低 ATP 和 GSH 水平 [1-s2.0-S1936523321002047-main]。
> - 策略：【LAT1 ChemComm 2026】報告了 LffVLKK-4Phe 自組裝肽 (self-assembling peptide) 處理，可抑制 LAT1 介導的氨基酸運輸（早期）以及細胞膜破裂 (cell membrane disruption)（後期）[LAT1 ChemComm 2026]。
> - 策略：【cas0106-0279】報告了 p-Boronophenylalanine (BPA) 遞送隨後進行中子束照射 (neutron beam irradiation)，通過產生 alpha 粒子和鋰核的核捕獲 (nuclear capture) 與裂變反應 (fission reactions) 實現選擇性細胞死亡 [cas0106-0279]。
> - 機制：【1-s2.0-S1347861320300633-main】報告了 JPH203 預孵育 (preincubation) 可能通過 cis-side 黏附、在面向細胞質側的 trans-inhibition（反式抑制），或影響細胞膜定位 (plasma mem …（完整內容見 JSON）

---

### ⚠️ Q10 · contradiction_check

**問題**：Do the papers agree on the hydrolytic stability of boroxine structures? Describe any agreements or differences between them.

- detected_paper：`None`
- 選出論文：['Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation', '41467_2024_Article_45464', 'a-method-for-the-deprotection-of-alkylpinacolyl-boronate-esters']
- gold_papers：['41467_2024_Article_45464', 'Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation']
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 4/4
- translation fidelity：1.0　raw：5/5　judge：`translation_fidelity_v2`
- translation reason：0 material and 0 minor semantic errors
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：75.0%　grounding：0.714
- 延遲：803.0s　問題標記：{'conflicts': 0, 'unsupported': 2}

**答案預覽**：

> 比較框架：
> - 策略：【Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation】報告將三個 boronic acid (硼酸) 單元整合到一個靈活的 macrocycle (大環) 中，從而增強了 boroxines (硼氧環) 對於水解的穩定性【Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation】。
> - 策略：【41467_2024_Article_45464】報告了 2-hydroxyphenylboronic acid (HO-PBA) 的自發脫水，隨後接觸水，形成具有優異 pH 穩定性和水兼容動態共價鍵的 boroxine (硼氧環) 結構【41467_2024_Article_45464】。
> - 機制：【Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation】報告水解抗性 (Hydrolytic resistance) 通過熵因子 (entropic factors) 和 Lewis acidity (路易斯酸度) 的降低而增強【Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation】。
> - 回顧/比較來源：【Chemistry A …（完整內容見 JSON）

---
