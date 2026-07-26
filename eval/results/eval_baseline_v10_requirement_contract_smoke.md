# Eval Report — `baseline_v10_requirement_contract_smoke`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-24 01:27
- 題數：8

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.906 |
| Correctness judge 覆蓋 | 8/8（N/A 0） |
| 平均翻譯忠實度（LLM-judge） | 0.607 |
| Translation judge 覆蓋 | 7/8（N/A 1） |
| 平均 grounding 分數 | 0.964 |
| 平均論文選擇命中率 | 100.0% |
| 平均 retriever candidate 覆蓋率 | 93.3% |
| 平均 Stage 2 evidence 覆蓋率 | 81.1% |
| 平均總延遲 | 542.7s |
| 平均 planning 延遲 | 29.1s |
| 平均 retrieval 延遲 | 5.6s |
| └ Phase A embed/vector/BM25 | 5.5s |
| └ Phase B 子答案生成 | 0.0s |
| 平均 grounding 延遲 | 24.0s |
| └ 其中 NLI | 0.6s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | correctness | 翻譯忠實度 | 選擇命中 | candidate recall | Stage 2 recall | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|-------------|------------|---------|------------------|----------------|-----------|------|------|
| ❌ | Q02 | single_paper | 0.75 | 0.0 | 100.0% | 100.0% | 100.0% | 1.0 | 527.3s | C0/U0 |
| ⚠️ | Q03 | figure_dependent | 1.0 | 0.5 | 100.0% | 100.0% | 100.0% | 1.0 | 411.4s | C0/U0 |
| ✅ | Q05 | single_paper | 1.0 | 1.0 | 100.0% | 100.0% | 100.0% | 1.0 | 505.1s | C0/U0 |
| ❌ | Q06 | multi_chunk | 1.0 | 0.25 | 100.0% | 80.0% | 100.0% | 1.0 | 420.5s | C0/U0 |
| ⚠️ | Q07 | figure_dependent | 0.75 | N/A | 100.0% | 66.7% | 50.0% | 1.0 | 497.6s | C0/U0 |
| ⚠️ | Q08 | cross_paper | 1.0 | 1.0 | 100.0% | 100.0% | 66.7% | 0.857 | 613.2s | C0/U1 |
| ⚠️ | Q09 | cross_paper | 0.75 | 0.5 | 100.0% | 100.0% | 57.1% | 0.857 | 842.4s | C0/U1 |
| ✅ | Q10 | contradiction_check | 1.0 | 1.0 | 100.0% | 100.0% | 75.0% | 1.0 | 523.8s | C0/U0 |

## 逐題細節

### ❌ Q02 · single_paper

**問題**：What is the hybrid process used for the asymmetric synthesis of optically pure L-p-boronophenylalanine, and what are its key steps?

- detected_paper：`None`
- 選出論文：['bbb0683', 'CMDC-20-e202500059']
- gold_papers：['bbb0683']
- correctness candidate：`answer_for_judge`
- correctness：0.75　raw：4/5　judge：`structured_fact_audit_v1`
- judge reason：covered 5/6; missing F5
- translation fidelity：0.0　raw：1/5　judge：`translation_fidelity_v2`
- translation reason：4 material and 0 minor semantic errors; The source sentence S6 explicitly states the synthesis is based on alkylation of compound 3 with derivative 2 and subsequent hydrolysis. The target T7 omits this specific chemical basis entirely, instead starting with a completely different subject ('We now describe...') which corresponds to no cont
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：100.0%　grounding：1.0
- 延遲：527.3s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> Method evidence:
> - 鋰化的 (Lithiated) (2R)-( - )-2,5-dihydro- 2-isoprop}'1-3,6-dimethoxypyrazine 與 4-bromo-methylbenzeneboronate (2) 反應，以 74% e.e. 產出加成物 (adduct) 4 【bbb0683】。
> - 將 4 以鹽酸 (hydrochloric acid) 逐步處理得到 L-BP A methyl ester 6，隨後使用 chymotrypsin 進行水解以製得光學純的 (optically pure) L-BPA 【bbb0683】。
> - 將 4 以 0.1 N 鹽酸 (hydrochloric acid) 處理，得到 D-valine methyl ester (D-Val-OMe) 與具有所需立體化學 (stereochemistry) 的 amino acid methyl ester 5 之等莫量混合物 (equimolar mixture)，隨後透過蒸餾 (distillation) 去除 D-Val-OMe 以獲得粗產物 (crude) 5，接著進一步以 1 N 鹽酸 (hydrochloric acid) 處理進行脫保護 (deprotected)，從 4 出發以 85% 的產率得到 L-BPA methyl ester 6 【bbb0683】。
> - 另一方面，在 pH 5.0 下使用 chymo- trypsin 5 ) 對 6 進行酶水解 (enzymatic hydrolysis)，以 79% 的產率製得光學純的 L-BPA，mp 275-283-'C (decomp.)，[(X]~8 - 8.6 (c 0.49, 0.1 N HCI) [lit.,41 [tx]53 - 8.2( (c 0.7, 0.1  …（完整內容見 JSON）

---

### ⚠️ Q03 · figure_dependent

**問題**：According to the cryo-EM structure, how does JPH203 bind to and inhibit the LAT1-4F2hc complex?

- detected_paper：`None`
- 選出論文：['s41421-024-00697-6', '1-s2.0-S1347861320300633-main']
- gold_papers：['s41421-024-00697-6']
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 6/6
- translation fidelity：0.5　raw：3/5　judge：`translation_fidelity_v2`
- translation reason：1 material and 0 minor semantic errors; The source sentence S8 contains the phrase 'TM3 Received: 2 February 2024 Accepted: 31 May 2024 www.nature.com/celldisc Cell Discovery 1234567890();,:' which appears to be metadata/artifacts inserted into the text. The target sentence T8 omits this entire segment ('TM3 Received...'). While likely an
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：100.0%　grounding：1.0
- 延遲：411.4s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
>
> - 在我們的結構分析中，我們觀察到 JPH203 結合在傳統的基質結合口袋 (substrate-binding pocket) 內，與先前確定的 LAT1 抑制劑如 Diiodo-Tyr 或 JX-075、JX-078 和 JX-119 相似 (Fig. 1b, c)13 [Source: s41421-024-00697-6]。
> - 頂端 (head) 的 $\alpha$-氨基 ($\alpha$-amino group) 和 $\alpha$-羧基 ($\alpha$-carboxyl group) 在穩定分子方面起著關鍵作用，與 LAT1 中 TM1 和 TM6 展開區域 (unwound region) 的主鏈原子形成氫鍵網絡 (hydrogen bond network) (Fig. 1b) [Source: s41421-024-00697-6]。
> - 此外，JPH203 的雙鹵代酪氨酸核心 (bi-halogenated tyrosine core) 上的氯原子與 Tyr259 形成鹵鍵 (halogen bond) [Source: s41421-024-00697-6]。
> - 疏水性尾部片段 (hydrophobic tail moiety) 緊密地擬合在一個由特定氨基酸構成的疏水口袋 (hydrophobic pocket) 中，這些氨基酸包括：TM1 上的 Ile63、Ile64 和 Ile68；TM3 上的 Ile140、Ile147、Ile148；TM6 上的 Phe252 和 Gly255；以及 TM10 上的 Ile397、Phe400 和 Phe403 (Fig. 1c) [Source: s41421-024-00697-6]。
> - 具體而言，尾部的氨基與 TM10 上 Phe400 的主鏈形成氫鍵 [Sour …（完整內容見 JSON）

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
- 延遲：505.1s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
>
> - 利用 picric acid (苦味酸) 作為溫和且有效的催化劑，此方法能夠在無溶劑條件 (solvent-free conditions) 下對芳香胺 (aromatic amines)、雜環芳香胺 (heterocyclic aromatic amines)、開鏈脂肪胺 (open-chain aliphatic amines) 以及環狀脂肪胺 (cyclic aliphatic amines) 進行 N-Boc 保護 [Source: 1-s2.0-S2773223124000268-main]。
> - 此方法利用 di-tert-butyl dicarbonate ((Boc)2O) 來保護芳香胺與脂肪胺 (aromatic and aliphatic amines)，並使用易於取得的 picric acid (2 mol%) 作為 Brønsted 酸催化劑 (Brønsted acid catalyst) [Source: 1-s2.0-S2773223124000268-main]。
> - 此方法能夠在完全無溶劑且環保的條件 (completely solvent-free and eco-friendly conditions) 下，實現胺類的高產率 (high-yield, 最高達 98 %) 快速保護，從而促進更清潔且更永續的合成過程 [Source: 1-s2.0-S2773223124000268-main]。
> - 在此背景下，我們開發了一種環保且無溶劑的綠色胺保護方法 (environmentally friendly and solvent-free green approach for amine protection)，使用 picric acid 作為有機催化劑 (organic catalyst) 以生產具有良好 …（完整內容見 JSON）

---

### ❌ Q06 · multi_chunk

**問題**：How does poly(vinyl alcohol) enhance the therapeutic effect of 4-L-boronophenylalanine in neutron capture therapy of thoracic tumors? Explain the proposed mechanism with supporting data.

- detected_paper：`None`
- 選出論文：['1-s2.0-S0378517325007926-main', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'CMDC-20-e202500059']
- gold_papers：['1-s2.0-S0378517325007926-main']
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 7/7
- translation fidelity：0.25　raw：2/5　judge：`translation_fidelity_v2`
- translation reason：2 material and 0 minor semantic errors; The source says 'could show drastic therapeutic efficiency', indicating a potential or observed capability. The target uses '可顯示出極高的治療效率' which is closer, but the combination of 'drastic' (change) vs 'extremely high' (magnitude) remains an error as noted above.
- 論文選擇命中率：100.0%　candidate 覆蓋率：80.0%　Stage 2 evidence 覆蓋率：100.0%　grounding：1.0
- 延遲：420.5s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
>
> - 在此，我們報告由聚乙烯醇 (poly(vinyl alcohol) (PVA)) 與 4-L-硼苯丙氨酸 (4-L-boronophenylalanine (BPA))（BNCT 中唯一經臨床批准的藥物）組成的複合物，對小鼠體內模擬難治性惡性胸膜間皮瘤 (malignant pleural mesothelioma (MPM)) 的胸腔腫瘤可顯示出極高的治療效率 [Source: 1-s2.0-S0378517325007926-main]。
> - 系統性給藥的 PVA-sorbitol-BPA 在目標腫瘤內顯示出高效的累積與長期滯留 (long-term retention) [Source: 1-s2.0-S0378517325007926-main]。
> - 與傳統 BPA 相比，使用 PVA-sorbitol-BPA 的 BNCT 顯著提高了生存率 [Source: 1-s2.0-S0378517325007926-main]。
> - 在這方面，我們之前報告了一項使用聚乙烯醇 (poly(vinyl alcohol) (PVA)) 來延長 BPA 在腫瘤內滯留時間 (intratumoral retention time) 的簡單技術 (Nomoto et al., 2020) [Source: 1-s2.0-S0378517325007926-main]。
> - PVA 通過硼酸酯 (boronate esters) 與多個 BPA 分子形成複合物，保留了用於靶向 LAT1 的必需類苯丙氨酸結構 (phenylalanine-like structure)（圖 1C），且該複合物的形成已通過 NMR 確認 [Source: 1-s2.0-S0378517325007926-main]。
> - 被稱為 PVA-BPA 的複合物與 LAT1 相互作 …（完整內容見 JSON）

---

### ⚠️ Q07 · figure_dependent

**問題**：What is the water-stable boroxine structure reported, and what role do the dynamic covalent bonds play in its fluoride binding and hydrogel formation?

- detected_paper：`None`
- 選出論文：['41467_2024_Article_45464', 'Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation']
- gold_papers：['41467_2024_Article_45464']
- correctness candidate：`answer_for_judge`
- correctness：0.75　raw：4/5　judge：`structured_fact_audit_v1`
- judge reason：covered 7/8; missing F3
- translation fidelity：N/A　raw：N/A　judge：`translation_fidelity_v2`
- translation reason：invalid translation audit: top-level 'errors' must be a list
- 論文選擇命中率：100.0%　candidate 覆蓋率：66.7%　Stage 2 evidence 覆蓋率：50.0%　grounding：1.0
- 延遲：497.6s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
>
> - 有趣的是，在接觸水後，二聚體 (dimer) 迅速轉化為具有優異 pH 穩定性和水兼容動態共價鍵 (water-compatible dynamic covalent bonds) 的 boroxine 結構 【Source: 41467_2024_Article_45464】。
> - 討論總結：我們發現了一種水穩定的 boroxine 結構，具有優異的 pH 穩定性和能在室溫下進行交換的水兼容動態共價鍵 (water-compatible DCBs)，從根本上解決了 boroxines 長期以來的水解不穩定性挑戰 【Source: 41467_2024_Article_45464】。
> - 憑藉這一發現，我們實現了 boroxines 在水相介質 (aqueous media) 中對 F− 的選擇性識別，其親和力顯著強於廣泛使用的 F− 受體 PBA，這為開發用於 F− 檢測和分離的先進材料與技術提供了途徑 【Source: 41467_2024_Article_45464】。
> - 此外，我們開發了一種由水穩定 boroxines 交聯的水凝膠 (hydrogel)，由於 boroxine 結構的優異 pH 穩定性和動態特性，該水凝膠表現出高酸鹼穩定性和可逆的凝膠-溶膠轉變 (reversible gel–sol transition) 【Source: 41467_2024_Article_45464】。
> - 在此，我們揭示了 HO‑PBA 三聚體 (trimer) 中的 B–O 鍵是水兼容的動態共價鍵 (water-compatible DCBs)，為 DCBs 家族提供了一個重要的成員 【Source: 41467_2024_Article_45464】。
> - 更有趣的是，在接觸水後，HO‑PBA 二聚體 (dimers) 在室溫下迅速轉化 …（完整內容見 JSON）

---

### ⚠️ Q08 · cross_paper

**問題**：Compare the different synthetic routes to 4-borono-L-phenylalanine reported across the papers, focusing on isotopic enrichment, scalability, and cost-effectiveness.

- detected_paper：`None`
- 選出論文：['CMDC-20-e202500059', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'bbb0683', 'cas0106-0279', '1-s2.0-S0378517325007926-main']
- gold_papers：['CMDC-20-e202500059', 'bbb0683']
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 7/7
- translation fidelity：1.0　raw：5/5　judge：`translation_fidelity_v2`
- translation reason：0 material and 0 minor semantic errors
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：66.7%　grounding：0.857
- 延遲：613.2s　問題標記：{'conflicts': 0, 'unsupported': 1}

**答案預覽**：

> ⚠️ **資料可能不足**：檢索內容僅部分涵蓋此問題，以下回答可能不完整，請謹慎參考並自行查證。
>
> 比較框架：
> - 路徑：【bbb0683】報告了一種涉及對映選擇性烷基化 (enantioselective alkylation) 及隨後酶水解 (enzymatic hydrolysis) 的混合製程 (hybrid process)，可產生光學純 L-BPA 【bbb0683】。
> - 評論/比較來源：【CMDC-20-e202500059】報告指出，L-BPA 的合成 (Synthesis of L-BPA) 在同位素富集 (isotopic enrichment)、材料成本以及去保護過程中的製程安全性 (process safety during deprotection) 方面面臨挑戰 【CMDC-20-e202500059】。
> - 評論維度：該評論強調了每種方法在成本效益 (cost-effectiveness)、可擴展性 (scalability) 和安全性 (safety) 方面的局限性 【CMDC-20-e202500059】。
>
> 核心權衡（高純度/同位素富集與可擴展性及成本效益）：
> - 高純度/同位素富集：10B 通常透過化學交換蒸餾 (chemical exchange distillation) 從 11B 中分離，然後轉化為硼酸酯 (boronic esters) 或水解為 10B 硼酸 (10B boric acid) 【CMDC-20-e202500059】。
> - 可擴展性：混合製程 (hybrid process) 在操作與後處理的便捷性 (ease of performance and workup) 以及反應步驟較少方面具有優勢 【bbb0683】。
> - 成本效益：同位素富集化合物 (isotopically enriched compoun …（完整內容見 JSON）

---

### ⚠️ Q09 · cross_paper

**問題**：Several papers target LAT1 for cancer therapy. What different therapeutic strategies targeting LAT1 are described across the literature, and how do they differ in mechanism?

- detected_paper：`None`
- 選出論文：['LAT1 ChemComm 2026', '1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 's41421-024-00697-6', 'cas0106-0279']
- gold_papers：['1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 'LAT1 ChemComm 2026', 's41421-024-00697-6']
- correctness candidate：`answer_for_judge`
- correctness：0.75　raw：4/5　judge：`structured_fact_audit_v1`
- judge reason：covered 6/7; missing F4
- translation fidelity：0.5　raw：3/5　judge：`translation_fidelity_v2`
- translation reason：1 material and 0 minor semantic errors; The source specifies 'conformational changes in LAT1' (changes occurring within the protein), while the target translates this as 'in LAT1 induces... conformational changes', which grammatically implies the changes are located inside the molecule but loses the explicit prepositional phrase structure
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：57.1%　grounding：0.857
- 延遲：842.4s　問題標記：{'conflicts': 0, 'unsupported': 1}

**答案預覽**：

> Comparison scaffold:
> - 策略 (Strategy)：【1-s2.0-S1936523321002047-main】報告了 JPH203 與放射治療 (radiation therapy) 的聯合應用，透過 mTOR 下調 (downregulation) 和細胞衰老 (cellular senescence) 增強放射敏感性 [1-s2.0-S1936523321002047-main]。
> - 策略 (Strategy)：【LAT1 ChemComm 2026】報告了靶向 LAT1 的自組裝肽 (self-assembling peptide) LffVLKK-4Phe，透過多價交互作用 (multivalent interaction) 及隨後的膜破壞 (membrane disruption) 抑制癌細胞增殖 [LAT1 ChemComm 2026]。
> - 策略 (Strategy)：【cas0106-0279】報告了用於 BNCT 的 p-硼苯丙氨酸 (p-Boronophenylalanine, BPA) 的選擇性遞送，使 10B 定位於惡性細胞，以便在中子照射 (neutron irradiation) 時發生核捕獲與裂變反應 [cas0106-0279]。
> - 機制 (Mechanism)：【s41421-024-00697-6】報告 JPH203 的結合由氫鍵網絡 (hydrogen bond network)、與 Tyr259 的鹵鍵 (halogen bond) 以及尾部基團 (tail moiety) 的疏水交互作用 (hydrophobic interactions) 所穩定 [s41421-024-00697-6]。
> - 機制 (Mechanism)：【1-s2.0-S1347861320300633-main】報告 JPH20 …（完整內容見 JSON）

---

### ✅ Q10 · contradiction_check

**問題**：Do the papers agree on the hydrolytic stability of boroxine structures? Describe any agreements or differences between them.

- detected_paper：`None`
- 選出論文：['Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation', '41467_2024_Article_45464', 'a-method-for-the-deprotection-of-alkylpinacolyl-boronate-esters']
- gold_papers：['41467_2024_Article_45464', 'Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation']
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 4/4
- translation fidelity：1.0　raw：5/5　judge：`translation_fidelity_v2`
- translation reason：0 material and 0 minor semantic errors
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：75.0%　grounding：1.0
- 延遲：523.8s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> 比較框架：
> - 策略：【Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation】報導將三個硼酸 (boronic acid) 單元併入一個柔性大環 (flexible macrocycle) 中以進行分子內硼氧環 (intramolecular boroxine) 的形成，由於熵因子 (entropic factors) 和路易斯酸度 (Lewis acidity) 的降低，提高了對水解的穩定性【Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation】。
> - 策略：【41467_2024_Article_45464】報導了 2-羥基苯硼酸 (2-hydroxyphenylboronic acid) 的自發脫水及其隨後接觸水的過程，形成了具有優異 pH 穩定性和水兼容動態共價鍵 (water-compatible dynamic covalent bonds) 的硼氧環 (boroxine) 結構【41467_2024_Article_45464】。
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
