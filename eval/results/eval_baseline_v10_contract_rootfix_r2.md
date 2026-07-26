# Eval Report — `baseline_v10_contract_rootfix_r2`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-24 08:45
- 題數：5

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.9 |
| Correctness judge 覆蓋 | 5/5（N/A 0） |
| 平均翻譯忠實度（LLM-judge） | 0.9 |
| Translation judge 覆蓋 | 5/5（N/A 0） |
| 平均 grounding 分數 | 0.894 |
| 平均論文選擇命中率 | 100.0% |
| 平均 retriever candidate 覆蓋率 | 93.3% |
| 平均 Stage 2 evidence 覆蓋率 | 74.8% |
| 平均總延遲 | 542.9s |
| 平均 planning 延遲 | 31.5s |
| 平均 retrieval 延遲 | 5.9s |
| └ Phase A embed/vector/BM25 | 5.8s |
| └ Phase B 子答案生成 | 0.0s |
| 平均 grounding 延遲 | 16.4s |
| └ 其中 NLI | 1.1s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | correctness | 翻譯忠實度 | 選擇命中 | candidate recall | Stage 2 recall | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|-------------|------------|---------|------------------|----------------|-----------|------|------|
| ⚠️ | Q02 | single_paper | 0.75 | 0.5 | 100.0% | 100.0% | 100.0% | 0.857 | 421.6s | C0/U1 |
| ✅ | Q03 | figure_dependent | 1.0 | 1.0 | 100.0% | 100.0% | 100.0% | 1.0 | 389.2s | C0/U0 |
| ⚠️ | Q07 | figure_dependent | 0.75 | 1.0 | 100.0% | 66.7% | 50.0% | 1.0 | 466.4s | C0/U0 |
| ⚠️ | Q08 | cross_paper | 1.0 | 1.0 | 100.0% | 100.0% | 66.7% | 0.833 | 766.3s | C0/U1 |
| ⚠️ | Q09 | cross_paper | 1.0 | 1.0 | 100.0% | 100.0% | 57.1% | 0.778 | 670.9s | C0/U2 |

## 逐題細節

### ⚠️ Q02 · single_paper

**問題**：What is the hybrid process used for the asymmetric synthesis of optically pure L-p-boronophenylalanine, and what are its key steps?

- detected_paper：`None`
- 選出論文：['bbb0683', 'CMDC-20-e202500059']
- gold_papers：['bbb0683']
- correctness candidate：`answer_for_judge`
- correctness：0.75　raw：4/5　judge：`structured_fact_audit_v1`
- judge reason：covered 5/6; missing F3
- translation fidelity：0.5　raw：3/5　judge：`translation_fidelity_v2`
- translation reason：1 material and 0 minor semantic errors; The source text contains a garbled yield value '8()l~·~' which appears to be an OCR error for 80% (or similar). The target translation retains this exact garbage string instead of correcting it or omitting the clearly corrupted data, resulting in a nonsensical scientific claim. Additionally, the tar
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：100.0%　grounding：0.857
- 延遲：421.6s　問題標記：{'conflicts': 0, 'unsupported': 1}

**答案預覽**：

> Method evidence:
> - 鋰化的 (lithiated) (2R)-( - )-2,5-dihydro- 2-isopropyl-3,6-dimethoxypyrazine 與 4-bromo- methylbenzeneboronate (2) 反應，以 74% e.e 產出加成物 4 (adduct 4) 【bbb0683】。
> - 我們對 L-BPA 的合成基於 (2R)-( - )-2,5-dihydro-2-isopropyl-3,6-dimethoxypyrazine (bislactimether, 3) 與受保護的 4-bromomethylbenzeneboronic acid 衍生物 2 的對映選擇性烷基化 (enantioselective alkylation)，以及隨後的酶水解 (enzymatic hydrolysis) 【bbb0683】。
> - 將 4 分步處理 (stepwise treatment) 以鹽酸 (hydrochloric acid) 得到 L-BPA methyl ester 6，接著使用 chymotrypsin 水解以提供光學純的 (optically pure) L-BPA 【bbb0683】。
> - 總結而言，透過從市售的手性輔助劑 (chiral auxiliary) 開始，結合對映選擇性烷基化 (enantioselective alkylation) 與酶水解 (enzymatic hydrolysis)，在少數步驟內實現了光學純 L-BPA 的不對稱合成 (asymmetric synthesis) 【bbb0683】。
> - 用 0.1 N 鹽酸 (hydrochloric acid) 處理 4 可得到 D-valine methyl ester (D-Val-OMe) 和具有期望立體化學 (desir …（完整內容見 JSON）

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
- 延遲：389.2s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
>
> - 具體而言，尾端 (tail) 的氨基 (amino group) 與 TM10 上 Phe400 的主鏈 (main chain) 形成氫鍵 (hydrogen bond) [Source: s41421-024-00697-6]。
> - 此外，尾端基團 (tail group) 的苯環 (benzene ring) 與 Phe252 的側鏈 (side chain) 產生 T 型 π–π 交互作用 (T-shaped π–π interaction) (Fig. 1c) [Source: s41421-024-00697-6]。
> - JPH203 延伸的尾端 (elongated tails) 似乎能防止 Phe400 的移動，導致其產生的構象變化 (conformational changes) 與 Diiodo-Tyr 或 2-amino-2-norbornanecarboxylic acid (BCH) 所誘導的變化截然不同 [Source: s41421-024-00697-6]。
> - 在我們的結構分析 (structural analysis) 中，我們觀察到 JPH203 結合在傳統的底物結合口袋 (substrate-binding pocket) 內，與先前確定的 LAT1 抑制劑 (inhibitors) 如 Diiodo-Tyr 或 JX-075, JX-078 以及 JX-119 相似 (Fig. 1b, c)13 [Source: s41421-024-00697-6]。
> - 頭端 (head) 的 $\alpha$-氨基 ($\alpha$-amino group) 和 $\alpha$-羧基 ($\alpha$-carboxyl group) 在穩定分子方面起著關鍵作用，與 LAT1 中 TM1 和 TM6 解旋區域 …（完整內容見 JSON）

---

### ⚠️ Q07 · figure_dependent

**問題**：What is the water-stable boroxine structure reported, and what role do the dynamic covalent bonds play in its fluoride binding and hydrogel formation?

- detected_paper：`None`
- 選出論文：['41467_2024_Article_45464']
- gold_papers：['41467_2024_Article_45464']
- correctness candidate：`answer_for_judge`
- correctness：0.75　raw：4/5　judge：`structured_fact_audit_v1`
- judge reason：covered 7/8; missing F3
- translation fidelity：1.0　raw：5/5　judge：`translation_fidelity_v2`
- translation reason：0 material and 0 minor semantic errors
- 論文選擇命中率：100.0%　candidate 覆蓋率：66.7%　Stage 2 evidence 覆蓋率：50.0%　grounding：1.0
- 延遲：466.4s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
>
> - 結果的 1H NMR 光譜清楚地表明，HO‑PBA dimer 在室溫下已完全轉化為 trimer，且 1H NMR、ESI-Q-TOF MS 和 UV–Raman 的測量結果顯示，在 7 天的研究期間，HO‑PBA trimers 在環境條件下保持穩定 (Supplementary Fig. 12) [Source: 41467_2024_Article_45464]。
> - 在此，我們揭示 HO‑PBA trimer 中的 B–O 鍵是水兼容的 DCBs (動態共價鍵)，為 DCBs 家族提供了一個重要的成員 [Source: 41467_2024_Article_45464]。
> - 值得注意的是，這種水穩定的 boroxine 結構在寬 pH 範圍內保持穩定，且其 B–O 鍵具有動態性，可以在室溫下的水相介質 (aqueous media) 中快速交換 [Source: 41467_2024_Article_45464]。
> - 有趣的是，在接觸水後，dimer 會迅速轉化為具有優異 pH 穩定性和水兼容動態共價鍵 (water-compatible dynamic covalent bonds) 的 boroxine 結構 [Source: 41467_2024_Article_45464]。
> - 討論：總結來說，我們發現了一種水穩定的 boroxine 結構，具有優異的 pH 穩定性和能在室溫下進行交換的水兼容 DCBs，從根本上解決了 boroxines 長期以來的水解不穩定性 (hydrolytic instability) 挑戰 [Source: 41467_2024_Article_45464]。
> - 更有趣的是，在接觸水後，HO‑PBA dimers 在室溫下迅速轉化為 HO‑PBA trimer–H2O complexes ( …（完整內容見 JSON）

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
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：66.7%　grounding：0.833
- 延遲：766.3s　問題標記：{'conflicts': 0, 'unsupported': 1}

**答案預覽**：

> ⚠️ **資料可能不足**：檢索內容僅部分涵蓋此問題，以下回答可能不完整，請謹慎參考並自行查證。
>
> 比較框架：
> - 路徑 (Route)：【bbb0683】報告了一種 hybrid process involving enantioselective alkylation and subsequent enzymatic hydrolysis（涉及對映選擇性烷基化及隨後酶水解的混合製程），從而產生光學純 L-BPA 【bbb0683】。
> - 評論/比較來源 (Review/comparison source)：【CMDC-20-e202500059】報告指出，目前尚無製造該物質的共識方法 (consensus approach)——L-BPA 的合成已透過多種路徑進行，反映了生產高純度、同位素富集材料 (high-purity, isotopically enriched material) 的挑戰 【CMDC-20-e202500059】。
> - 評論維度 (Review dimensions)：該評論強調了每種方法在可擴展性 (scalability)、成本效益 (cost-effectiveness) 和安全性 (safety) 方面的局限性 【CMDC-20-e202500059】。
>
> 核心權衡（高純度/同位素富集與可擴展性及成本效益）：
> - 高純度/同位素富集：生產高純度、10B 同位素富集材料是 L-BPA 合成中的主要挑戰 【CMDC-20-e202500059】。
> - 可擴展性 (Scalability)：大規模使用任何氧化劑 (oxidant) 本質上都存在製程安全風險 (process safety risk) 【CMDC-20-e202500059】。
> - 成本效益 (Cost-effectiveness)：在製備同位素富集化合物 (isotopical …（完整內容見 JSON）

---

### ⚠️ Q09 · cross_paper

**問題**：Several papers target LAT1 for cancer therapy. What different therapeutic strategies targeting LAT1 are described across the literature, and how do they differ in mechanism?

- detected_paper：`None`
- 選出論文：['LAT1 ChemComm 2026', '1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 's41421-024-00697-6', 'cas0106-0279']
- gold_papers：['1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 'LAT1 ChemComm 2026', 's41421-024-00697-6']
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 7/7
- translation fidelity：1.0　raw：5/5　judge：`translation_fidelity_v2`
- translation reason：0 material and 0 minor semantic errors
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：57.1%　grounding：0.778
- 延遲：670.9s　問題標記：{'conflicts': 0, 'unsupported': 2}

**答案預覽**：

> 比較框架：
> - 策略：【1-s2.0-S1347861320300633-main】報告了 JPH203 對 LAT1 功能的抑制，其協同抑制作用（Synergistic inhibition）是透過預孵育（preincubation）與共同孵育（co-incubation）效應實現的 [1-s2.0-S1347861320300633-main]。
> - 策略：【1-s2.0-S1936523321002047-main】報告了 JPH203 與放射治療（radiation therapy）的聯合應用，透過 mTOR 下調（downregulation）與細胞衰老（cellular senescence）來增強放射敏感性（radiosensitivity） [1-s2.0-S1936523321002047-main]。
> - 策略：【LAT1 ChemComm 2026】報告了 LffVLKK-4Phe 自組裝肽（self-assembling peptide）處理，透過多價交互作用（multivalent interaction）與膜破壞（membrane disruption）來抑制增殖 [LAT1 ChemComm 2026]。
> - 策略：【cas0106-0279】報告了用於 BNCT 的 p-Boronophenylalanine (BPA) 遞送，其特點是 10B 在惡性細胞中選擇性累積，隨後進行中子照射裂變反應（neutron irradiation fission reactions） [cas0106-0279]。
> - 機制：【s41421-024-00697-6】報告了 JPH203 的結構結合涉及與 TM1 和 TM6 的氫鍵網絡（hydrogen bond network）、與 Tyr259 的鹵鍵（halogen bond），以及在由 Ile63, …（完整內容見 JSON）

---
