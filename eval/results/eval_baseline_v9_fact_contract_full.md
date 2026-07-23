# Eval Report — `baseline_v9_fact_contract_full`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-07-23 09:43
- 題數：12

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 0.646 |
| Correctness judge 覆蓋 | 12/12（N/A 0） |
| 平均翻譯忠實度（LLM-judge） | 0.8 |
| Translation judge 覆蓋 | 10/12（N/A 2） |
| 平均 grounding 分數 | 0.908 |
| 平均論文選擇命中率 | 100.0% |
| 平均 retriever candidate 覆蓋率 | 94.7% |
| 平均 Stage 2 evidence 覆蓋率 | 75.4% |
| 平均總延遲 | 397.6s |
| 平均 planning 延遲 | 6.3s |
| 平均 retrieval 延遲 | 6.0s |
| └ Phase A embed/vector/BM25 | 6.0s |
| └ Phase B 子答案生成 | 0.0s |
| 平均 grounding 延遲 | 14.1s |
| └ 其中 NLI | 1.2s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | correctness | 翻譯忠實度 | 選擇命中 | candidate recall | Stage 2 recall | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|-------------|------------|---------|------------------|----------------|-----------|------|------|
| ✅ | Q01 | single_paper | 1.0 | 1.0 | 100.0% | 100.0% | 100.0% | 1.0 | 289.8s | C0/U0 |
| ❌ | Q02 | single_paper | 0.25 | 1.0 | 100.0% | 100.0% | 75.0% | 1.0 | 391.1s | C0/U0 |
| ❌ | Q03 | figure_dependent | 0.25 | 1.0 | 100.0% | 100.0% | 100.0% | 1.0 | 277.1s | C0/U0 |
| ✅ | Q04 | table_lookup | 1.0 | 1.0 | 100.0% | 100.0% | 100.0% | 1.0 | 334.2s | C0/U0 |
| ⚠️ | Q05 | single_paper | 0.5 | 0.5 | 100.0% | 100.0% | 50.0% | 1.0 | 339.4s | C0/U0 |
| ❌ | Q06 | multi_chunk | 0.25 | 1.0 | 100.0% | 80.0% | 80.0% | 1.0 | 263.8s | C0/U0 |
| ❌ | Q07 | figure_dependent | 0.25 | 0.5 | 100.0% | 66.7% | 50.0% | 1.0 | 324.2s | C0/U0 |
| ⚠️ | Q08 | cross_paper | 1.0 | 1.0 | 100.0% | 100.0% | 66.7% | 0.833 | 477.2s | C0/U1 |
| ⚠️ | Q09 | cross_paper | 1.0 | 0.5 | 100.0% | 100.0% | 57.1% | 0.75 | 1162.5s | C0/U3 |
| ⚠️ | Q10 | contradiction_check | 1.0 | 0.5 | 100.0% | 100.0% | 75.0% | 0.5 | 654.7s | C0/U2 |
| ✅ | Q11 | out_of_scope | 1.0 | N/A | N/A | N/A | N/A | — | 132.5s | C0/U0 |
| ❌ | Q12 | false_premise | 0.25 | N/A | N/A | N/A | N/A | — | 124.1s | C0/U0 |

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
- 延遲：289.8s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
>
> - 此外，preincubation 與 co-incubation 共同作用的 IC50 值低於單獨 co-incubation 抑制的效果 (34.2 ± 3.6 nM vs. 99.2 ± 11.0 nM) [Source: 1-s2.0-S1347861320300633-main]。
> - 這些結果表明，JPH203 對於 HT-29 細胞中 LAT1 功能的 preincubation 抑制效果呈濃度與時間依賴性 [Source: 1-s2.0-S1347861320300633-main]。
> - 根據結果，IC50 值被確定為 193 ± 50 (nM) [Source: 1-s2.0-S1347861320300633-main]。
> - 如圖 5 所示，雖然 co-incubation 抑制分析中的 IC50 值為 99.2 ± 11.0 nM，但加入 preincubation 顯著增強了其抑制效能 (IC50 = 34.2 ± 3.6 nM) [Source: 1-s2.0-S1347861320300633-main]。
> - 與 IC50 分析結果一致，pre-plus co-incubation 抑制分析中的 Ki 值為 0.37 mM，低於 co-incubation 抑制分析中的值 (0.46 mM) [Source: 1-s2.0-S1347861320300633-main]。
>
> ---
> 📋 **答案品質報告**
>
> 📊 **分段論文依據率：**
>
>   ✅ 【論文直接依據】：100.0%（5/5 句）
>
> ✅ **直引依據率**：100.0%　高（直接引用高度忠實於論文）
>
> ✅ **所有陳述均有論文依據**
>
>
> <!-- grounding_score=1.000 -->
> ---

---

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
- 延遲：391.1s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> 光學純的 L-p-boronophenylalanine 是透過一個包含對映選擇性烷基化 (enantioselective alkylation) 以及隨後的酶水解 (enzymatic hydrolysis) 的混合製程合成而得 【bbb0683】。此方法的關鍵步驟定義如下：對映選擇性烷基化 (enantioselective alkylation) 啟動合成過程 【bbb0683】。使用鹽酸 (hydrochloric acid) 的逐步處理得到了 L-BPA methyl ester 6 【bbb0683】。Hydrolysis with chymotrypsin（使用 chymotrypsin 的水解）獲得了光學純的 L-BPA 【bbb0683】。
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

### ❌ Q03 · figure_dependent

**問題**：According to the cryo-EM structure, how does JPH203 bind to and inhibit the LAT1-4F2hc complex?

- detected_paper：`None`
- 選出論文：['s41421-024-00697-6', '1-s2.0-S1347861320300633-main']
- gold_papers：['s41421-024-00697-6']
- correctness candidate：`answer_for_judge`
- correctness：0.25　raw：2/5　judge：`structured_fact_audit_v1`
- judge reason：covered 1/6; missing F1, F2, F4, F5, F6
- translation fidelity：1.0　raw：5/5　judge：`translation_fidelity_v2`
- translation reason：0 material and 0 minor semantic errors
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：100.0%　grounding：1.0
- 延遲：277.1s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
>
> - 此外，JPH203 的雙鹵化酪胺酸核心 (bi-halogenated tyrosine core) 上的氯原子與 Tyr259 形成了一個鹵鍵 (halogen bond) [Source: s41421-024-00697-6]。
> - 具體而言，其尾端的胺基 (amino group) 與 TM10 上 Phe400 的主鏈形成了一個氫鍵 (hydrogen bond) [Source: s41421-024-00697-6]。
> - 然而，在 JPH203 結合結構 (JPH203-bound structure) 中，TM10 的二級結構 (secondary structure) 被破壞，而 TM3 Received: 2 February 2024 Accepted: 31 May 2024 www.nature.com/celldisc Cell Discovery 1234567890();,: 則保持完全摺疊 (fully folded)，這表明其與 JX075 相比具有不同的抑制作用 [Source: s41421-024-00697-6]。
> - 看起來 JPH203 的延伸尾端 (elongated tails) 阻止了 Phe400 的移動，導致其產生的構象變化 (conformational changes) 與 Diiodo-Tyr 或 2-amino-2-norbornanecarboxylic acid (BCH) 所誘導的變化截然不同 [Source: s41421-024-00697-6]。
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
> ✅ **所有陳述均有論文依 …（完整內容見 JSON）

---

### ✅ Q04 · table_lookup

**問題**：What are the main degradation products and impurities of BPA identified by HPLC, and under which storage conditions do they form?

- detected_paper：`None`
- 選出論文：['1-s2.0-S0731708511003451-main', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water']
- gold_papers：['1-s2.0-S0731708511003451-main']
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 7/7
- translation fidelity：1.0　raw：5/5　judge：`translation_fidelity_v2`
- translation reason：0 material and 0 minor semantic errors
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：100.0%　grounding：1.0
- 延遲：334.2s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
>
> - BPA 與 tyrosine 的分離在 230 nm 處定量，保留時間 (retention times) 分別為 5.3 和 4.5 min [Source: 1-s2.0-S0731708511003451-main]。
> - Phenylalanine 在 11.0 min 洗脫 (elutes)，並在 256 nm 處定量 [Source: 1-s2.0-S0731708511003451-main]。
> - BrPD 與 FBBA 在 256 nm 處檢出，洗脫時間分別為 17.3 和 23.7 min [Source: 1-s2.0-S0731708511003451-main]。
> - BDPA 在 270 nm 處檢出，但與 FBBA 共洗脫 (co-elutes) [Source: 1-s2.0-S0731708511003451-main]。
> - 所有三種合成雜質 (synthetic impurities) 在 BPA 存在或不存在的情況下，在濃度為 0.5 µg/ml（或 BPA NWC 的 0.1%）時均可檢出；且在現有的 BPA 樣本中未觀察到 BrPD 或 FBBA/BDPA 組合 [Source: 1-s2.0-S0731708511003451-main]。
> - 降解 (Degradation)：BPA 作為原料粉末 (raw powder) 是穩定的，在 55 °C 下儲存 6 個月或在 40 °C 下儲存 12 個月時，未產生可檢出的降解 [Source: 1-s2.0-S0731708511003451-main]。
> - 亦觀察到 BPA 在酸性溶液和 FeCl3 溶液（後者用於測試金屬催化降解 metal-catalysed degradation）中具有穩定性 [Source: 1-s2.0-S073170 …（完整內容見 JSON）

---

### ⚠️ Q05 · single_paper

**問題**：What is the solvent-free green protocol for N-Boc protection of amines using picric acid as a catalyst, and what reaction conditions are used?

- detected_paper：`None`
- 選出論文：['1-s2.0-S2773223124000268-main']
- gold_papers：['1-s2.0-S2773223124000268-main']
- correctness candidate：`answer_for_judge`
- correctness：0.5　raw：3/5　judge：`structured_fact_audit_v1`
- judge reason：covered 5/7; missing F4, F6
- translation fidelity：0.5　raw：3/5　judge：`translation_fidelity_v2`
- translation reason：1 material and 0 minor semantic errors; The source specifies that TFA is used 'on a bench scale' for removal, while the target sentence omits this condition entirely when mentioning TFA. The phrase 'in 5–10 min at room temperature on a bench scale' in S8 applies to the TFA method specifically; T8 incorrectly generalizes or drops the scope
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：50.0%　grounding：1.0
- 延遲：339.4s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
>
> - 利用苦味酸 (picric acid) 作為溫和且有效的催化劑，此方法可在無溶劑條件下實現芳香胺 (aromatic amines)、雜環芳香胺 (heterocyclic aromatic amines)、開鏈脂肪胺 (open-chain aliphatic amines) 以及環狀脂肪胺 (cyclic aliphatic amines) 的 N-Boc 保護 [Source: 1-s2.0-S2773223124000268-main]。
> - 此方法利用二叔丁基二碳酸酯 (di-tert-butyl dicarbonate, (Boc)2O) 來保護芳香胺與脂肪胺，並以易於取得的苦味酸 (picric acid, 2 mol%) 作為 Brønsted 酸催化劑 (Brønsted acid catalyst) [Source: 1-s2.0-S2773223124000268-main]。
> - 基於這些篩選反應條件，我們得出結論：在室溫、無溶劑條件下，使用 0.2 當量 (equivalents) 的苦味酸 7 (picric acid 7) 作為催化劑並反應 60 min，是獲得產率極佳之產物的最佳條件（Table 1, Entry 13）[Source: 1-s2.0-S2773223124000268-main]。
> - 此方法可在完全無溶劑且環保的條件下，快速且高產率（最高達 98 %）地對胺類進行保護，從而促進更清潔且更永續的合成過程 [Source: 1-s2.0-S2773223124000268-main]。
> - 當反應在無催化劑的情況下進行時，48 h 後僅獲得 55 % 的產物（Table 1, Entry 23）[Source: 1-s2.0-S2773223124000268-main]。
> - 在高極性溶劑如 D …（完整內容見 JSON）

---

### ❌ Q06 · multi_chunk

**問題**：How does poly(vinyl alcohol) enhance the therapeutic effect of 4-L-boronophenylalanine in neutron capture therapy of thoracic tumors? Explain the proposed mechanism with supporting data.

- detected_paper：`None`
- 選出論文：['1-s2.0-S0378517325007926-main', 'synthesis-and-biological-properties-of-water-soluble-p-boronophenylalanine-derivatives-relationship-between-water', 'CMDC-20-e202500059']
- gold_papers：['1-s2.0-S0378517325007926-main']
- correctness candidate：`answer_for_judge`
- correctness：0.25　raw：2/5　judge：`structured_fact_audit_v1`
- judge reason：covered 3/7; missing F1, F2, F3, F6
- translation fidelity：1.0　raw：5/5　judge：`translation_fidelity_v2`
- translation reason：0 material and 0 minor semantic errors
- 論文選擇命中率：100.0%　candidate 覆蓋率：80.0%　Stage 2 evidence 覆蓋率：80.0%　grounding：1.0
- 延遲：263.8s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
>
> - 在此，我們報告由 poly(vinyl alcohol) (PVA) 與 4-L-boronophenylalanine (BPA)（BNCT 中唯一經臨床批准的藥物）組成的複合物，對小鼠體內模擬難治性 malignant pleural mesothelioma (MPM, 惡性胸膜間皮瘤) 的胸腔腫瘤可顯示出顯著的治療效率 [Source: 1-s2.0-S0378517325007926-main]。
> - 在這方面，我們之前報導了一項使用 poly(vinyl alcohol) (PVA) 來延長 BPA 之 intratumoral retention time (腫瘤內滯留時間) 的簡單技術 (Nomoto et al., 2020) [Source: 1-s2.0-S0378517325007926-main]。
> - 與此一致，即使在這種難治性模型中，PVA formulation (PVA 配方) 在 thermal neutron irradiation (熱中子照射) 下也顯著提高了生存率 [Source: 1-s2.0-S0378517325007926-main]。
> - 系統性給藥的 PVA-sorbitol-BPA 在目標腫瘤內顯示出高效的累積與長期滯留 [Source: 1-s2.0-S0378517325007926-main]。
> - 與傳統的 BPA 相比，使用 PVA-sorbitol-BPA 的 BNCT 顯著提高了生存率 [Source: 1-s2.0-S0378517325007926-main]。
>
> ---
> 📋 **答案品質報告**
>
> 📊 **分段論文依據率：**
>
>   ✅ 【論文直接依據】：100.0%（5/5 句）
>
> ✅ **直引依據率**：100.0%　高（直接引用高度忠實於論文）
>
> ✅ **所有陳 …（完整內容見 JSON）

---

### ❌ Q07 · figure_dependent

**問題**：What is the water-stable boroxine structure reported, and what role do the dynamic covalent bonds play in its fluoride binding and hydrogel formation?

- detected_paper：`None`
- 選出論文：['41467_2024_Article_45464', 'Chemistry A European J - 2023 - Ono - Structural Interconversion Based on Intramolecular Boroxine Formation']
- gold_papers：['41467_2024_Article_45464']
- correctness candidate：`answer_for_judge`
- correctness：0.25　raw：2/5　judge：`structured_fact_audit_v1`
- judge reason：covered 2/8; missing F1, F3, F5, F6, F7, F8
- translation fidelity：0.5　raw：3/5　judge：`translation_fidelity_v2`
- translation reason：1 material and 0 minor semantic errors; The source states the hydrogel is cross-linked by water-stable boroxines (developed from them), but T3 omits that these specific boroxines are 'water-stable' in the context of describing what they do to the gel, only mentioning it for the stability properties later. More critically, S3 says 'exhibit
- 論文選擇命中率：100.0%　candidate 覆蓋率：66.7%　Stage 2 evidence 覆蓋率：50.0%　grounding：1.0
- 延遲：324.2s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ⚠️ **資料可能不足**：檢索內容僅部分涵蓋此問題，以下回答可能不完整，請謹慎參考並自行查證。
>
> ## 【論文直接依據】
>
> - 有趣的是，在接觸水後，該二聚體 (dimer) 迅速轉化為具有優異 pH 穩定性和水兼容動態共價鍵 (water-compatible dynamic covalent bonds) 的 boroxine 結構 [Source: 41467_2024_Article_45464]。
> - 討論總結：我們發現了一種水穩定的 boroxine 結構，具有優異的 pH 穩定性和可在室溫下進行交換的水兼容 DCBs (dynamic covalent bonds)，從根本上解決了 boroxines 長期以來的水解不穩定性 (hydrolytic instability) 挑戰 [Source: 41467_2024_Article_45464]。
> - 此外，我們開發了一種由水穩定 boroxines 交聯的水凝膠 (hydrogel)，由於 boroxine 結構的優異 pH 穩定性和動態特性，該水凝膠表現出高酸鹼穩定性以及可逆的凝膠-溶膠轉變 (reversible gel–sol transition) [Source: 41467_2024_Article_45464]。
> - 在此，我們揭示了 HO‑PBA 三聚體 (trimer) 中的 B–O 鍵是水兼容的 DCBs，為 DCBs 家族提供了一個重要的成員 [Source: 41467_2024_Article_45464]。
> - 值得注意的是，這種水穩定的 boroxine 結構在寬廣的 pH 範圍內保持穩定，且其 B–O 鍵具有動態性，可以在室溫下的水相介質 (aqueous media) 中迅速交換 [Source: 41467_2024_Article_45464]。
> - 在此，我們報導 …（完整內容見 JSON）

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
- 延遲：477.2s　問題標記：{'conflicts': 0, 'unsupported': 1}

**答案預覽**：

> ⚠️ **資料可能不足**：檢索內容僅部分涵蓋此問題，以下回答可能不完整，請謹慎參考並自行查證。
>
> 比較框架：
> - 路徑：【bbb0683】報導了一種涉及對映選擇性烷基化 (enantioselective alkylation) 及隨後的酶促水解 (enzymatic hydrolysis) 的混合製程，產出光學純的 L-BPA 【bbb0683】。
> - 回顧/比較來源：【CMDC-20-e202500059】報導 4-borono-L-phenylalanine (L-BPA) 的合成已透過多種路徑實現 【CMDC-20-e202500059】。
> - 回顧維度：該回顧強調了每種方法在可擴展性 (scalability)、成本效益 (cost-effectiveness) 和安全性方面的局限性 【CMDC-20-e202500059】。
>
> 核心權衡（高純度/同位素富集與可擴展性及成本效益）：
> - 高純度/同位素富集：10B 通常透過化學交換蒸餾 (chemical exchange distillation) 從 11B 中分離，然後轉化為硼酸酯 (boronic esters) 或水解為 10B 硼酸 (10B boric acid) 【CMDC-20-e202500059】。
> - 可擴展性：該混合合成方法在操作與後處理 (workup) 的便捷性以及反應步驟較少方面，比已報導的方法具有優勢 【bbb0683】。
> - 成本效益：在製備同位素富集化合物時，主要成本通常來自於同位素起始原料；10B 的成本很高（增加至普通硼酸的 1000 倍以上） 【CMDC-20-e202500059】。
>
> ---
> 📋 **答案品質報告**
>
> ✅ **整體論文依據率**：83.3%　高（答案高度忠實於論文內容）
>
> ⚠️  **以下 1 個陳述未找到明確論文依據，請謹慎參考：**
>
> - Co …（完整內容見 JSON）

---

### ⚠️ Q09 · cross_paper

**問題**：Several papers target LAT1 for cancer therapy. What different therapeutic strategies targeting LAT1 are described across the literature, and how do they differ in mechanism?

- detected_paper：`None`
- 選出論文：['LAT1 ChemComm 2026', '1-s2.0-S0378517325007926-main', '1-s2.0-S1347861320300633-main', 's41421-024-00697-6', '1-s2.0-S1936523321002047-main']
- gold_papers：['1-s2.0-S1347861320300633-main', '1-s2.0-S1936523321002047-main', 'LAT1 ChemComm 2026', 's41421-024-00697-6']
- correctness candidate：`answer_for_judge`
- correctness：1.0　raw：5/5　judge：`structured_fact_audit_v1`
- judge reason：covered 7/7
- translation fidelity：0.5　raw：3/5　judge：`translation_fidelity_v2`
- translation reason：1 material and 0 minor semantic errors; Target T8 omits the mechanism detail 'followed by later-stage cell membrane disruption' (隨後在後期導致細胞膜破裂) which is present in Source S4 and correctly translated in Target T4.
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：57.1%　grounding：0.75
- 延遲：1162.5s　問題標記：{'conflicts': 0, 'unsupported': 3}

**答案預覽**：

> Comparison scaffold:
> *   **route**: PVA-BPA complex administration 【1-s2.0-S0378517325007926-main】 (PVA-BPA 複合物給藥): 通過保留類苯丙氨酸 (phenylalanine-like) 結構以利於攝取，經由 LAT1 定向增加 BPA 對胸腔腫瘤的治療效果。
> *   **route**: JPH203 inhibition (preincubation and co-incubation) 【1-s2.0-S1347861320300633-main】 (JPH203 抑制，包括預孵育與共同孵育): 通過協同效應抑制 LAT1 功能，進而抑制 HT-29 細胞中的 Na+-independent leucine uptake activity (非鈉依賴性亮氨酸攝取活性)。
> *   **route**: Combination of JPH203 and radiation therapy (X-irradiation) 【1-s2.0-S1936523321002047-main】 (JPH203 與放射治療/X 射線照射之結合): 通過抑制 LAT1 使癌細胞對放射線更敏感，導致 mTOR 下調並增強 cellular senescence (細胞衰老)。
> *   **route**: LffVLKK-4Phe self-assembling peptide treatment 【LAT1 ChemComm 2026】 (LffVLKK-4Phe 自組裝肽治療): 通過早期抑制 LAT1-mediated amino acid transport activity (LAT1 介導的氨基酸運輸活性)，隨後在後期導致細胞膜破裂，從而抑制 MCF-7 和 MDA-MB …（完整內容見 JSON）

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
- translation reason：1 material and 0 minor semantic errors; The source specifies the subject as 'Ono' (referring to a specific study or author), but T1 omits this subject, starting directly with '報導了...' without indicating who reported it.
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：75.0%　grounding：0.5
- 延遲：654.7s　問題標記：{'conflicts': 0, 'unsupported': 2}

**答案預覽**：

> 比較框架：
> - 策略 (Strategy)：【Ono】報導了分子內 boroxine 形成，且對水解具有極佳的穩定性 [Ono]。
> - 策略 (Strategy)：【41467_2024_Article_45464】報導了 2-hydroxyphenylboronic acid (HO-PBA) 的自發性脫水 (spontaneous dehydration) 以形成二聚體 (dimer)，隨後接觸水，具有水穩定 boroxine 結構，且具備優異的 pH 穩定性以及與水兼容的動態共價鍵 (water-compatible dynamic covalent bonds) [41467_2024_Article_45464]。
> - 機制 (Mechanism)：【Ono】報導抗水解能力 (hydrolytic resistance) 是透過熵優勢 (entropic advantage) 和降低的 Lewis 酸度 (Lewis acidity) 來實現的 [Ono]。
> - 核心權衡 (Central trade-off)：雖然 boroxines 通常以水解不穩定性 (hydrolytic instability) 為特徵，且容易分解為普通硼酸 (trivial boronic acids) (Ono, 41467_2024_Article_45464)，但穩定性可以透過利用熵穩定化 (entropic stabilization) 和降低的 Lewis 酸度來實現分子內 boroxine 形成 (Ono)，或者透過利用 2-hydroxyphenylboronic acid 來形成具有優異 pH 穩定性和與水兼容的動態共價鍵 (water-compatible dynamic covalent bonds) 的結構 (41467_2024_Article_45464)。
>  …（完整內容見 JSON）

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
- 延遲：132.5s　問題標記：{'conflicts': 0, 'unsupported': 0}

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
- 延遲：124.1s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ⚠️ **誠實棄答**：檢索到的內容與此問題相關，但並不包含可直接回答的資訊。為避免編造不存在的數據，本系統選擇不作答。建議換個問法，或確認該主題是否涵蓋於文獻庫中。

---
