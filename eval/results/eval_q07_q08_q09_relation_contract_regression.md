# Eval Report — `q07_q08_q09_relation_contract_regression`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-08-01 19:32
- 題數：3

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 1.0 |
| Correctness judge 覆蓋 | 3/3（N/A 0） |
| 平均翻譯忠實度（LLM-judge） | 1.0 |
| Translation judge 覆蓋 | 3/3（N/A 0） |
| 平均 grounding 分數 | 0.952 |
| 平均論文選擇命中率 | 100.0% |
| 平均 retriever candidate 覆蓋率 | 88.9% |
| 平均 Stage 2 evidence 覆蓋率 | 69.0% |
| 平均總延遲 | 538.0s |
| 平均 planning 延遲 | 28.9s |
| 平均 retrieval 延遲 | 6.6s |
| └ Phase A embed/vector/BM25 | 6.5s |
| └ Phase B 子答案生成 | 0.0s |
| 平均 grounding 延遲 | 21.1s |
| └ 其中 NLI | 1.1s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | correctness | 翻譯忠實度 | 選擇命中 | candidate recall | Stage 2 recall | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|-------------|------------|---------|------------------|----------------|-----------|------|------|
| ⚠️ | Q07 | figure_dependent | 1.0 | 1.0 | 100.0% | 66.7% | 50.0% | 1.0 | 441.8s | C0/U0 |
| ✅ | Q08 | cross_paper | 1.0 | 1.0 | 100.0% | 100.0% | 100.0% | 0.857 | 514.5s | C0/U1 |
| ⚠️ | Q09 | cross_paper | 1.0 | 1.0 | 100.0% | 100.0% | 57.1% | 1.0 | 657.8s | C0/U0 |

## 逐題細節

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
- 延遲：441.8s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
>
> - 討論：總結而言，我們發現了一種水穩定的 boroxine (boroxine) 結構，具有優異的 pH 穩定性以及能在室溫下進行交換的水兼容 DCBs (water-compatible DCBs, 水兼容的動態共價鍵)，從根本上解決了 boroxines 長期以來的水解不穩定性挑戰 【41467_2024_Article_45464】。
> - 此外，我們開發了一種由水穩定 boroxines 交聯的水凝膠 (hydrogel)，由於 boroxine 結構優異的 pH 穩定性和動態特性，該水凝膠表現出高酸鹼穩定性和可逆的凝膠-溶膠轉變 (reversible gel–sol transition) 【41467_2024_Article_45464】。
> - 我們發現，在環境條件下，2-hydroxyphenylboronic acid 會發生自發性脫水，形成具有動態共價鍵 (dynamic covalent bonds) 和聚集誘導增強發光活性 (aggregation-induced enhanced emission activity) 的二聚體 (dimer) 【41467_2024_Article_45464】。
> - 更有趣的是，在接觸水後，HO‑PBA 二聚體在室溫下迅速轉化為 HO‑PBA 三聚體-H2O 複合物 (HO‑PBA trimer–H2O complexes)（圖 1b），其中包含水穩定的 boroxine 結構，這一點已通過核磁共振 (NMR) 光譜、質譜 (MS)、UV–Raman 光譜以及理論計算得到證實 【41467_2024_Article_45464】。
> - 這一推測得到了以下事實的支持：如圖 7b 所示，該水凝膠在酸性 (pH=2) 和鹼性 (pH=10) 溶液中均保持穩定 【41467_2024_Arti …（完整內容見 JSON）

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
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：100.0%　grounding：0.857
- 延遲：514.5s　問題標記：{'conflicts': 0, 'unsupported': 1}

**答案預覽**：

> ⚠️ **資料可能不足**：檢索內容僅部分涵蓋此問題，以下回答可能不完整，請謹慎參考並自行查證。
>
> 比較框架：
> - 路線：【bbb0683】報導了一種涉及對映選擇性烷基化 (enantioselective alkylation) 及隨後酶水解 (enzymatic hydrolysis) 的混合製程，產出光學純的 L-BPA 【bbb0683】。
> - 路線詳情：使用 chymotrypsin 水解以提供光學純的 L-BPA 【bbb0683】。
> - 評論/比較來源：儘管其結構看似簡單，但目前尚無共識的製備方法——L-BPA 的合成已透過多種路線進行，反映出在生產高純度、同位素富集 (isotopically enriched) 材料方面的挑戰 【CMDC-20-e202500059】。
> - 評論維度：該評論強調了每種方法在可擴展性 (scalability)、成本效益 (cost-effectiveness) 和安全性方面的局限性 【CMDC-20-e202500059】。
>
> 核心權衡（高純度/同位素富集與可擴展性及成本效益）：
> - 高純度/同位素富集：L-BPA 的合成因生產高純度、同位素富集 (isotopically enriched) 材料的挑戰而變得複雜 【CMDC-20-e202500059】。
> - 可擴展性 (Scalability)：在大規模生產中使用任何氧化劑 (oxidant) 本質上都存在製程安全風險 【CMDC-20-e202500059】。
> - 成本效益 (Cost-effectiveness)：與同位素富集的 10B 相關的成本很高 【CMDC-20-e202500059】。
>
> ---
> 📋 **答案品質報告**
>
> ✅ **整體論文依據率**：85.7%　高（答案高度忠實於論文內容）
>
> ⚠️  **以下 1 個陳述未找到明確論文依據，請謹慎參考： …（完整內容見 JSON）

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
- 論文選擇命中率：100.0%　candidate 覆蓋率：100.0%　Stage 2 evidence 覆蓋率：57.1%　grounding：1.0
- 延遲：657.8s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> 比較框架：
> - 策略：【1-s2.0-S1347861320300633-main】報告了 JPH203 單一療法 (monotherapy)（預孵育 preincubation + 共孵育 co-incubation），具有對 LAT1 功能的協同抑制 (synergistic inhibition) 【1-s2.0-S1347861320300633-main】。
> - 策略：【1-s2.0-S1936523321002047-main】報告了 JPH203 結合放射治療 (radiation therapy)，透過 mTOR 下調 (downregulation) 與細胞衰老 (cellular senescence) 增強放射敏感性 (radiosensitivity) 【1-s2.0-S1936523321002047-main】。
> - 策略：【LAT1 ChemComm 2026】報告了 LffVLKK-4Phe 自組裝肽 (self-assembling peptide) 治療，透過多價 LAT1 結合與運輸抑制來抑制細胞增殖 (cell proliferation) 【LAT1 ChemComm 2026】。
> - 策略依據：這可能與顯示 JPH203 作為非底物阻斷劑 (non-substrate blocker) 的報告相關，儘管作者表示其抑制模式為競爭性 (competitive) 【1-s2.0-S1347861320300633-main】。
> - 策略依據：我們證明了 X 射線照射 (X-irradiation) 增加了 A549 和 MIA Paca-2 細胞中透過 LAT1 的細胞中性氨基酸攝取 (neutral amino acid uptake)，且 LAT1 抑制劑 JPH203 在最低毒性濃度 (minimally toxic concen …（完整內容見 JSON）

---
