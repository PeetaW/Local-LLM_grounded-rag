# Eval Report — `q07_q08_q09_cost_canonical_stability`

- 模式：Mode 2（對照 gold 真相）
- 產生時間：2026-08-01 22:39
- 題數：3

## 彙總

| 指標 | 值 |
|------|-----|
| 平均正確性（LLM-judge） | 1.0 |
| Correctness judge 覆蓋 | 3/3（N/A 0） |
| 平均翻譯忠實度（LLM-judge） | 1.0 |
| Translation judge 覆蓋 | 3/3（N/A 0） |
| 平均 grounding 分數 | 1.0 |
| 平均論文選擇命中率 | 100.0% |
| 平均 retriever candidate 覆蓋率 | 88.9% |
| 平均 Stage 2 evidence 覆蓋率 | 69.0% |
| 平均總延遲 | 547.3s |
| 平均 planning 延遲 | 29.4s |
| 平均 retrieval 延遲 | 6.7s |
| └ Phase A embed/vector/BM25 | 6.6s |
| └ Phase B 子答案生成 | 0.1s |
| 平均 grounding 延遲 | 17.0s |
| └ 其中 NLI | 1.0s |
| └ 其中 gemma4 | 0.0s |

## 逐題速覽

| | ID | 類型 | correctness | 翻譯忠實度 | 選擇命中 | candidate recall | Stage 2 recall | grounding | 延遲 | 衝突/未支撐 |
|---|----|------|-------------|------------|---------|------------------|----------------|-----------|------|------|
| ⚠️ | Q07 | figure_dependent | 1.0 | 1.0 | 100.0% | 66.7% | 50.0% | 1.0 | 455.4s | C0/U0 |
| ✅ | Q08 | cross_paper | 1.0 | 1.0 | 100.0% | 100.0% | 100.0% | 1.0 | 546.3s | C0/U0 |
| ⚠️ | Q09 | cross_paper | 1.0 | 1.0 | 100.0% | 100.0% | 57.1% | 1.0 | 640.2s | C0/U0 |

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
- 延遲：455.4s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ## 【論文直接依據】
>
> - 討論：總結而言，我們發現了一種水穩定的 boroxine 結構，具有優異的 pH 穩定性以及能在室溫下進行交換的水兼容動態共價鍵 (DCBs)，從根本上解決了 boroxines 長期以來的水解不穩定性挑戰 【Source: 41467_2024_Article_45464】。
> - 此外，我們開發了一種由水穩定的 boroxines 交聯的水凝膠 (hydrogel)，由於 boroxine 結構優異的 pH 穩定性和動態特性，該水凝膠表現出高酸鹼穩定性和可逆的凝膠-溶膠轉變 (gel–sol transition) 【Source: 41467_2024_Article_45464】。
> - 我們發現，在環境條件下，2-hydroxyphenylboronic acid 會發生自發脫水，形成具有動態共價鍵和聚集誘導增強發光活性 (aggregation-induced enhanced emission activity) 的二聚體 (dimer) 【Source: 41467_2024_Article_45464】。
> - 更有趣的是，在接觸水後，HO‑PBA 二聚體 (dimers) 在室溫下迅速轉化為 HO‑PBA 三聚體-H2O 複合物 (HO‑PBA trimer–H2O complexes)（圖 1b），其中包含水穩定的 boroxine 結構，這一點已通過核磁共振 (NMR) 光譜、質譜 (MS)、UV–Raman 光譜以及理論計算得到證實 【Source: 41467_2024_Article_45464】。
> - 這一推測得到了以下事實的支持：如圖 7b 所示，該水凝膠在酸性 (pH=2) 和鹼性 (pH=10) 溶液中均保持穩定 【Source: 41467_2024_Article_45464】。
> - 這些發現使得 borox …（完整內容見 JSON）

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
- 延遲：546.3s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> ⚠️ **資料可能不足**：檢索內容僅部分涵蓋此問題，以下回答可能不完整，請謹慎參考並自行查證。
>
> Comparison scaffold：
> - Route：【bbb0683】報導了對映選擇性烷基化 (enantioselective alkylation) 以及隨後的酶促水解 (enzymatic hydrolysis)，從而產生光學純的 L-BPA 【bbb0683】。
> - Route detail：使用 chymotrypsin 進行水解，以提供光學純的 L-BPA 【bbb0683】。
> - Review/comparison source：L-BPA 的合成已透過多種路徑進行，反映出在生產高純度、同位素富集 (isotopically enriched) 材料方面的挑戰 【CMDC-20-e202500059】。
> - Review dimensions：該評論強調了每種方法在可擴展性 (scalability)、成本效益 (cost-effectiveness) 和安全性方面的局限性 【CMDC-20-e202500059】。
>
> Central trade-off (high-purity/isotopic enrichment versus scalability and cost-effectiveness)：
> - High-purity/isotopic enrichment：在 L-BPA 合成中，生產高純度、同位素富集 (isotopically enriched) 的材料是一項挑戰 【CMDC-20-e202500059】。
> - Scalability：雜交合成法 (hybrid synthesis method) 在操作與後處理的便捷性以及反應步驟較少方面具有優勢 【bbb0683】。
> - Cost-effectiveness：主要成本通常來自於同位素 …（完整內容見 JSON）

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
- 延遲：640.2s　問題標記：{'conflicts': 0, 'unsupported': 0}

**答案預覽**：

> 比較框架：
> - 策略：【1-s2.0-S1347861320300633-main】報告了 JPH203 單一療法 (monotherapy)，透過協同的預孵育 (preincubation) 與共同孵育 (co-incubation) 效應抑制 LAT1 功能 【1-s2.0-S1347861320300633-main】。
> - 策略：【1-s2.0-S1936523321002047-main】報告了 JPH203 結合放射治療 (radiation therapy)，可增強放射敏感性 (radiosensitivity) 與細胞衰老 (cellular senescence) 【1-s2.0-S1936523321002047-main】。
> - 策略：【LAT1 ChemComm 2026】報告了 LffVLKK-4Phe 自組裝肽 (self-assembling peptide) 治療，透過多價 LAT1 結合與運輸抑制來抑制增殖 【LAT1 ChemComm 2026】。
> - 策略依據：這可能與顯示 JPH203 作為非底物阻斷劑 (non-substrate blocker) 的報告相關，儘管作者表示其抑制模式為競爭性 (competitive) 【1-s2.0-S1347861320300633-main】。
> - 策略依據：我們證明 X 射線照射 (X-irradiation) 增加了 A549 和 MIA Paca-2 細胞中透過 LAT1 的細胞中性氨基酸攝取 (neutral amino acid uptake)，且 LAT1 抑制劑 JPH203 在極低毒性濃度 (minimally toxic concentrations) 下，能顯著增加癌細胞對放射線的敏感性 【1-s2.0-S1936523321002047-main】。
> - 策略依據：我們設 …（完整內容見 JSON）

---
